import { mat4, vec4 } from 'gl-matrix';
import type { SceneCamera, Vector2 } from './litbox/scene.ts';
import type { LitboxScene } from './litbox/litbox_scene.ts';
import { SceneGraph } from './litbox/scene_graph.ts';
import { TextureCache } from './litbox/texture_cache.ts';
import { LightResources } from './litbox/light_resources.ts';
import { RaytracedResources } from './litbox/raytraced_resources.ts';
import { SimulationResources } from './litbox/simulation.ts';
import { SpriteResources } from './litbox/sprite_resources.ts';
import { TonemapResources } from './litbox/tonemap.ts';
import { RingBufferedUniform } from './litbox/ring_buffered_uniform.ts';
import { TransformResources } from './litbox/transform_resources.ts';
import { ComputedDataManager } from './litbox/computed_data_manager.ts';
import { DebugViewBlitResources, DEBUG_VIEW_MODE, type DebugView } from './litbox/debug_view.ts';

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
// viewProjection (mat4) + simInverseWorldTransform (mat4) + debugMode (f32, padded to 16
// bytes), matching CameraUniform in sprite.wgsl.
const CAMERA_UNIFORM_SIZE_BYTES = 4 * 16 * 2 + 16;
const FRAMES_IN_FLIGHT = 2;

interface ActiveCamera {
    camera: SceneCamera;
    worldTransform: mat4;
}

/**
 * Renders a Litbox Scene (see src/litbox/scene.ts): runs the (currently
 * stubbed) light simulation, then paints sprites layer-by-layer around an
 * additive composite of the simulation's HDR lightmap, into an offscreen
 * HDR frame buffer that's finally tonemapped to the canvas. See the
 * project's litbox_scene_renderer plan doc for the full architecture
 * rationale.
 *
 * WGSL gotcha (confirmed on a Pixel 10 Pro, both Chrome and Brave): dynamically
 * indexing a function-local array literal (`var x = array<T, N>(...); x[runtimeIndex]`)
 * can silently corrupt geometry/output on some mobile GPU drivers, with zero validation
 * error, exception, or device loss to catch it - see tonemap.wgsl's vertex_main for the
 * workaround (branching instead of indexing). Prefer buffer-backed data (storage/uniform/
 * workgroup) for anything indexed at runtime; reserve this rule for vertex, fragment, and
 * compute shaders alike, and confirm any new shader trick on real mobile hardware, not just
 * desktop.
 */
export class LitboxSceneRenderer {
    private canvas: HTMLCanvasElement;
    private adapter!: GPUAdapter;
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private presentationFormat!: GPUTextureFormat;
    private presentationSize!: [number, number];

    private hdrFrameTexture!: GPUTexture;
    private hdrFrameTextureView!: GPUTextureView;

    private cameraBindGroupLayout!: GPUBindGroupLayout;
    private cameraUniform!: RingBufferedUniform;

    private sceneGraph: SceneGraph | null = null;
    private textureCache!: TextureCache;
    private transformResources!: TransformResources;
    private lightResources!: LightResources;
    private computedDataManager!: ComputedDataManager;
    private raytracedResources!: RaytracedResources;
    private simulationResources!: SimulationResources;
    private spriteResources!: SpriteResources;
    private tonemapResources!: TonemapResources;
    private debugViewBlitResources!: DebugViewBlitResources;
    /** Named debug views (see debugView), populated once in createSharedResources - see debug_view.ts's DebugView doc for why each entry is a closure, not a captured GPUTextureView. */
    private debugViews = new Map<string, DebugView>();

    private activeScene: LitboxScene | null = null;
    private lastFrameTimeMs: number | null = null;

    /**
     * Diagnostic aid: when true, sprites render as flat, fully-opaque, shape-colored
     * quads, bypassing opacity/shading entirely - useful for confirming transforms/camera/
     * layering are correct independent of per-sprite opacity, image, and shape data.
     */
    public debugSolidColor = false;

    /**
     * When set (to a key registered in debugViews - currently 'albedo', 'density', 'normal',
     * 'roughness', all contributed by the raytraced G-Buffer, see createSharedResources), replaces
     * the entire normal render (simulation/sprites/tonemap) with a direct blit of that view's
     * source texture to the swapchain, transformed for actual legibility (see debug_view.wgsl) -
     * a diagnostic aid for verifying render-target contents before anything downstream consumes
     * them. The G-Buffer itself is still rendered every frame regardless
     * (raytracedResources.renderGBuffer runs unconditionally), so a G-Buffer-sourced view reflects
     * live scene changes. Set to null to return to normal rendering. An unknown key is treated the
     * same as null (silently falls through to normal rendering) rather than throwing, since this
     * is diagnostic-only.
     */
    public debugView: string | null = null;

    /**
     * Divisor applied before clamping to a displayable [0,1] range in whichever active debugView
     * mode consumes it (currently only the G-Buffer's 'density' view - see debug_view_blit.wgsl;
     * other modes ignore this). The "typical" value range depends entirely on the active scene and
     * which view is selected, neither of which is knowable in advance - tune this live rather than
     * guessing a fixed constant.
     */
    public debugViewScale = 0.5;

    /**
     * Exposure fed to the tonemap pass (see tonemap.wgsl). Seeded from the active camera's
     * own exposure value whenever the active camera changes (including first selection on
     * scene load) - see syncExposureToActiveCamera. Free to reassign afterwards (e.g. from
     * a UI slider) as a modifier on top of that starting point.
     */
    public exposureOverride: number | null = null;

    /** Owner id of the camera exposureOverride was last synced to - see syncExposureToActiveCamera. */
    private lastActiveCameraOwnerId: number | null = null;

    constructor(canvas?: HTMLCanvasElement) {
        this.canvas = canvas || document.createElement('canvas');
        if (!canvas) {
            document.body.appendChild(this.canvas);
        }
    }

    public async start(): Promise<void> {
        if (!await this.initWebGPU()) {
            return;
        }

        this.configureCanvas();
        this.createSharedResources();

        if (this.activeScene) {
            await this.rebuildFromScene();
        }

        this.render = this.render.bind(this);
        requestAnimationFrame(this.render);
    }

    /** Stages (or swaps in) a scene. Safe to call before or after start(). */
    public async setScene(scene: LitboxScene): Promise<void> {
        this.activeScene = scene;
        scene.onLoad(this);
        if (this.device) {
            await this.rebuildFromScene();
        }
    }

    /**
     * The canvas this renderer draws into. Exposed so a scene's onLoad can wire up DOM
     * interaction (e.g. click listeners) against it, converting event coordinates to world space
     * via screenToWorld - LitboxScene itself has no DOM access.
     */
    public getCanvas(): HTMLCanvasElement {
        return this.canvas;
    }

    private async initWebGPU(): Promise<boolean> {
        try {
            if (!navigator.gpu) {
                console.error("WebGPU not supported on this browser.");
                return false;
            }

            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.error("No appropriate GPUAdapter found.");
                return false;
            }
            this.adapter = adapter;
            // BC1-compressed textures (see TextureCache) need this feature enabled on the
            // device to be usable; request it opportunistically since not every GPU/browser
            // supports it (mobile GPUs typically don't) - TextureCache falls back gracefully
            // when it's absent.
            const requiredFeatures: GPUFeatureName[] = adapter.features.has('texture-compression-bc')
                ? ['texture-compression-bc']
                : [];
            this.device = await this.adapter.requestDevice({ requiredFeatures });
            this.device.addEventListener('uncapturederror', (event) => {
                console.error('WebGPU device error:', (event as GPUUncapturedErrorEvent).error.message);
            });
            this.device.lost.then((info) => {
                console.error(`WebGPU device lost (${info.reason}):`, info.message);
            });
        } catch (error) {
            console.error("Error initializing WebGPU:", error);
            return false;
        }
        return true;
    }

    private configureCanvas(): void {
        const context = this.canvas.getContext('webgpu');
        if (!context) {
            throw new Error("Could not get WebGPU context from canvas.");
        }
        this.context = context;
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.presentationSize = [this.canvas.width, this.canvas.height];
        this.context.configure({
            device: this.device,
            format: this.presentationFormat,
            alphaMode: 'premultiplied',
        });
    }

    private createSharedResources(): void {
        this.textureCache = new TextureCache(this.device);
        this.transformResources = new TransformResources(this.device);
        this.lightResources = new LightResources(this.device);
        this.computedDataManager = new ComputedDataManager(this.device);
        this.raytracedResources = new RaytracedResources(this.device, this.computedDataManager);
        this.simulationResources = new SimulationResources(this.device, this.computedDataManager);
        this.spriteResources = new SpriteResources(this.device);
        this.tonemapResources = new TonemapResources(this.device, this.presentationFormat);
        this.debugViewBlitResources = new DebugViewBlitResources(this.device, this.presentationFormat);

        // Shared by the sprite and simulation-composite pipelines. Ring-buffered because
        // it's the only per-frame-rewritten uniform in this renderer (everything else is
        // only rewritten on scene rebuilds).
        this.cameraBindGroupLayout = this.device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true } }],
        });
        this.cameraUniform = new RingBufferedUniform(this.device, this.cameraBindGroupLayout, CAMERA_UNIFORM_SIZE_BYTES, FRAMES_IN_FLIGHT);

        this.simulationResources.initialize(this.cameraBindGroupLayout);
        this.raytracedResources.initialize();
        this.spriteResources.initialize(this.cameraBindGroupLayout, HDR_FORMAT);

        // Named debug views this renderer can blit in place of the normal render (see
        // debugView) - currently all 4 come from the raytraced G-Buffer, but this map is the
        // generic extension point: any future *Resources class that owns a debug-worthy texture
        // just needs another entry here, no other renderer-level plumbing.
        this.debugViews.set('albedo', { getSourceView: () => this.raytracedResources.getAlbedoView(), mode: DEBUG_VIEW_MODE.PASSTHROUGH });
        this.debugViews.set('density', { getSourceView: () => this.raytracedResources.getDensityView(), mode: DEBUG_VIEW_MODE.DENSITY });
        this.debugViews.set('normal', { getSourceView: () => this.raytracedResources.getNormalRoughnessView(), mode: DEBUG_VIEW_MODE.NORMAL_REMAP });
        this.debugViews.set('roughness', { getSourceView: () => this.raytracedResources.getNormalRoughnessView(), mode: DEBUG_VIEW_MODE.ALPHA_AS_LUMINANCE });

        this.createHdrFrameTexture();
    }

    private createHdrFrameTexture(): void {
        this.hdrFrameTexture?.destroy();
        this.hdrFrameTexture = this.device.createTexture({
            size: this.presentationSize,
            format: HDR_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.hdrFrameTextureView = this.hdrFrameTexture.createView();
    }

    private async rebuildFromScene(): Promise<void> {
        if (!this.activeScene) {
            return;
        }
        const scene = this.activeScene.data;
        this.sceneGraph = new SceneGraph(scene);
        this.textureCache.loadScene(this.activeScene.baseUrl, scene.textureAtlasKeys);

        // Force the next getActiveCamera() call to resync exposureOverride: ownerIds are
        // scene-local, so a new scene's camera could coincidentally reuse the previous
        // scene's active camera's id and be mistaken for "the same camera, no change".
        this.lastActiveCameraOwnerId = null;

        this.lightResources.loadFromScene(scene, this.sceneGraph, this.transformResources);
        this.simulationResources.loadFromScene(scene, this.sceneGraph);
        await this.raytracedResources.loadFromScene(scene, this.sceneGraph, this.textureCache, this.simulationResources, this.transformResources);
        await this.spriteResources.loadFromScene(scene, this.sceneGraph, this.textureCache, this.simulationResources, this.transformResources);

        // rebuildFromScene runs outside the per-frame render() loop (from start()/setScene()),
        // so nothing else will flush these until the next render() call - without this, the
        // very first frame after a scene load would draw against unwritten GPU buffers.
        this.transformResources.flush();
        this.lightResources.flush();
        this.raytracedResources.flush();
        this.spriteResources.flush();
    }

    /**
     * Applies every structural change (create/destroy/reparent) queued by the active scene's
     * onFrame() this frame - see LitboxScene's createObject/destroyObject/reparentObject - to the
     * live SceneGraph and, where needed, the GPU resource managers. Runs before
     * applyDynamicSceneUpdates() so the SceneGraph is structurally up to date before that pass
     * consults it.
     */
    private applyPendingStructuralOps(): void {
        if (!this.activeScene || !this.sceneGraph) {
            return;
        }

        const sceneGraph = this.sceneGraph;
        const scene = this.activeScene.data;
        const ops = this.activeScene.getPendingStructuralOps();
        for (const op of ops) {
            if (op.type === 'create') {
                sceneGraph.addObject(op.object);
                if (op.sprite) {
                    void this.spriteResources.addSprite(op.sprite, sceneGraph, this.textureCache, this.transformResources);
                }
                if (op.raytraced) {
                    void this.raytracedResources.addRaytraced(op.raytraced, sceneGraph, this.textureCache, this.transformResources);
                }
                if (op.light && op.lightKind) {
                    this.lightResources.addLight(op.lightKind, op.light, sceneGraph, this.transformResources);
                }
            } else if (op.type === 'destroy') {
                const removed = sceneGraph.removeObject(op.rootId);
                // A whole-subtree destroy can remove an arbitrary number of lights at once
                // (unlike destroyLight's single light) - a full rebuild against the
                // already-filtered scene.data light arrays is simpler and matches this
                // renderer's pre-existing behavior for this case.
                this.lightResources.loadFromScene(scene, sceneGraph, this.transformResources);
                this.raytracedResources.removeByOwnerIds(new Set(removed), this.transformResources);
                this.spriteResources.removeByOwnerIds(new Set(removed), this.transformResources);
            } else if (op.type === 'destroySprite') {
                this.spriteResources.removeSprite(op.sprite, this.transformResources);
            } else if (op.type === 'destroyRaytraced') {
                this.raytracedResources.removeRaytraced(op.raytraced, this.transformResources);
            } else if (op.type === 'destroyLight') {
                this.lightResources.removeLight(op.light, this.transformResources);
            } else {
                sceneGraph.setParent(op.id, op.newParentId);
                this.refreshTransformCascade(op.id);
            }
        }

        if (ops.length > 0) {
            this.activeScene.clearPendingStructuralOps();
        }
    }

    /**
     * Invalidates `rootId`'s cached world transform (and every descendant's, since world
     * transforms are hierarchical) and pushes a targeted GPU refresh of just the transform-derived
     * data for each affected owner. Shared by applyDynamicSceneUpdates (per dynamic/dirty
     * transform) and applyPendingStructuralOps (for a reparent, which only changes transforms -
     * no owned sprite/light/raytraced data changes, so no other GPU-resource code is needed).
     *
     * Only one refreshTransform call per owner is needed regardless of how many
     * sprite/light/raytraced components it owns: transform data lives in one shared array (see
     * TransformResources), not duplicated per component. Also refreshes each sprite owner's
     * active-in-hierarchy cull flag alongside its transform, since SceneGraph invalidates and
     * re-derives both together (invalidateSubtree) - an owner's active state can change
     * (SceneObject.active toggled directly) without its transform changing, but this cascade is
     * the only thing that picks either up, matching this project's dynamic/dirty-marking
     * convention (see SpriteResources.refreshActiveState).
     */
    private refreshTransformCascade(rootId: number): number[] {
        if (!this.sceneGraph) {
            return [];
        }
        const sceneGraph = this.sceneGraph;
        sceneGraph.invalidateSubtree(rootId);
        const affectedIds = [rootId, ...sceneGraph.getDescendantIds(rootId)];
        for (const ownerId of affectedIds) {
            this.transformResources.refreshTransform(ownerId, sceneGraph);
            this.spriteResources.refreshActiveState(ownerId, sceneGraph);
            this.raytracedResources.refreshActiveState(ownerId, sceneGraph);
        }
        return affectedIds;
    }

    /**
     * Consults the active scene's dynamic/dirty flags (see LitboxScene) and pushes
     * targeted GPU updates for exactly the affected entries, instead of re-uploading
     * everything (wasteful) or nothing (broken for animation/interaction). Runs before
     * any GPU pass is recorded.
     *
     * Two independent things happen here: entries persistently marked dynamic (not
     * one-shot dirty) get their packed-array entry moved into that array's dynamic region
     * (markDynamic - idempotent, safe to call every frame even for an already-dynamic entry);
     * separately, every dynamic-or-dirty entry gets its GPU data actually refreshed
     * (writeEntry/refreshTransform/refreshProperties), staged into each array's CPU mirror.
     * Nothing reaches the GPU until the flush() calls at the end.
     *
     * Transform changes cascade to the whole descendant subtree (world transforms are
     * hierarchical), but only ever touch each entry's transform-derived GPU data -
     * never its properties (color, opacity, etc.), which are refreshed independently
     * when the entry itself (not its owner's transform) is marked dynamic/dirty.
     */
    private applyDynamicSceneUpdates(): void {
        if (!this.activeScene || !this.sceneGraph) {
            return;
        }

        const frameState = this.activeScene.getDynamicFrameState();

        for (const obj of frameState.persistentTransforms) {
            this.transformResources.markDynamic(obj.id);
        }
        for (const light of frameState.persistentLights) {
            this.lightResources.markDynamic(light);
        }
        for (const sprite of frameState.persistentSprites) {
            this.spriteResources.markDynamic(sprite);
        }
        for (const raytraced of frameState.persistentRaytraced) {
            this.raytracedResources.markDynamic(raytraced);
        }

        const transformAffectedIds = new Set<number>();
        for (const obj of frameState.transforms) {
            for (const affectedId of this.refreshTransformCascade(obj.id)) {
                transformAffectedIds.add(affectedId);
            }
        }

        for (const light of frameState.lights) {
            this.lightResources.refreshProperties(light, this.transformResources);
        }
        for (const sprite of frameState.sprites) {
            this.spriteResources.refreshProperties(sprite);
        }
        for (const raytraced of frameState.raytraced) {
            this.raytracedResources.refreshProperties(raytraced);
        }

        const simOwnerId = this.simulationResources.getOwnerId();
        if (simOwnerId !== null && transformAffectedIds.has(simOwnerId)) {
            this.simulationResources.refreshWorldTransform(this.sceneGraph);
            this.raytracedResources.refreshViewProjection(this.sceneGraph);
        }

        // One flush per array per frame, after every mutation above (structural ops earlier
        // this same frame included) and before any GPU pass is recorded - see PackedUniformArray.flush().
        this.transformResources.flush();
        this.lightResources.flush();
        this.raytracedResources.flush();
        this.spriteResources.flush();

        this.activeScene.clearFrameDirtyFlags();
    }

    private getActiveCamera(): ActiveCamera | null {
        if (!this.activeScene || !this.sceneGraph) {
            return null;
        }
        const scene = this.activeScene.data;
        if (scene.cameras.length !== 1) {
            console.warn(`Litbox: expected exactly 1 camera, found ${scene.cameras.length}; using the first active one.`);
        }
        const sceneGraph = this.sceneGraph;
        const camera = scene.cameras.find((c: SceneCamera) => sceneGraph.isActiveInHierarchy(c.ownerId));
        if (!camera) {
            this.lastActiveCameraOwnerId = null;
            return null;
        }
        if (camera.ownerId !== this.lastActiveCameraOwnerId) {
            this.lastActiveCameraOwnerId = camera.ownerId;
            this.exposureOverride = camera.exposure;
        }
        return { camera, worldTransform: sceneGraph.getWorldTransform(camera.ownerId) };
    }

    private writeCameraUniform(activeCamera: ActiveCamera | null): number {
        const viewProjection = mat4.create();
        let exposure = 0;

        if (activeCamera) {
            exposure = this.exposureOverride ?? activeCamera.camera.exposure;

            const view = mat4.create();
            mat4.invert(view, activeCamera.worldTransform);

            const aspect = this.presentationSize[1] > 0 ? this.presentationSize[0] / this.presentationSize[1] : 1;
            const halfHeight = activeCamera.camera.verticalSize;
            const halfWidth = halfHeight * aspect;
            const projection = mat4.create();
            // orthoZO (not the OpenGL-convention `ortho`/orthoNO): WebGPU's clip volume
            // expects NDC z in [0, 1], not [-1, 1].
            mat4.orthoZO(projection, -halfWidth, halfWidth, -halfHeight, halfHeight, -1000, 1000);

            mat4.multiply(viewProjection, projection, view);
        }

        const simInverse = mat4.create();
        mat4.invert(simInverse, this.simulationResources.getWorldTransform());

        const data = new Float32Array(CAMERA_UNIFORM_SIZE_BYTES / 4);
        data.set(viewProjection as Float32Array, 0);
        data.set(simInverse as Float32Array, 16);
        data[32] = this.debugSolidColor ? 1 : 0;
        this.cameraUniform.write(data);

        return exposure;
    }

    /**
     * Converts a canvas-space pixel coordinate (origin top-left, +y down - matching
     * MouseEvent.offsetX/offsetY against this renderer's canvas) into the active camera's
     * world-space XY plane, by inverting the same orthographic projection writeCameraUniform
     * builds. Returns null if there's no active camera to project against.
     */
    public screenToWorld(canvasX: number, canvasY: number): Vector2 | null {
        const activeCamera = this.getActiveCamera();
        if (!activeCamera || this.presentationSize[0] <= 0 || this.presentationSize[1] <= 0) {
            return null;
        }

        const ndcX = (canvasX / this.presentationSize[0]) * 2 - 1;
        const ndcY = 1 - (canvasY / this.presentationSize[1]) * 2;

        const aspect = this.presentationSize[0] / this.presentationSize[1];
        const halfHeight = activeCamera.camera.verticalSize;
        const halfWidth = halfHeight * aspect;

        const worldPoint = vec4.transformMat4(
            vec4.create(),
            vec4.fromValues(ndcX * halfWidth, ndcY * halfHeight, 0, 1),
            activeCamera.worldTransform,
        );
        return { x: worldPoint[0], y: worldPoint[1] };
    }

    public render(timeMs: number = performance.now()): void {
        if (!this.device) {
            return;
        }

        const deltaTimeSeconds = this.lastFrameTimeMs !== null ? (timeMs - this.lastFrameTimeMs) / 1000 : 0;
        this.lastFrameTimeMs = timeMs;
        this.activeScene?.onFrame(deltaTimeSeconds);
        this.applyPendingStructuralOps();
        this.applyDynamicSceneUpdates();

        if (this.canvas.width !== this.presentationSize[0] || this.canvas.height !== this.presentationSize[1]) {
            this.presentationSize = [this.canvas.width, this.canvas.height];
            this.createHdrFrameTexture();
        }

        try {
            const encoder = this.device.createCommandEncoder();

            // Step 1: render the G-Buffer (albedo/alpha, density, normal/roughness) for
            // raytraced objects. Runs unconditionally (even in debug-view mode below) so the
            // G-Buffer stays live against scene changes.
            this.raytracedResources.renderGBuffer(encoder);

            // Step 2: run the simulation (convert the photon-receptor buffer into the lightmap).
            this.simulationResources.run(encoder, this.raytracedResources);

            if (this.debugView) {
                const view = this.debugViews.get(this.debugView);
                const sourceView = view?.getSourceView() ?? null;
                if (view && sourceView) {
                    const debugPass = encoder.beginRenderPass({
                        colorAttachments: [{
                            view: this.context.getCurrentTexture().createView(),
                            clearValue: { r: 0, g: 0, b: 0, a: 1 },
                            loadOp: 'clear',
                            storeOp: 'store',
                        }],
                    });
                    this.debugViewBlitResources.apply(debugPass, sourceView, view.mode, this.debugViewScale);
                    debugPass.end();
                    this.device.queue.submit([encoder.finish()]);
                    return;
                }
            }

            const exposure = this.writeCameraUniform(this.getActiveCamera());

            // Steps 3-6: sprites + additive simulation composite into the offscreen HDR frame buffer.
            // No depth/stencil attachment - It's safe to presume the vast majority of objects have
            // transparency, so just draw them back to front.
            const hdrPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: this.hdrFrameTextureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            hdrPass.setBindGroup(0, this.cameraUniform.getBindGroup(), [this.cameraUniform.getCurrentOffset()]);
            this.spriteResources.draw(hdrPass, layer => layer <= 0);
            this.simulationResources.compositeInto(hdrPass);
            this.spriteResources.draw(hdrPass, layer => layer >= 1);
            hdrPass.end();

            // Step 7: tonemap the HDR frame buffer to the swapchain.
            const tonemapPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            this.tonemapResources.updateUniforms({ exposure });
            this.tonemapResources.updateInputs(this.hdrFrameTextureView);
            this.tonemapResources.execute(tonemapPass);
            tonemapPass.end();

            this.device.queue.submit([encoder.finish()]);
            this.cameraUniform.advance();
        } finally {
            // Third-party WebGPU instrumentation (browser devtools extensions wrapping
            // queue.submit, etc.) can throw after our own work is done. Keep the render
            // loop alive regardless, rather than letting one bad frame permanently kill it.
            requestAnimationFrame(this.render);
        }
    }
}
