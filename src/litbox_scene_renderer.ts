import { mat4 } from 'gl-matrix';
import type { SceneCamera } from './litbox/scene.ts';
import type { LitboxScene } from './litbox/litbox_scene.ts';
import { SceneGraph } from './litbox/scene_graph.ts';
import { TextureCache } from './litbox/texture_cache.ts';
import { LightResources } from './litbox/light_resources.ts';
import { RaytracedResources } from './litbox/raytraced_resources.ts';
import { SimulationResources } from './litbox/simulation.ts';
import { SpriteResources } from './litbox/sprite_resources.ts';
import { TonemapResources } from './litbox/tonemap.ts';
import { RingBufferedUniform } from './litbox/ring_buffered_uniform.ts';

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
    private lightResources!: LightResources;
    private raytracedResources!: RaytracedResources;
    private simulationResources!: SimulationResources;
    private spriteResources!: SpriteResources;
    private tonemapResources!: TonemapResources;

    private activeScene: LitboxScene | null = null;
    private lastFrameTimeMs: number | null = null;

    /**
     * Diagnostic aid: when true, sprites render as flat, fully-opaque, shape-colored
     * quads, bypassing opacity/shading entirely - useful for confirming transforms/camera/
     * layering are correct independent of per-sprite opacity, image, and shape data.
     */
    public debugSolidColor = false;

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
        scene.onLoad();
        if (this.device) {
            await this.rebuildFromScene();
        }
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
            this.device = await this.adapter.requestDevice();
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
        this.lightResources = new LightResources(this.device);
        this.raytracedResources = new RaytracedResources();
        this.simulationResources = new SimulationResources(this.device);
        this.spriteResources = new SpriteResources(this.device);
        this.tonemapResources = new TonemapResources(this.device, this.presentationFormat);

        // Shared by the sprite and simulation-composite pipelines. Ring-buffered because
        // it's the only per-frame-rewritten uniform in this renderer (everything else is
        // only rewritten on scene rebuilds).
        this.cameraBindGroupLayout = this.device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true } }],
        });
        this.cameraUniform = new RingBufferedUniform(this.device, this.cameraBindGroupLayout, CAMERA_UNIFORM_SIZE_BYTES, FRAMES_IN_FLIGHT);

        this.simulationResources.initialize(this.cameraBindGroupLayout);
        this.spriteResources.initialize(this.cameraBindGroupLayout, HDR_FORMAT);

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

        this.lightResources.updateFromScene(scene, this.sceneGraph);
        await this.raytracedResources.updateFromScene(scene, this.sceneGraph, this.textureCache);
        this.simulationResources.updateFromScene(scene, this.sceneGraph);
        await this.spriteResources.updateFromScene(scene, this.sceneGraph, this.textureCache, this.simulationResources);
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
        const ops = this.activeScene.getPendingStructuralOps();
        for (const op of ops) {
            if (op.type === 'create') {
                sceneGraph.addObject(op.object);
            } else if (op.type === 'destroy') {
                const removed = sceneGraph.removeObject(op.rootId);
                const scene = this.activeScene.data;
                this.lightResources.updateFromScene(scene, sceneGraph);
                void this.raytracedResources.updateFromScene(scene, sceneGraph, this.textureCache);
                this.spriteResources.removeByOwnerIds(new Set(removed));
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
     */
    private refreshTransformCascade(rootId: number): number[] {
        if (!this.sceneGraph) {
            return [];
        }
        const sceneGraph = this.sceneGraph;
        sceneGraph.invalidateSubtree(rootId);
        const affectedIds = [rootId, ...sceneGraph.getDescendantIds(rootId)];
        for (const ownerId of affectedIds) {
            this.lightResources.refreshTransform(ownerId, sceneGraph);
            this.spriteResources.refreshTransform(ownerId, sceneGraph);
            this.raytracedResources.refreshEntry(ownerId, sceneGraph);
        }
        return affectedIds;
    }

    /**
     * Consults the active scene's dynamic/dirty flags (see LitboxScene) and pushes
     * targeted GPU updates for exactly the affected entries, instead of re-uploading
     * everything (wasteful) or nothing (broken for animation/interaction). Runs before
     * any GPU pass is recorded since every write here is a bare queue.writeBuffer.
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
        const transformAffectedIds = new Set<number>();
        for (const obj of frameState.transforms) {
            for (const affectedId of this.refreshTransformCascade(obj.id)) {
                transformAffectedIds.add(affectedId);
            }
        }

        for (const light of frameState.lights) {
            this.lightResources.refreshProperties(light.ownerId);
        }
        for (const sprite of frameState.sprites) {
            this.spriteResources.refreshProperties(sprite.ownerId);
        }
        // frameState.raytraced entries deliberately have no per-entry properties update here:
        // RaytracedResources has no GPU buffer yet (see its refreshEntry TODO), so a raytraced
        // entry marked dynamic/dirty in isolation (owner transform unchanged) has nothing to
        // upload to - it's tracked purely for API symmetry until the simulation pass exists.

        const simOwnerId = this.simulationResources.getOwnerId();
        if (simOwnerId !== null && transformAffectedIds.has(simOwnerId)) {
            this.simulationResources.refreshWorldTransform(this.sceneGraph);
        }

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
            return null;
        }
        return { camera, worldTransform: sceneGraph.getWorldTransform(camera.ownerId) };
    }

    private writeCameraUniform(activeCamera: ActiveCamera | null): number {
        const viewProjection = mat4.create();
        let exposure = 0;

        if (activeCamera) {
            exposure = activeCamera.camera.exposure;

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

            // Step 1: run the (stubbed) simulation.
            this.simulationResources.run(encoder);

            const exposure = this.writeCameraUniform(this.getActiveCamera());

            // Steps 2-5: sprites + additive simulation composite into the offscreen HDR frame buffer.
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

            // Step 6: tonemap the HDR frame buffer to the swapchain.
            const tonemapPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            this.tonemapResources.apply(tonemapPass, this.hdrFrameTextureView, exposure);
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
