import type { Scene, SceneSprite, UvTransform } from './scene.ts';
import type { SceneGraph } from './scene_graph.ts';
import type { TextureCache } from './texture_cache.ts';
import type { SimulationResources } from './simulation.ts';
import type { TransformResources } from './transform_resources.ts';
import { Entry, PackedUniformArray } from './packed_uniform_array.ts';
import { QUAD_VERTEX_COUNT, QUAD_VERTEX_BUFFER_LAYOUT, getQuadVertexBuffer } from './quad_mesh.ts';
import { resolvePrimitiveShapeId } from './primitive_shape.ts';
import { clusterByTextureWithinTiedGroups } from './draw_order.ts';
import { srgbColorToLinear } from './color_space.ts';
import spriteShaderCode from './shaders/sprite.wgsl?raw';
import { preprocessShader } from './shaders/shader_preprocessor.ts';

// Must match the SpriteIndex/SpriteProperties/SpriteAtlasTransform struct layouts in sprite.wgsl.
const SPRITE_INDEX_STRIDE_BYTES = 16;
const SPRITE_PROPERTIES_STRIDE_BYTES = 80;
const SPRITE_ATLAS_STRIDE_BYTES = 32;

/**
 * Fixed number of full-resolution *blurred* cascade texture slots reserved in the lightmap bind
 * group layout/pipeline, matching the max of the blurCascadeLevelCount tunable - see this
 * project's plan: "Fix lightmap sprite-blur pixelation on high-contrast content". Doesn't count
 * the sharp (level 0, unblurred) lightmap itself, which always gets its own separate binding -
 * see buildLightmapBindGroup. Fixed (not sized to the live tunable) so changing it never requires
 * rebuilding pipeline layouts - unused slots bind the shared black-texture fallback instead. Must
 * match sprite.wgsl's own hardcoded slot count (WGSL can't import a TS constant across the
 * language boundary - same manually-kept-in-sync situation as RaytracedResources.DENSITY_SCALE,
 * see CLAUDE.md) and denoiser_tunables_panel.ts's blurCascadeLevelCount row max.
 */
export const MAX_CASCADE_LEVELS = 4;

interface ResolvedSprite {
    ownerId: number;
    sprite: SceneSprite; // live reference - re-read on every properties update, not just captured at resolve time
    layer: number;
    sortOrder: number;
    isActive: boolean;
    // Captured once at resolve time, like layer/sortOrder - nothing currently toggles this on a
    // live sprite. See draw()/drawBypass() and litbox_scene_renderer.ts's render() for how this
    // routes a sprite around the tonemap curve entirely.
    bypassTonemapping: boolean;
    texture: GPUTexture;
    transformEntry: Entry; // into the shared TransformResources array
    propertiesEntry: Entry;
    atlasEntry: Entry;
    lastResolvedImage: string; // image currently baked into this sprite's atlasEntry and texture bind group
    pendingImage: string | null; // image an in-flight refreshTexture() call is resolving, if any
}

/**
 * A single shared render pipeline draws every sprite. Per-sprite transform/properties/atlas
 * data lives in 3 shared, packed storage-buffer arrays (see PackedUniformArray and
 * TransformResources) rather than one GPUBuffer trio per sprite; each drawn sprite's slot in
 * those arrays is looked up in-shader via a small GPU-resident index buffer, indexed by
 * @builtin(instance_index) - see sprite.wgsl. Data is written once per scene rebuild by
 * default; a sprite (or its owner's transform) marked dynamic/dirty via LitboxScene gets its
 * corresponding entry rewritten every affected frame instead. Only the camera view-projection
 * uniform (bind group 0, owned by LitboxSceneRenderer) is unconditionally rewritten every frame.
 *
 * Draw order is a correctness requirement, not a performance knob: this renderer draws
 * back-to-front with no depth buffer, so sprites must be visited in ascending (layer,
 * sortOrder) order for overlapping transparency to blend correctly. draw() minimizes draw
 * calls by coalescing each maximal run of consecutive, visible, same-texture sprites in that
 * order into a single instanced draw - see draw() and rebuildDrawOrder's doc comments. This
 * never reorders sprites relative to each other, so blending is unaffected by how many draw
 * calls batching happens to produce.
 */
export class SpriteResources {
    private device: GPUDevice;
    private pipeline: GPURenderPipeline | null = null;
    // Same shader module/pipeline layout as `pipeline` - only the fragment target format differs
    // (the canvas's own `-srgb` view format instead of the HDR float format). Used by
    // drawBypass() to draw Background/Overlay sprites straight into the swapchain, skipping the
    // tonemap curve entirely - see litbox_scene_renderer.ts's render(). Deliberately not a
    // WGSL-level variant (no #ifdef, no second entry point): the shader always just outputs
    // linear color, and it's the destination view's format that decides whether that gets stored
    // as-is (HDR float) or hardware-gamma-encoded on write (the `-srgb` view), the same mechanism
    // already used in reverse when sampling an `-srgb` texture auto-decodes on read.
    private bypassPipeline: GPURenderPipeline | null = null;
    private sharedBindGroupLayout: GPUBindGroupLayout | null = null;
    private textureBindGroupLayout: GPUBindGroupLayout | null = null;
    private lightmapBindGroupLayout: GPUBindGroupLayout | null = null;
    private vertexBuffer: GPUBuffer;

    private propertiesArray: PackedUniformArray;
    private atlasArray: PackedUniformArray;
    private indexArray: PackedUniformArray;

    /** Draw-ordered (ascending layer, then sortOrder); sprites[i]'s index-buffer entry is always at position i - see rebuildDrawOrder. */
    private sprites: ResolvedSprite[] = [];
    private indexEntries: Entry[] = [];

    private textureBindGroups = new Map<GPUTexture, GPUBindGroup>();
    private sharedBindGroup: GPUBindGroup | null = null;
    private sharedBindGroupDirty = true;
    private lightmapBindGroup: GPUBindGroup | null = null;

    private textureCache: TextureCache | null = null;
    private transformResources: TransformResources | null = null;
    /** Stored from loadFromScene purely so draw() can poll it each frame for refreshLightmapCascade's change detection - never used for anything else. */
    private simulationResources: SimulationResources | null = null;
    /** SimulationResources.getSpriteBlurMipOffset() as of the last loadFromScene - see writePropertiesData. */
    private blurMipOffset = 0;
    /** SimulationResources.getBlurCascadeLevelCount() as of the last refreshLightmapCascade (or loadFromScene) - see writePropertiesData and draw()'s change-detection check. */
    private blurCascadeLevelCount = -1;

    /**
     * True while loadFromScene/removeSprite/removeByOwnerIds is already mid-flight - each of
     * those always finishes with its own rebuildDrawOrder() call against fully consistent state,
     * so a relocation-triggered rebuild during that window is both redundant and unsafe (it would
     * run against `this.sprites`/`this.indexEntries` mid-mutation - see registerTransformResources'
     * onOwnerRelocated listener).
     */
    private bulkOpInProgress = false;

    constructor(device: GPUDevice) {
        this.device = device;
        this.vertexBuffer = getQuadVertexBuffer(device);
        this.propertiesArray = new PackedUniformArray(device, SPRITE_PROPERTIES_STRIDE_BYTES);
        this.atlasArray = new PackedUniformArray(device, SPRITE_ATLAS_STRIDE_BYTES);
        this.indexArray = new PackedUniformArray(device, SPRITE_INDEX_STRIDE_BYTES);
        this.propertiesArray.onBufferReplaced(() => { this.sharedBindGroupDirty = true; });
        this.atlasArray.onBufferReplaced(() => { this.sharedBindGroupDirty = true; });
        this.indexArray.onBufferReplaced(() => { this.sharedBindGroupDirty = true; });
    }

    public initialize(cameraBindGroupLayout: GPUBindGroupLayout, hdrFormat: GPUTextureFormat, presentationFormatSrgb: GPUTextureFormat): void {
        this.sharedBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // spriteIndices
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // transforms (shared)
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // spriteProperties
                { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // atlasTransforms
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } }, // mainSampler
            ],
        });
        this.textureBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
            ],
        });
        // Binding 0: the sharp, unblurred lightmap itself (level 0 - see SimulationResources.
        // getLightmapView). Bindings 1..MAX_CASCADE_LEVELS: LightmapBlurCascade's full-resolution
        // blurred levels 1..MAX_CASCADE_LEVELS (getLightmapCascadeViews). Binding
        // MAX_CASCADE_LEVELS+1: the decimated mip chain past the cascade (getLightmapMipChainView).
        // Binding MAX_CASCADE_LEVELS+2: shared sampler - one suffices, every texture here is either
        // single-mip or sampled at an explicit LOD, so mipmapFilter never matters.
        const lightmapBindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = [];
        for (let i = 0; i <= MAX_CASCADE_LEVELS + 1; i++) {
            lightmapBindGroupLayoutEntries.push({ binding: i, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } });
        }
        lightmapBindGroupLayoutEntries.push({ binding: MAX_CASCADE_LEVELS + 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } });
        this.lightmapBindGroupLayout = this.device.createBindGroupLayout({ entries: lightmapBindGroupLayoutEntries });

        const shaderModule = this.device.createShaderModule({ code: preprocessShader(spriteShaderCode) });
        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [cameraBindGroupLayout, this.sharedBindGroupLayout, this.textureBindGroupLayout, this.lightmapBindGroupLayout],
        });
        const vertex: GPUVertexState = {
            module: shaderModule,
            entryPoint: 'vertex_main',
            buffers: [QUAD_VERTEX_BUFFER_LAYOUT],
        };
        // Standard straight-alpha "over" blend - also relied on by drawBypass()'s callers (see
        // litbox_scene_renderer.ts's render()) to composite over whatever's already in the
        // swapchain rather than overwrite it.
        const blend: GPUBlendState = {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        };
        this.pipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex,
            fragment: {
                module: shaderModule,
                entryPoint: 'fragment_main',
                targets: [{ format: hdrFormat, blend }],
            },
            primitive: { topology: 'triangle-list' },
        });
        // Same shaderModule/pipelineLayout/blend as `pipeline` above - only the target format
        // differs. See the bypassPipeline field doc for why this needs no WGSL changes.
        this.bypassPipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex,
            fragment: {
                module: shaderModule,
                entryPoint: 'fragment_main',
                targets: [{ format: presentationFormatSrgb, blend }],
            },
            primitive: { topology: 'triangle-list' },
        });
    }

    /**
     * Full teardown-and-rebuild from `scene`. Called only on an actual scene load/swap (see
     * LitboxSceneRenderer.rebuildFromScene, its only caller) - never per-frame, and never for a
     * single sprite's create/destroy/property change, which go through addSprite/removeSprite/
     * refreshProperties instead.
     */
    public async loadFromScene(
        scene: Scene,
        sceneGraph: SceneGraph,
        textureCache: TextureCache,
        simulationResources: SimulationResources,
        transformResources: TransformResources,
    ): Promise<void> {
        if (!this.sharedBindGroupLayout || !this.lightmapBindGroupLayout) {
            throw new Error('SpriteResources.initialize() must be called before loadFromScene().');
        }
        this.textureCache = textureCache;
        this.simulationResources = simulationResources;
        this.blurMipOffset = simulationResources.getSpriteBlurMipOffset();
        this.blurCascadeLevelCount = simulationResources.getBlurCascadeLevelCount();
        this.registerTransformResources(transformResources);

        this.bulkOpInProgress = true;
        try {
            for (const entry of this.indexEntries) {
                this.indexArray.remove(entry);
            }
            for (const resolved of this.sprites) {
                this.propertiesArray.remove(resolved.propertiesEntry);
                this.atlasArray.remove(resolved.atlasEntry);
                transformResources.releaseEntry(resolved.ownerId);
            }
            this.sprites = [];
            this.indexEntries = [];

            this.lightmapBindGroup = this.buildLightmapBindGroup(simulationResources, textureCache);

            this.sprites = await Promise.all(scene.sprites.map(sprite => this.resolveSprite(sprite, sceneGraph, textureCache, transformResources)));
        } finally {
            this.bulkOpInProgress = false;
        }
        this.rebuildDrawOrder();
    }

    /**
     * Draws every visible (active, layerFilter-passing), non-bypass sprite - the regular HDR
     * path. See drawBypass() for the complementary Background/Overlay (bypassTonemapping: true)
     * sprites, which skip this pipeline (and the tonemap curve) entirely.
     */
    public draw(passEncoder: GPURenderPassEncoder, layerFilter: (layer: number) => boolean): void {
        this.drawFiltered(passEncoder, this.pipeline, resolved => resolved.isActive && layerFilter(resolved.layer) && !resolved.bypassTonemapping);
    }

    /**
     * Draws every visible, bypassTonemapping sprite matching layerFilter, using bypassPipeline
     * (targets the swapchain's own `-srgb` view, not the HDR buffer) instead of the regular
     * pipeline - see litbox_scene_renderer.ts's render() for where this is called relative to the
     * tonemap pass (once before it for the layer<=0 "Background" tier, once after for the
     * layer>=1 "Overlay" tier) and why the swapchain must be cleared opaque beforehand.
     */
    public drawBypass(passEncoder: GPURenderPassEncoder, layerFilter: (layer: number) => boolean): void {
        this.drawFiltered(passEncoder, this.bypassPipeline, resolved => resolved.isActive && layerFilter(resolved.layer) && resolved.bypassTonemapping);
    }

    /**
     * Shared draw-order walk behind draw()/drawBypass(): walks the draw-ordered list once and
     * issues one instanced draw call per maximal run of consecutive, isVisible-passing entries
     * that share a texture - a run breaks on a texture change *or* on a non-visible entry in
     * between, since a single instanced draw can't skip an instance in the middle of its
     * [firstInstance, firstInstance + instanceCount) range. This never changes draw order, only
     * how many draw calls express it - see rebuildDrawOrder for why same-texture entries tend to
     * already be adjacent.
     */
    private drawFiltered(passEncoder: GPURenderPassEncoder, pipeline: GPURenderPipeline | null, isVisible: (resolved: ResolvedSprite) => boolean): void {
        if (!pipeline || !this.lightmapBindGroup) {
            return;
        }
        if (this.sharedBindGroupDirty) {
            this.rebuildSharedBindGroup();
        }
        if (this.simulationResources && this.simulationResources.getBlurCascadeLevelCount() !== this.blurCascadeLevelCount) {
            this.refreshLightmapCascade();
        }
        if (!this.sharedBindGroup) {
            return;
        }

        passEncoder.setPipeline(pipeline);
        passEncoder.setVertexBuffer(0, this.vertexBuffer);
        passEncoder.setBindGroup(1, this.sharedBindGroup);
        passEncoder.setBindGroup(3, this.lightmapBindGroup);

        let runStart = -1;
        let runTexture: GPUTexture | null = null;
        const flushRun = (endExclusive: number): void => {
            if (runStart === -1) {
                return;
            }
            const textureBindGroup = this.textureBindGroups.get(runTexture!);
            if (textureBindGroup) {
                passEncoder.setBindGroup(2, textureBindGroup);
                passEncoder.draw(QUAD_VERTEX_COUNT, endExclusive - runStart, 0, runStart);
            }
            runStart = -1;
            runTexture = null;
        };

        for (let i = 0; i < this.sprites.length; i++) {
            const resolved = this.sprites[i];
            const visible = isVisible(resolved);
            if (!visible || resolved.texture !== runTexture) {
                flushRun(i);
            }
            if (visible && runStart === -1) {
                runStart = i;
                runTexture = resolved.texture;
            }
        }
        flushRun(this.sprites.length);
    }

    /**
     * Resolves and uploads a single newly-created sprite, appending it to the draw list without
     * touching any existing sprite's entries or texture - the targeted counterpart (for a
     * structural create op) to loadFromScene's full rebuild.
     */
    public async addSprite(sprite: SceneSprite, sceneGraph: SceneGraph, textureCache: TextureCache, transformResources: TransformResources): Promise<void> {
        if (!this.sharedBindGroupLayout) {
            throw new Error('SpriteResources.initialize() must be called before addSprite().');
        }
        this.registerTransformResources(transformResources);
        const resolved = await this.resolveSprite(sprite, sceneGraph, textureCache, transformResources);
        this.sprites.push(resolved);
        this.rebuildDrawOrder();
    }

    /**
     * Removes exactly one sprite (matched by reference, not ownerId) and releases its transform
     * reference, leaving any sibling sprites the same owner has untouched - the targeted
     * counterpart (for a destroySprite structural op) to removeByOwnerIds below, which removes
     * every sprite an owner has.
     */
    public removeSprite(sprite: SceneSprite, transformResources: TransformResources): void {
        const index = this.sprites.findIndex(resolved => resolved.sprite === sprite);
        if (index === -1) {
            return;
        }
        this.bulkOpInProgress = true;
        try {
            const [removed] = this.sprites.splice(index, 1);
            this.propertiesArray.remove(removed.propertiesEntry);
            this.atlasArray.remove(removed.atlasEntry);
            transformResources.releaseEntry(removed.ownerId);
        } finally {
            this.bulkOpInProgress = false;
        }
        this.rebuildDrawOrder();
    }

    /**
     * Removes every sprite owned by an id in `ownerIds`, releasing their transform references.
     * Unlike loadFromScene, this never touches any surviving sprite's entries or texture -
     * the targeted counterpart for structural destroy ops.
     */
    public removeByOwnerIds(ownerIds: Set<number>, transformResources: TransformResources): void {
        const kept: ResolvedSprite[] = [];
        let removedAny = false;
        this.bulkOpInProgress = true;
        try {
            for (const resolved of this.sprites) {
                if (!ownerIds.has(resolved.ownerId)) {
                    kept.push(resolved);
                    continue;
                }
                this.propertiesArray.remove(resolved.propertiesEntry);
                this.atlasArray.remove(resolved.atlasEntry);
                transformResources.releaseEntry(resolved.ownerId);
                removedAny = true;
            }
            this.sprites = kept;
        } finally {
            this.bulkOpInProgress = false;
        }
        if (removedAny) {
            this.rebuildDrawOrder();
        }
    }

    /** Targeted re-upload of `sprite`'s properties (and, if changed, its texture). No-op if untracked. */
    public refreshProperties(sprite: SceneSprite): void {
        const resolved = this.sprites.find(r => r.sprite === sprite);
        if (!resolved) {
            return;
        }
        const shapeId = resolvePrimitiveShapeId(sprite.primitiveShape);
        this.propertiesArray.writeEntry(resolved.propertiesEntry, (view, byteOffset) => writePropertiesData(view, byteOffset, sprite, shapeId, this.blurMipOffset, this.blurCascadeLevelCount));

        const targetImage = sprite.image;
        if (targetImage !== resolved.lastResolvedImage && resolved.pendingImage !== targetImage) {
            resolved.pendingImage = targetImage;
            this.refreshTexture(resolved).catch((error) => console.error('Litbox: SpriteResources.refreshTexture failed:', error));
        }
    }

    /**
     * CPU-only refresh of the active-in-hierarchy cull flag for every sprite owned by
     * `ownerId`. No GPU write: isActive is consulted directly by draw()'s visibility check,
     * never uploaded. Paired with TransformResources.refreshTransform in the renderer's
     * transform cascade, since SceneGraph invalidates and re-derives both together (see
     * SceneGraph.invalidateSubtree) - an owner's active state can change without its transform
     * changing (e.g. toggling SceneObject.active), but both are only picked up on the same
     * cascade, matching this project's existing dynamic/dirty-marking convention.
     */
    public refreshActiveState(ownerId: number, sceneGraph: SceneGraph): void {
        for (const resolved of this.sprites) {
            if (resolved.ownerId === ownerId) {
                resolved.isActive = sceneGraph.isActiveInHierarchy(ownerId);
            }
        }
    }

    /** Moves `sprite`'s properties entry into the dynamic region. No-op if untracked, or if already dynamic. */
    public markDynamic(sprite: SceneSprite): void {
        const resolved = this.sprites.find(r => r.sprite === sprite);
        if (!resolved) {
            return;
        }
        // propertiesArray.markDynamic can relocate up to two entries: resolved's own (moving to
        // the dynamic region) and whichever sprite currently occupies the last static slot
        // (displaced to make the dynamic region contiguous - see PackedUniformArray.markDynamic).
        // Every sprite's spriteIndices entry holds a *snapshot* of its propertiesEntry.index taken
        // by rebuildDrawOrder - so either relocation leaves that snapshot stale (pointing at
        // whatever now occupies the old slot) until the index buffer is rederived. Only do this
        // on an actual (first-time) transition, so per-frame calls on an already-dynamic sprite
        // (the common case - see LitboxSceneRenderer.applyDynamicSceneUpdates) stay a cheap no-op.
        const wasStatic = resolved.propertiesEntry.index < this.propertiesArray.getStaticCount();
        this.propertiesArray.markDynamic(resolved.propertiesEntry);
        if (wasStatic) {
            this.rebuildDrawOrder();
        }
    }

    public flush(): void {
        this.propertiesArray.flush();
        this.atlasArray.flush();
        this.indexArray.flush();
    }

    private async resolveSprite(sprite: SceneSprite, sceneGraph: SceneGraph, textureCache: TextureCache, transformResources: TransformResources): Promise<ResolvedSprite> {
        const isActive = sceneGraph.isActiveInHierarchy(sprite.ownerId);
        const { texture, uvTransform } = await textureCache.resolve(sprite.image, 'white');
        const shapeId = resolvePrimitiveShapeId(sprite.primitiveShape);

        const transformEntry = transformResources.ensureEntry(sprite.ownerId, sceneGraph);
        const propertiesEntry = this.propertiesArray.insertStatic((view, byteOffset) => writePropertiesData(view, byteOffset, sprite, shapeId, this.blurMipOffset, this.blurCascadeLevelCount));
        const atlasEntry = this.atlasArray.insertStatic((view, byteOffset) => writeAtlasData(view, byteOffset, uvTransform));
        this.ensureTextureBindGroup(texture);

        return {
            ownerId: sprite.ownerId,
            sprite,
            layer: sprite.layer,
            sortOrder: sprite.sortOrder,
            isActive,
            bypassTonemapping: sprite.bypassTonemapping,
            texture,
            transformEntry,
            propertiesEntry,
            atlasEntry,
            lastResolvedImage: sprite.image,
            pendingImage: null,
        };
    }

    /**
     * Resolves a sprite's newly-assigned image (and its atlas transform) and swaps its texture
     * bind group in once ready. Fire-and-forget from refreshProperties (not awaited by the render
     * loop): draw() keeps using this sprite's old texture - still valid, still bound to a live
     * texture - until this completes, so there's no flicker or invalid-binding window.
     */
    private async refreshTexture(resolved: ResolvedSprite): Promise<void> {
        const targetImage = resolved.sprite.image;
        const { texture, uvTransform } = await this.textureCache!.resolve(targetImage, 'white');

        if (resolved.sprite.image !== targetImage) {
            // Superseded by a newer image change while this resolve was in flight;
            // refreshProperties will have kicked off (or will kick off) a fresh
            // refreshTexture for whatever the current target is.
            resolved.pendingImage = null;
            return;
        }

        this.atlasArray.writeEntry(resolved.atlasEntry, (view, byteOffset) => writeAtlasData(view, byteOffset, uvTransform));
        this.ensureTextureBindGroup(texture);
        resolved.texture = texture;
        resolved.lastResolvedImage = targetImage;
        resolved.pendingImage = null;
    }

    /**
     * Rebuilds the sprite index buffer in strict ascending (layer, sortOrder) draw order - a
     * full clear-and-reinsert, since arbitrary reordering isn't something insertStatic/remove
     * support (and don't need to: this array holds one small 16-byte struct per sprite and this
     * only runs on structural change or a resolveSprite's initial insert, never per frame).
     * Within each maximal run of sprites sharing the same (layer, sortOrder) - whose relative
     * order is unobserved by design, see compareDrawOrder - entries are then locally regrouped
     * by texture, so draw()'s run-length batching gets more (and longer) same-texture runs to
     * coalesce, at zero cost to draw-order correctness.
     */
    private rebuildDrawOrder(): void {
        for (const entry of this.indexEntries) {
            this.indexArray.remove(entry);
        }
        this.sprites.sort(compareDrawOrder);
        clusterByTextureWithinTiedGroups(this.sprites, compareDrawOrder);
        this.indexEntries = this.sprites.map(resolved =>
            this.indexArray.insertStatic((view, byteOffset) => writeIndexData(view, byteOffset, resolved)));
    }

    private ensureTextureBindGroup(texture: GPUTexture): void {
        if (this.textureBindGroups.has(texture)) {
            return;
        }
        this.textureBindGroups.set(texture, this.device.createBindGroup({
            layout: this.textureBindGroupLayout!,
            entries: [{ binding: 0, resource: texture.createView() }],
        }));
    }

    private registerTransformResources(transformResources: TransformResources): void {
        if (this.transformResources === transformResources) {
            return;
        }
        this.transformResources = transformResources;
        transformResources.onBufferReplaced(() => { this.sharedBindGroupDirty = true; });
        // indexArray bakes each sprite's transformEntry.index as of rebuildDrawOrder() time (see
        // writeIndexData) - if that owner's entry later relocates for a reason outside this
        // sprite's own control (see TransformResources.onOwnerRelocated's doc - e.g. some other
        // owner's transform being marked dynamic), the baked index goes stale and the draw
        // silently samples a different object's transform. A full rebuildDrawOrder() re-bakes
        // every entry from its current (fresh) index; scenes here are small enough that this is
        // cheap even though only one sprite's index actually changed.
        transformResources.onOwnerRelocated((ownerId) => {
            if (!this.bulkOpInProgress && this.sprites.some(resolved => resolved.ownerId === ownerId)) {
                this.rebuildDrawOrder();
            }
        });
    }

    /** Binds the sharp lightmap, LightmapBlurCascade's current blurred levels, and the mip chain, padding any unused cascade slot (below MAX_CASCADE_LEVELS) with the shared black texture - see the lightmapBindGroupLayout comment in initialize(). */
    private buildLightmapBindGroup(simulationResources: SimulationResources, textureCache: TextureCache): GPUBindGroup {
        const blackView = textureCache.getBlackTexture().createView();
        const cascadeViews = simulationResources.getLightmapCascadeViews();
        const entries: GPUBindGroupEntry[] = [];
        entries.push({ binding: 0, resource: simulationResources.getLightmapView() ?? blackView });
        for (let i = 0; i < MAX_CASCADE_LEVELS; i++) {
            entries.push({ binding: i + 1, resource: cascadeViews[i] ?? blackView });
        }
        entries.push({ binding: MAX_CASCADE_LEVELS + 1, resource: simulationResources.getLightmapMipChainView() ?? blackView });
        entries.push({ binding: MAX_CASCADE_LEVELS + 2, resource: simulationResources.getSampler() });
        return this.device.createBindGroup({ layout: this.lightmapBindGroupLayout!, entries });
    }

    /**
     * Rebinds the lightmap bind group against LightmapBlurCascade's just-reallocated textures and
     * re-derives every sprite's simBlurBucket/simBlurLod split against the new cascade level
     * count - see writePropertiesData. Triggered from draw()'s per-frame poll of
     * SimulationResources.getBlurCascadeLevelCount(), which only actually changes right after the
     * blurCascadeLevelCount tunable does (see SimulationResources.ensureBlurCascadeAllocated), so
     * this is cheap to poll but rarely does real work.
     */
    private refreshLightmapCascade(): void {
        if (!this.simulationResources || !this.textureCache || !this.lightmapBindGroupLayout) {
            return;
        }
        this.lightmapBindGroup = this.buildLightmapBindGroup(this.simulationResources, this.textureCache);
        this.blurCascadeLevelCount = this.simulationResources.getBlurCascadeLevelCount();

        for (const resolved of this.sprites) {
            const shapeId = resolvePrimitiveShapeId(resolved.sprite.primitiveShape);
            this.propertiesArray.writeEntry(resolved.propertiesEntry, (view, byteOffset) =>
                writePropertiesData(view, byteOffset, resolved.sprite, shapeId, this.blurMipOffset, this.blurCascadeLevelCount));
        }
    }

    private rebuildSharedBindGroup(): void {
        if (!this.sharedBindGroupLayout || !this.transformResources || !this.textureCache) {
            return;
        }
        this.sharedBindGroup = this.device.createBindGroup({
            layout: this.sharedBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.indexArray.getBuffer() } },
                { binding: 1, resource: { buffer: this.transformResources.getBuffer() } },
                { binding: 2, resource: { buffer: this.propertiesArray.getBuffer() } },
                { binding: 3, resource: { buffer: this.atlasArray.getBuffer() } },
                { binding: 4, resource: this.textureCache.trilinearClamped },
            ],
        });
        this.sharedBindGroupDirty = false;
    }
}

/**
 * The draw-order comparator: ascending layer, then ascending sortOrder within a layer. Equal
 * (layer, sortOrder) pairs return 0 - their relative order is unobserved by design, so it's
 * left to Array.sort's stability rather than an arbitrary tiebreak here. Exported so its
 * ordering behavior can be unit-tested directly, without needing a full SpriteResources +
 * GPU-stub fixture just to exercise a two-field comparison.
 */
export function compareDrawOrder(a: { layer: number; sortOrder: number }, b: { layer: number; sortOrder: number }): number {
    return (a.layer - b.layer) || (a.sortOrder - b.sortOrder);
}

function writeIndexData(view: DataView, byteOffset: number, resolved: ResolvedSprite): void {
    view.setUint32(byteOffset + 0, resolved.transformEntry.index, true);
    view.setUint32(byteOffset + 4, resolved.propertiesEntry.index, true);
    view.setUint32(byteOffset + 8, resolved.atlasEntry.index, true);
    view.setUint32(byteOffset + 12, 0, true);
}

function writeAtlasData(view: DataView, byteOffset: number, uvTransform: UvTransform): void {
    view.setFloat32(byteOffset + 0, uvTransform.a, true);
    view.setFloat32(byteOffset + 4, uvTransform.b, true);
    view.setFloat32(byteOffset + 8, uvTransform.c, true);
    view.setFloat32(byteOffset + 12, 0, true);
    view.setFloat32(byteOffset + 16, uvTransform.d, true);
    view.setFloat32(byteOffset + 20, uvTransform.e, true);
    view.setFloat32(byteOffset + 24, uvTransform.f, true);
    view.setFloat32(byteOffset + 28, 0, true);
}

function writePropertiesData(view: DataView, byteOffset: number, sprite: SceneSprite, shapeId: number, blurMipOffset: number, blurCascadeLevelCount: number): void {
    // ambient/emissive/simContribution/colorMod are authored/stored in sRGB (matching Unity's
    // Inspector-authored Color, see color_space.ts) - converted to linear here, at GPU-upload
    // time, since PortfolioSpriteShader.shader declares all 4 corresponding properties
    // (_Ambience/_Emissive/_LightMod/_Color) as Color-typed (each gets Unity's own automatic
    // conversion via RTDemoSprite.cs's MaterialPropertyBlock.SetColor calls).
    const ambient = srgbColorToLinear(sprite.ambient);
    const emissive = srgbColorToLinear(sprite.emissive);
    const simContribution = srgbColorToLinear(sprite.simContribution);
    const colorMod = srgbColorToLinear(sprite.colorMod);
    view.setFloat32(byteOffset + 0, ambient.r, true);
    view.setFloat32(byteOffset + 4, ambient.g, true);
    view.setFloat32(byteOffset + 8, ambient.b, true);
    view.setFloat32(byteOffset + 12, ambient.a, true);
    view.setFloat32(byteOffset + 16, emissive.r, true);
    view.setFloat32(byteOffset + 20, emissive.g, true);
    view.setFloat32(byteOffset + 24, emissive.b, true);
    view.setFloat32(byteOffset + 28, emissive.a, true);
    view.setFloat32(byteOffset + 32, simContribution.r, true);
    view.setFloat32(byteOffset + 36, simContribution.g, true);
    view.setFloat32(byteOffset + 40, simContribution.b, true);
    view.setFloat32(byteOffset + 44, simContribution.a, true);
    view.setFloat32(byteOffset + 48, colorMod.r, true);
    view.setFloat32(byteOffset + 52, colorMod.g, true);
    view.setFloat32(byteOffset + 56, colorMod.b, true);
    view.setFloat32(byteOffset + 60, colorMod.a, true);
    view.setFloat32(byteOffset + 64, sprite.opacity, true);
    // See computeSpriteBlurMipOffset (simulation.ts): simBlur is authored assuming the lightmap's
    // mip 0 is at the scene's natural resolution, which a scaled-down device profile violates -
    // clamped to 0 since a negative level isn't valid. Rounded since simBlur is always meant to be
    // a whole number (see sprite.wgsl's header comment) - defends only against float drift from
    // the blurMipOffset subtraction, not against genuinely fractional authoring.
    const level = Math.round(Math.max(0, sprite.simBlur - blurMipOffset));
    // Split into which texture to sample (simBlurBucket) and the LOD within it (simBlurLod, only
    // meaningful for the mip-chain bucket - every cascade slot, including the sharp level-0
    // binding, is single-mip) - see sprite.wgsl's SpriteProperties and this project's plan.
    // Bucket 0 is the sharp lightmap itself (level 0); buckets 1..cascadeLevelCount are
    // LightmapBlurCascade's exposed blurred levels (bucket i <-> cascade slot i-1, since level 0
    // isn't part of that array - see SimulationResources.getLightmapCascadeViews); anything past
    // cascadeLevelCount falls back to the mip-chain sentinel bucket (MAX_CASCADE_LEVELS+1).
    // cascadeLevelCount is clamped defensively: blurCascadeLevelCount comes from a live tunable
    // (or -1 before the first SimulationResources.run() of a session has allocated anything), not
    // a value this function controls.
    const cascadeLevelCount = Math.max(0, Math.min(Math.round(blurCascadeLevelCount), MAX_CASCADE_LEVELS));
    const useMipChain = level > cascadeLevelCount;
    const simBlurBucket = useMipChain ? MAX_CASCADE_LEVELS + 1 : level;
    const simBlurLod = useMipChain ? level - cascadeLevelCount - 1 : 0;
    view.setFloat32(byteOffset + 68, simBlurLod, true);
    view.setUint32(byteOffset + 72, shapeId, true);
    view.setUint32(byteOffset + 76, simBlurBucket, true);
}
