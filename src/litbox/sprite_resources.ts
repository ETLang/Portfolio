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

interface ResolvedSprite {
    ownerId: number;
    sprite: SceneSprite; // live reference - re-read on every properties update, not just captured at resolve time
    layer: number;
    sortOrder: number;
    isActive: boolean;
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

    public initialize(cameraBindGroupLayout: GPUBindGroupLayout, hdrFormat: GPUTextureFormat): void {
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
        this.lightmapBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ],
        });

        const shaderModule = this.device.createShaderModule({ code: preprocessShader(spriteShaderCode) });
        this.pipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [cameraBindGroupLayout, this.sharedBindGroupLayout, this.textureBindGroupLayout, this.lightmapBindGroupLayout],
            }),
            vertex: {
                module: shaderModule,
                entryPoint: 'vertex_main',
                buffers: [QUAD_VERTEX_BUFFER_LAYOUT],
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragment_main',
                targets: [{
                    format: hdrFormat,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                    },
                }],
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
        this.registerTransformResources(transformResources);

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

        const lightmapView = simulationResources.getLightmapView();
        this.lightmapBindGroup = this.device.createBindGroup({
            layout: this.lightmapBindGroupLayout,
            entries: [
                { binding: 0, resource: lightmapView ?? textureCache.getBlackTexture().createView() },
                { binding: 1, resource: simulationResources.getSampler() },
            ],
        });

        this.sprites = await Promise.all(scene.sprites.map(sprite => this.resolveSprite(sprite, sceneGraph, textureCache, transformResources)));
        this.rebuildDrawOrder();
    }

    /**
     * Draws every visible (active, layerFilter-passing) sprite, walking the draw-ordered list
     * once and issuing one instanced draw call per maximal run of consecutive, visible entries
     * that share a texture - a run breaks on a texture change *or* on a non-visible entry in
     * between (an inactive sprite, or one on the wrong side of layerFilter), since a single
     * instanced draw can't skip an instance in the middle of its [firstInstance, firstInstance
     * + instanceCount) range. This never changes draw order, only how many draw calls express
     * it - see rebuildDrawOrder for why same-texture entries tend to already be adjacent.
     */
    public draw(passEncoder: GPURenderPassEncoder, layerFilter: (layer: number) => boolean): void {
        if (!this.pipeline || !this.lightmapBindGroup) {
            return;
        }
        if (this.sharedBindGroupDirty) {
            this.rebuildSharedBindGroup();
        }
        if (!this.sharedBindGroup) {
            return;
        }

        passEncoder.setPipeline(this.pipeline);
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
            const visible = resolved.isActive && layerFilter(resolved.layer);
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
        const [removed] = this.sprites.splice(index, 1);
        this.propertiesArray.remove(removed.propertiesEntry);
        this.atlasArray.remove(removed.atlasEntry);
        transformResources.releaseEntry(removed.ownerId);
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
        const shapeId = this.resolveShapeId(sprite);
        this.propertiesArray.writeEntry(resolved.propertiesEntry, (view, byteOffset) => writePropertiesData(view, byteOffset, sprite, shapeId));

        const targetImage = sprite.image;
        if (targetImage !== resolved.lastResolvedImage && resolved.pendingImage !== targetImage) {
            resolved.pendingImage = targetImage;
            void this.refreshTexture(resolved);
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
        const shapeId = this.resolveShapeId(sprite);

        const transformEntry = transformResources.ensureEntry(sprite.ownerId, sceneGraph);
        const propertiesEntry = this.propertiesArray.insertStatic((view, byteOffset) => writePropertiesData(view, byteOffset, sprite, shapeId));
        const atlasEntry = this.atlasArray.insertStatic((view, byteOffset) => writeAtlasData(view, byteOffset, uvTransform));
        this.ensureTextureBindGroup(texture);

        return {
            ownerId: sprite.ownerId,
            sprite,
            layer: sprite.layer,
            sortOrder: sprite.sortOrder,
            isActive,
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

    private resolveShapeId(sprite: SceneSprite): number {
        return resolvePrimitiveShapeId(sprite.primitiveShape, sprite.ownerId);
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

function writePropertiesData(view: DataView, byteOffset: number, sprite: SceneSprite, shapeId: number): void {
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
    view.setFloat32(byteOffset + 68, sprite.simBlur, true);
    view.setUint32(byteOffset + 72, shapeId, true);
    // Bytes 76-79 are unused padding (WGSL rounds the struct up to a 16-byte multiple).
}
