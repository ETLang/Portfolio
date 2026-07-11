import type { Scene, SceneSprite, UvTransform } from './scene.ts';
import type { SceneGraph } from './scene_graph.ts';
import type { TextureCache } from './texture_cache.ts';
import type { SimulationResources } from './simulation.ts';
import type { TransformResources } from './transform_resources.ts';
import { Entry, PackedUniformArray } from './packed_uniform_array.ts';
import { QUAD_VERTEX_COUNT, QUAD_VERTEX_BUFFER_LAYOUT, getQuadVertexBuffer } from './quad_mesh.ts';
import spriteShaderCode from './shaders/sprite.wgsl?raw';

// erasableSyntaxOnly forbids `enum` - matches shapeId encoding in sprite.wgsl.
const PRIMITIVE_SHAPE_ID: Record<string, number> = { '': 0, rect: 1, ellipse: 2 };

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
 * sortOrder) order for overlapping transparency to blend correctly. draw() issues one draw
 * call per visible sprite, in that order (no texture-batching yet - see the project's
 * uniform-array packing plan for the batched-instanced-draw follow-up, which only changes how
 * many draw calls are issued, never their order).
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

        const shaderModule = this.device.createShaderModule({ code: spriteShaderCode });
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

    public async updateFromScene(
        scene: Scene,
        sceneGraph: SceneGraph,
        textureCache: TextureCache,
        simulationResources: SimulationResources,
        transformResources: TransformResources,
    ): Promise<void> {
        if (!this.sharedBindGroupLayout || !this.lightmapBindGroupLayout) {
            throw new Error('SpriteResources.initialize() must be called before updateFromScene().');
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

        for (let i = 0; i < this.sprites.length; i++) {
            const resolved = this.sprites[i];
            if (!resolved.isActive || !layerFilter(resolved.layer)) {
                continue;
            }
            const textureBindGroup = this.textureBindGroups.get(resolved.texture);
            if (!textureBindGroup) {
                continue; // defensive - resolveSprite/refreshTexture always ensure one exists
            }
            passEncoder.setBindGroup(2, textureBindGroup);
            // No batching yet (instanceCount 1): i is this sprite's fixed position in the
            // draw-ordered index buffer - see rebuildDrawOrder.
            passEncoder.draw(QUAD_VERTEX_COUNT, 1, 0, i);
        }
    }

    /**
     * Resolves and uploads a single newly-created sprite, appending it to the draw list without
     * touching any existing sprite's entries or texture - the targeted counterpart (for a
     * structural create op) to updateFromScene's full rebuild.
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
     * Unlike updateFromScene, this never touches any surviving sprite's entries or texture -
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

    /** Moves `sprite`'s properties entry into the dynamic region. No-op if untracked, or if already dynamic. */
    public markDynamic(sprite: SceneSprite): void {
        const resolved = this.sprites.find(r => r.sprite === sprite);
        if (!resolved) {
            return;
        }
        this.propertiesArray.markDynamic(resolved.propertiesEntry);
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
     * Sprites with equal (layer, sortOrder) keep their prior relative order (Array.sort is
     * stable) - fine, since that relative order is unobserved by design.
     */
    private rebuildDrawOrder(): void {
        for (const entry of this.indexEntries) {
            this.indexArray.remove(entry);
        }
        this.sprites.sort(compareDrawOrder);
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
        let shapeId = PRIMITIVE_SHAPE_ID[sprite.primitiveShape];
        if (shapeId === undefined) {
            console.warn(`Litbox: unrecognized primitiveShape "${sprite.primitiveShape}" on sprite owner ${sprite.ownerId}; treating as unspecified.`);
            shapeId = PRIMITIVE_SHAPE_ID[''];
        }
        return shapeId;
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
    view.setFloat32(byteOffset + 0, sprite.ambient.r, true);
    view.setFloat32(byteOffset + 4, sprite.ambient.g, true);
    view.setFloat32(byteOffset + 8, sprite.ambient.b, true);
    view.setFloat32(byteOffset + 12, sprite.ambient.a, true);
    view.setFloat32(byteOffset + 16, sprite.emissive.r, true);
    view.setFloat32(byteOffset + 20, sprite.emissive.g, true);
    view.setFloat32(byteOffset + 24, sprite.emissive.b, true);
    view.setFloat32(byteOffset + 28, sprite.emissive.a, true);
    view.setFloat32(byteOffset + 32, sprite.simContribution.r, true);
    view.setFloat32(byteOffset + 36, sprite.simContribution.g, true);
    view.setFloat32(byteOffset + 40, sprite.simContribution.b, true);
    view.setFloat32(byteOffset + 44, sprite.simContribution.a, true);
    view.setFloat32(byteOffset + 48, sprite.colorMod.r, true);
    view.setFloat32(byteOffset + 52, sprite.colorMod.g, true);
    view.setFloat32(byteOffset + 56, sprite.colorMod.b, true);
    view.setFloat32(byteOffset + 60, sprite.colorMod.a, true);
    view.setFloat32(byteOffset + 64, sprite.opacity, true);
    view.setFloat32(byteOffset + 68, sprite.simBlur, true);
    view.setUint32(byteOffset + 72, shapeId, true);
    // Bytes 76-79 are unused padding (WGSL rounds the struct up to a 16-byte multiple).
}
