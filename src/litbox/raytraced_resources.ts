import { mat4 } from 'gl-matrix';
import type { RaytracedObject, Scene, UvTransform } from './scene.ts';
import type { SceneGraph } from './scene_graph.ts';
import type { TextureCache } from './texture_cache.ts';
import type { SimulationResources } from './simulation.ts';
import type { TransformResources } from './transform_resources.ts';
import { Entry, PackedUniformArray } from './packed_uniform_array.ts';
import { PRIMITIVE_MESH_VERTEX_BUFFER_LAYOUT, PRIMITIVE_MESH_REGIONS, getPrimitiveMeshVertexBuffer } from './primitive_mesh.ts';
import { resolvePrimitiveShapeId } from './primitive_shape.ts';
import { srgbColorToLinear } from './color_space.ts';
import { clusterByTextureWithinTiedGroups } from './draw_order.ts';
import { ComputedDataManager, ComputedTexture } from './computed_data_manager.ts';
import raytracedGBufferShaderCode from './shaders/raytraced_gbuffer.wgsl?raw';
import { preprocessShader } from './shaders/shader_preprocessor.ts';

// Must match the RaytracedIndex/RaytracedProperties/RaytracedAtlasTransform struct layouts in
// raytraced_gbuffer.wgsl.
const RAYTRACED_INDEX_STRIDE_BYTES = 16;
const RAYTRACED_PROPERTIES_STRIDE_BYTES = 32;
const RAYTRACED_ATLAS_STRIDE_BYTES = 32;
// viewProjection (mat4) + targetHeightPixels (f32, padded to 16 bytes), matching GBufferCamera
// in raytraced_gbuffer.wgsl.
const GBUFFER_CAMERA_UNIFORM_SIZE_BYTES = 4 * 16 + 16;

const ALBEDO_FORMAT: GPUTextureFormat = 'rgba8unorm';
const NORMAL_ROUGHNESS_FORMAT: GPUTextureFormat = 'rgba16float';
// Stores (1-transmittance)*DENSITY_SCALE, not raw transmittance - see LitboxCommon.wgsl's
// DENSITY_SCALE for the full rationale (confirmed empirically: linear transmittance clusters
// right at float16's rounding boundary near 1.0, collapsing indistinctly to exactly 1.0; density
// reframes the same information away from that boundary). This is what makes plain rg16float
// viable here at all - no float32/'float32-blendable' feature dependency.
const DENSITY_FORMAT: GPUTextureFormat = 'rg16float';
// Must match DENSITY_SCALE in shaders/LitboxCommon.wgsl exactly - WGSL and TS can't share a
// literal across that language boundary, so this is a necessary second copy. Exported so other TS
// code that needs to interpret the Density target's raw values (e.g. a future simulation pass, or
// tests) has one canonical source rather than a second hardcoded copy on the TS side.
export const DENSITY_SCALE = 8192;

interface ResolvedRaytracedEntry {
    ownerId: number;
    raytraced: RaytracedObject; // live reference, re-read on refreshProperties
    sortOrder: number;
    isActive: boolean;
    texture: GPUTexture; // albedoMap - the only one currently bound to the G-Buffer draw (group 2), see raytraced_gbuffer.wgsl's file header
    transformEntry: Entry; // into the shared TransformResources array
    propertiesEntry: Entry;
    atlasEntry: Entry; // albedoMap's atlas UV transform
    lastResolvedImage: string; // albedoMap image currently baked into this entry's atlasEntry and texture bind group
    pendingImage: string | null; // albedoMap image an in-flight refreshTexture() call is resolving, if any
    // logDensityMap/sdfNormalMap: resolved and bind-group-ready the same way albedoMap is, but not
    // yet sampled by raytraced_gbuffer.wgsl or referenced by RaytracedIndex - see resolveRaytraced's
    // doc comment. No refreshTexture-equivalent exists for these two yet (only the initial resolve
    // is handled), since nothing consumes them at runtime until the shader/index layout catches up.
    logDensityTexture: GPUTexture;
    logDensityAtlasEntry: Entry;
    sdfNormalTexture: GPUTexture;
    sdfNormalAtlasEntry: Entry;
    shapeId: number; // cached resolvePrimitiveShapeId(raytraced.primitiveShape, ownerId) - looks up this object's mesh region in PRIMITIVE_MESH_REGIONS for draw batching
}

/**
 * Renders every raytraced scene object into the 3-target G-Buffer (albedo/alpha, density,
 * normal/roughness) that a future raytracing/path-tracing pass will sample - see
 * raytraced_gbuffer.wgsl for the exact per-target semantics ported from the Unity
 * reference (RTObjectMat.shader/SimulationCamera.cs). Structurally this mirrors SpriteResources
 * closely: per-object transform/properties/atlas data lives in shared, packed storage-buffer
 * arrays, looked up in-shader via a small GPU-resident index buffer keyed by
 * @builtin(instance_index).
 *
 * Draw order (ascending sortOrder) is a correctness requirement, not a performance knob: the
 * Albedo/Alpha target blends "over" and the NormalRoughness target overwrites unconditionally
 * (last-drawn wins), both order-dependent - see rebuildDrawOrder. The density target blends
 * additively, which is exact when only one object touches a given pixel (the common case for a
 * flat 2D layout like this) and only an approximation - a slight overestimate, never an
 * underestimate - when several objects' densities genuinely overlap the same pixel; accepted as
 * fine given this is a 2D raytracer where "on top of" is already an artistic approximation, not
 * literal depth. renderGBuffer() minimizes draw calls similarly to SpriteResources.draw(), plus one
 * extra dimension: rebuildDrawOrder coalesces same-sortOrder, same-texture, same-primitiveShape
 * runs together (see clusterByTextureWithinTiedGroups), and renderGBuffer issues one instanced draw
 * call (drawing from that shape's mesh region - see primitive_mesh.ts) per maximal run of
 * consecutive, active, same-texture, same-shape entries - this never reorders objects relative to
 * each other, so blending is unaffected by how many draw calls batching happens to produce.
 *
 * The 3 G-Buffer textures are acquired from a shared ComputedDataManager, sized to the active
 * scene's simulation resolution - only loadFromScene() (a scene load, or equivalently a
 * simulation-dimension change) touches them; per-object structural changes never do.
 */
export class RaytracedResources {
    private device: GPUDevice;
    private computedDataManager: ComputedDataManager;

    private pipeline: GPURenderPipeline | null = null;
    private cameraBindGroupLayout: GPUBindGroupLayout | null = null;
    private sharedBindGroupLayout: GPUBindGroupLayout | null = null;
    private textureBindGroupLayout: GPUBindGroupLayout | null = null;
    private vertexBuffer: GPUBuffer;

    private propertiesArray: PackedUniformArray;
    private atlasArray: PackedUniformArray;
    private indexArray: PackedUniformArray;

    /** Draw-ordered (ascending sortOrder); objects[i]'s index-buffer entry is always at position i - see rebuildDrawOrder. */
    private objects: ResolvedRaytracedEntry[] = [];
    private indexEntries: Entry[] = [];

    private textureBindGroups = new Map<GPUTexture, GPUBindGroup>();
    private sharedBindGroup: GPUBindGroup | null = null;
    private sharedBindGroupDirty = true;
    private cameraBindGroup: GPUBindGroup | null = null;

    private textureCache: TextureCache | null = null;
    private transformResources: TransformResources | null = null;

    private cameraUniformBuffer: GPUBuffer;

    private albedoGBuffer: ComputedTexture | null = null;
    private densityGBuffer: ComputedTexture | null = null;
    private normalRoughnessGBuffer: ComputedTexture | null = null;
    private hasGBufferTarget = false;
    private simulationOwnerId: number | null = null;

    constructor(device: GPUDevice, computedDataManager: ComputedDataManager) {
        this.device = device;
        this.computedDataManager = computedDataManager;
        this.vertexBuffer = getPrimitiveMeshVertexBuffer(device);
        this.propertiesArray = new PackedUniformArray(device, RAYTRACED_PROPERTIES_STRIDE_BYTES);
        this.atlasArray = new PackedUniformArray(device, RAYTRACED_ATLAS_STRIDE_BYTES);
        this.indexArray = new PackedUniformArray(device, RAYTRACED_INDEX_STRIDE_BYTES);
        this.propertiesArray.onBufferReplaced(() => { this.sharedBindGroupDirty = true; });
        this.atlasArray.onBufferReplaced(() => { this.sharedBindGroupDirty = true; });
        this.indexArray.onBufferReplaced(() => { this.sharedBindGroupDirty = true; });

        this.cameraUniformBuffer = device.createBuffer({
            size: GBUFFER_CAMERA_UNIFORM_SIZE_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    public initialize(): void {
        this.cameraBindGroupLayout = this.device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
        });
        this.sharedBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // raytracedIndices
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // transforms (shared)
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }, // raytracedProperties
                { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // atlasTransforms
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } }, // mainSampler
            ],
        });
        this.textureBindGroupLayout = this.device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }],
        });

        const shaderModule = this.device.createShaderModule({ code: preprocessShader(raytracedGBufferShaderCode) });
        this.pipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.cameraBindGroupLayout, this.sharedBindGroupLayout, this.textureBindGroupLayout],
            }),
            vertex: {
                module: shaderModule,
                entryPoint: 'vertex_main',
                buffers: [PRIMITIVE_MESH_VERTEX_BUFFER_LAYOUT],
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragment_main',
                targets: [
                    {
                        // AlbedoAlpha: Unity's "One OneMinusSrcAlpha" - standard premultiplied "over".
                        format: ALBEDO_FORMAT,
                        blend: {
                            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        },
                    },
                    {
                        // Density (see raytraced_gbuffer.wgsl's file header): additive, not Unity's
                        // multiplicative "Zero SrcColor" - density combines via d1+d2-d1*d2 exactly,
                        // but plain addition is exact for the single-object-per-pixel case and only
                        // slightly overestimates when several objects genuinely overlap - see the
                        // class doc.
                        format: DENSITY_FORMAT,
                        blend: {
                            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                        },
                    },
                    {
                        // NormalRoughness: Unity's "One Zero" is an unconditional overwrite - omitting
                        // `blend` entirely produces the same result without needing a blendable format.
                        format: NORMAL_ROUGHNESS_FORMAT,
                    },
                ],
            },
            primitive: { topology: 'triangle-list' },
        });

        this.cameraBindGroup = this.device.createBindGroup({
            layout: this.cameraBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.cameraUniformBuffer } }],
        });
    }

    /**
     * Full teardown-and-rebuild from `scene`, including releasing and reacquiring all 3 G-Buffer
     * textures (see the class doc). Called only on an actual scene load/swap (see
     * LitboxSceneRenderer.rebuildFromScene, its only caller) - never per-frame, and never for a
     * single object's create/destroy/property change, which go through addRaytraced/
     * removeRaytraced/refreshProperties instead.
     */
    public async loadFromScene(
        scene: Scene,
        sceneGraph: SceneGraph,
        textureCache: TextureCache,
        simulationResources: SimulationResources,
        transformResources: TransformResources,
    ): Promise<void> {
        if (!this.sharedBindGroupLayout) {
            throw new Error('RaytracedResources.initialize() must be called before loadFromScene().');
        }
        this.textureCache = textureCache;
        this.registerTransformResources(transformResources);

        for (const entry of this.indexEntries) {
            this.indexArray.remove(entry);
        }
        for (const resolved of this.objects) {
            this.propertiesArray.remove(resolved.propertiesEntry);
            this.atlasArray.remove(resolved.atlasEntry);
            this.atlasArray.remove(resolved.logDensityAtlasEntry);
            this.atlasArray.remove(resolved.sdfNormalAtlasEntry);
            transformResources.releaseEntry(resolved.ownerId);
        }
        this.objects = [];
        this.indexEntries = [];

        if (this.albedoGBuffer) {
            this.computedDataManager.releaseTexture(this.albedoGBuffer);
            this.albedoGBuffer = null;
        }
        if (this.densityGBuffer) {
            this.computedDataManager.releaseTexture(this.densityGBuffer);
            this.densityGBuffer = null;
        }
        if (this.normalRoughnessGBuffer) {
            this.computedDataManager.releaseTexture(this.normalRoughnessGBuffer);
            this.normalRoughnessGBuffer = null;
        }
        this.hasGBufferTarget = false;
        this.simulationOwnerId = simulationResources.getOwnerId();

        // Sourced from simulationResources (device-profile-scaled), not scene.simulations[0]
        // directly (the scene's raw, unscaled config) - this G-Buffer must match the simulation's
        // actual target resolution exactly, since forward_monte_carlo.wgsl samples both at the
        // same target pixel coordinates - see SimulationResources.getEffectiveResolution.
        const resolution = simulationResources.getEffectiveResolution();
        if (!resolution) {
            return;
        }
        // Mip chains feed the denoiser's evidence gathering (this project's denoiser plan) - same
        // level-count formula as SimulationResources' lightmap. Albedo/NormalRoughness get
        // STORAGE_BINDING so MipDownsampleOperation can textureStore into their higher mips;
        // Density (rg16float) can't - not a valid WGSL storage-texture format - so its mips are
        // generated via a render-attachment blit instead (DensityMipBlitResources), for which
        // RENDER_ATTACHMENT (already present below) is sufficient.
        const mipLevelCount = Math.floor(Math.log2(Math.max(resolution.width, resolution.height))) + 1;
        const storageUsage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        const renderOnlyUsage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
        this.albedoGBuffer = this.computedDataManager.acquireTexture(resolution.width, resolution.height, ALBEDO_FORMAT, storageUsage, mipLevelCount);
        this.densityGBuffer = this.computedDataManager.acquireTexture(resolution.width, resolution.height, DENSITY_FORMAT, renderOnlyUsage, mipLevelCount);
        this.normalRoughnessGBuffer = this.computedDataManager.acquireTexture(resolution.width, resolution.height, NORMAL_ROUGHNESS_FORMAT, storageUsage, mipLevelCount);
        this.hasGBufferTarget = true;

        this.objects = await Promise.all(scene.raytraced.map(entry => this.resolveRaytraced(entry, sceneGraph, textureCache, transformResources)));
        this.refreshViewProjection(sceneGraph);
        this.rebuildDrawOrder();
    }

    /**
     * Renders every active raytraced object into the 3 G-Buffer targets, in ascending sortOrder
     * order. Self-guards on having a valid simulation + textures, so callers (LitboxSceneRenderer)
     * can call this unconditionally, mirroring SimulationResources.run/compositeInto. Minimizes
     * draw calls by coalescing each maximal run of consecutive, active, same-texture objects (in
     * draw order) into a single instanced draw - see the class doc and rebuildDrawOrder. This
     * mirrors SpriteResources.draw() exactly, minus the layerFilter (raytraced objects have no
     * layer concept, only sortOrder).
     */
    public renderGBuffer(encoder: GPUCommandEncoder): void {
        if (!this.pipeline || !this.cameraBindGroup || !this.hasGBufferTarget
            || !this.albedoGBuffer || !this.densityGBuffer || !this.normalRoughnessGBuffer) {
            return;
        }
        if (this.sharedBindGroupDirty) {
            this.rebuildSharedBindGroup();
        }
        if (!this.sharedBindGroup) {
            return;
        }

        // mip0 specifically, not the default .view - these targets now carry a full mip chain
        // for the denoiser's evidence gathering (this project's denoiser plan), and a render-pass
        // color attachment must be a single mip level.
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                { view: this.albedoGBuffer.getMipView(0), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
                // Additive identity (0), matching the density target's additive blend below.
                { view: this.densityGBuffer.getMipView(0), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
                { view: this.normalRoughnessGBuffer.getMipView(0), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
            ],
        });
        pass.setPipeline(this.pipeline);
        pass.setVertexBuffer(0, this.vertexBuffer);
        pass.setBindGroup(0, this.cameraBindGroup);
        pass.setBindGroup(1, this.sharedBindGroup);

        let runStart = -1;
        let runTexture: GPUTexture | null = null;
        let runShapeId = -1;
        const flushRun = (endExclusive: number): void => {
            if (runStart === -1) {
                return;
            }
            const textureBindGroup = this.textureBindGroups.get(runTexture!);
            if (textureBindGroup) {
                const region = PRIMITIVE_MESH_REGIONS[runShapeId];
                pass.setBindGroup(2, textureBindGroup);
                pass.draw(region.vertexCount, endExclusive - runStart, region.firstVertex, runStart);
            }
            runStart = -1;
            runTexture = null;
            runShapeId = -1;
        };

        for (let i = 0; i < this.objects.length; i++) {
            const resolved = this.objects[i];
            if (!resolved.isActive || resolved.texture !== runTexture || resolved.shapeId !== runShapeId) {
                flushRun(i);
            }
            if (resolved.isActive && runStart === -1) {
                runStart = i;
                runTexture = resolved.texture;
                runShapeId = resolved.shapeId;
            }
        }
        flushRun(this.objects.length);
        pass.end();
    }

    /**
     * Resolves and uploads a single newly-created raytraced object, appending it to the draw
     * list without touching any existing object's entries or texture - the targeted counterpart
     * (for a structural create op) to loadFromScene's full rebuild.
     */
    public async addRaytraced(raytraced: RaytracedObject, sceneGraph: SceneGraph, textureCache: TextureCache, transformResources: TransformResources): Promise<void> {
        if (!this.sharedBindGroupLayout) {
            throw new Error('RaytracedResources.initialize() must be called before addRaytraced().');
        }
        this.registerTransformResources(transformResources);
        const resolved = await this.resolveRaytraced(raytraced, sceneGraph, textureCache, transformResources);
        this.objects.push(resolved);
        this.rebuildDrawOrder();
    }

    /** Removes exactly one raytraced object (matched by reference) and releases its transform reference. */
    public removeRaytraced(raytraced: RaytracedObject, transformResources: TransformResources): void {
        const index = this.objects.findIndex(resolved => resolved.raytraced === raytraced);
        if (index === -1) {
            return;
        }
        const [removed] = this.objects.splice(index, 1);
        this.propertiesArray.remove(removed.propertiesEntry);
        this.atlasArray.remove(removed.atlasEntry);
        this.atlasArray.remove(removed.logDensityAtlasEntry);
        this.atlasArray.remove(removed.sdfNormalAtlasEntry);
        transformResources.releaseEntry(removed.ownerId);
        this.rebuildDrawOrder();
    }

    /** Removes every raytraced object owned by an id in `ownerIds`, releasing their transform references. */
    public removeByOwnerIds(ownerIds: Set<number>, transformResources: TransformResources): void {
        const kept: ResolvedRaytracedEntry[] = [];
        let removedAny = false;
        for (const resolved of this.objects) {
            if (!ownerIds.has(resolved.ownerId)) {
                kept.push(resolved);
                continue;
            }
            this.propertiesArray.remove(resolved.propertiesEntry);
            this.atlasArray.remove(resolved.atlasEntry);
            this.atlasArray.remove(resolved.logDensityAtlasEntry);
            this.atlasArray.remove(resolved.sdfNormalAtlasEntry);
            transformResources.releaseEntry(resolved.ownerId);
            removedAny = true;
        }
        this.objects = kept;
        if (removedAny) {
            this.rebuildDrawOrder();
        }
    }

    /** Targeted re-upload of `raytraced`'s properties (and, if changed, its texture). No-op if untracked. */
    public refreshProperties(raytraced: RaytracedObject): void {
        const resolved = this.objects.find(r => r.raytraced === raytraced);
        if (!resolved) {
            return;
        }
        const shapeId = resolvePrimitiveShapeId(raytraced.primitiveShape, raytraced.ownerId);
        resolved.shapeId = shapeId;
        this.propertiesArray.writeEntry(resolved.propertiesEntry, (view, byteOffset) => writePropertiesData(view, byteOffset, raytraced, shapeId));

        const targetImage = raytraced.albedoMap;
        if (targetImage !== resolved.lastResolvedImage && resolved.pendingImage !== targetImage) {
            resolved.pendingImage = targetImage;
            void this.refreshTexture(resolved);
        }
    }

    /**
     * CPU-only refresh of the active-in-hierarchy cull flag for the raytraced object owned by
     * `ownerId`. No GPU write: isActive is consulted directly by renderGBuffer()'s visibility
     * check. Paired with TransformResources.refreshTransform in the renderer's transform cascade
     * - transform refresh itself is already fully covered there, since raytraced objects share
     * that array like sprites/lights do.
     */
    public refreshActiveState(ownerId: number, sceneGraph: SceneGraph): void {
        for (const resolved of this.objects) {
            if (resolved.ownerId === ownerId) {
                resolved.isActive = sceneGraph.isActiveInHierarchy(ownerId);
            }
        }
    }

    /** Moves `raytraced`'s properties entry into the dynamic region. No-op if untracked, or if already dynamic. */
    public markDynamic(raytraced: RaytracedObject): void {
        const resolved = this.objects.find(r => r.raytraced === raytraced);
        if (!resolved) {
            return;
        }
        this.propertiesArray.markDynamic(resolved.propertiesEntry);
    }

    /**
     * Re-derives and re-uploads the G-Buffer camera uniform (viewProjection + target height) from
     * the simulation owner's current world transform. viewProjection = scale(2,2,1) *
     * inverse(simWorldTransform): the inverse first maps world space into the simulation's own
     * [-0.5,0.5]^2 local rect (same convention as LitboxSceneRenderer's simInverseWorldTransform),
     * then the scale expands that into WebGPU NDC [-1,1]^2. No-op if there's no simulation or no
     * G-Buffer target yet.
     */
    public refreshViewProjection(sceneGraph: SceneGraph): void {
        if (this.simulationOwnerId === null || !this.hasGBufferTarget || !this.albedoGBuffer) {
            return;
        }
        const simWorldTransform = sceneGraph.getWorldTransform(this.simulationOwnerId);
        const inverse = mat4.create();
        mat4.invert(inverse, simWorldTransform);

        const scaleMat = mat4.create();
        mat4.fromScaling(scaleMat, [2, 2, 1]);
        const viewProjection = mat4.create();
        mat4.multiply(viewProjection, scaleMat, inverse);

        const data = new Float32Array(GBUFFER_CAMERA_UNIFORM_SIZE_BYTES / 4);
        data.set(viewProjection as Float32Array, 0);
        data[16] = this.albedoGBuffer.height;
        this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
    }

    public flush(): void {
        this.propertiesArray.flush();
        this.atlasArray.flush();
        this.indexArray.flush();
    }

    // --- Debug-view plumbing (see debug_view.ts's DebugView, registered by LitboxSceneRenderer). Not used by the normal render path.

    public getAlbedoView(): GPUTextureView | null {
        return this.albedoGBuffer?.view ?? null;
    }

    /** R=G hold (1-transmittance)*DENSITY_SCALE, not raw transmittance - see raytraced_gbuffer.wgsl's file header. */
    public getDensityView(): GPUTextureView | null {
        return this.densityGBuffer?.view ?? null;
    }

    public getNormalRoughnessView(): GPUTextureView | null {
        return this.normalRoughnessGBuffer?.view ?? null;
    }

    // --- G-Buffer mip-chain accessors, for the denoiser's evidence-gathering mip generation (see
    // this project's denoiser plan) - SimulationResources drives the actual downsample loop
    // (MipDownsampleOperation for Albedo/NormalRoughness, DensityMipBlitResources for Density),
    // reading/writing these per-level views; RaytracedResources only owns producing mip0 each
    // frame (renderGBuffer) and exposing the chain, not the downsample operations themselves.

    /** Mip level count shared by all 3 G-Buffer textures (they're always sized/allocated together - see loadFromScene). */
    public getGBufferMipLevelCount(): number {
        return this.albedoGBuffer?.mipLevelCount ?? 0;
    }

    public getAlbedoMipView(level: number): GPUTextureView | null {
        return this.albedoGBuffer?.getMipView(level) ?? null;
    }

    public getDensityMipView(level: number): GPUTextureView | null {
        return this.densityGBuffer?.getMipView(level) ?? null;
    }

    public getNormalRoughnessMipView(level: number): GPUTextureView | null {
        return this.normalRoughnessGBuffer?.getMipView(level) ?? null;
    }

    /**
     * Resolves all 3 of a raytraced object's maps (albedoMap, logDensityMap, sdfNormalMap) the
     * same way: texture + atlas UV transform resolved via TextureCache, atlas transform uploaded
     * to the shared atlasArray, texture bind group ensured via ensureTextureBindGroup. Only
     * albedoMap is actually sampled by raytraced_gbuffer.wgsl today (see its file header) - the
     * other two are fully resolved and bind-group-ready in anticipation of a future shader change
     * that samples them, rather than discarded the way they used to be. Their atlas entries exist
     * in the shared array but nothing (RaytracedIndex) points at them yet.
     */
    private async resolveRaytraced(raytraced: RaytracedObject, sceneGraph: SceneGraph, textureCache: TextureCache, transformResources: TransformResources): Promise<ResolvedRaytracedEntry> {
        const isActive = sceneGraph.isActiveInHierarchy(raytraced.ownerId);
        const { texture, uvTransform } = await textureCache.resolve(raytraced.albedoMap, 'white');
        const { texture: logDensityTexture, uvTransform: logDensityUvTransform } = await textureCache.resolve(raytraced.logDensityMap, 'black');
        const { texture: sdfNormalTexture, uvTransform: sdfNormalUvTransform } = await textureCache.resolve(raytraced.sdfNormalMap, 'black');

        const shapeId = resolvePrimitiveShapeId(raytraced.primitiveShape, raytraced.ownerId);
        const transformEntry = transformResources.ensureEntry(raytraced.ownerId, sceneGraph);
        const propertiesEntry = this.propertiesArray.insertStatic((view, byteOffset) => writePropertiesData(view, byteOffset, raytraced, shapeId));
        const atlasEntry = this.atlasArray.insertStatic((view, byteOffset) => writeAtlasData(view, byteOffset, uvTransform));
        const logDensityAtlasEntry = this.atlasArray.insertStatic((view, byteOffset) => writeAtlasData(view, byteOffset, logDensityUvTransform));
        const sdfNormalAtlasEntry = this.atlasArray.insertStatic((view, byteOffset) => writeAtlasData(view, byteOffset, sdfNormalUvTransform));
        this.ensureTextureBindGroup(texture);
        this.ensureTextureBindGroup(logDensityTexture);
        this.ensureTextureBindGroup(sdfNormalTexture);

        return {
            ownerId: raytraced.ownerId,
            raytraced,
            sortOrder: raytraced.sortOrder,
            isActive,
            texture,
            transformEntry,
            propertiesEntry,
            atlasEntry,
            lastResolvedImage: raytraced.albedoMap,
            pendingImage: null,
            logDensityTexture,
            logDensityAtlasEntry,
            sdfNormalTexture,
            sdfNormalAtlasEntry,
            shapeId,
        };
    }

    /**
     * Resolves a raytraced object's newly-assigned albedoMap (and its atlas transform) and swaps
     * its texture bind group in once ready. Fire-and-forget from refreshProperties (not awaited
     * by the render loop): renderGBuffer() keeps using this object's old texture until this
     * completes, so there's no flicker or invalid-binding window.
     */
    private async refreshTexture(resolved: ResolvedRaytracedEntry): Promise<void> {
        const targetImage = resolved.raytraced.albedoMap;
        const { texture, uvTransform } = await this.textureCache!.resolve(targetImage, 'white');

        if (resolved.raytraced.albedoMap !== targetImage) {
            // Superseded by a newer image change while this resolve was in flight.
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
     * Rebuilds the raytraced index buffer in strict ascending sortOrder draw order - a full
     * clear-and-reinsert, mirroring SpriteResources.rebuildDrawOrder. Within each maximal run of
     * objects sharing the same sortOrder - whose relative order is unobserved by design, see
     * compareRaytracedDrawOrder - entries are then locally regrouped by texture, so
     * renderGBuffer()'s run-length batching gets more (and longer) same-texture runs to
     * coalesce, at zero cost to draw-order correctness.
     */
    private rebuildDrawOrder(): void {
        for (const entry of this.indexEntries) {
            this.indexArray.remove(entry);
        }
        this.objects.sort(compareRaytracedDrawOrder);
        // Objects with different shapeIds draw from different mesh vertex ranges (see
        // primitive_mesh.ts) and can't share a batched draw call even when their texture matches -
        // so the regrouping key combines both, not just texture. GPUTexture instances aren't
        // directly usable as half of a compound primitive key, hence the call-scoped id map.
        const textureIds = new Map<GPUTexture, number>();
        clusterByTextureWithinTiedGroups(this.objects, compareRaytracedDrawOrder, (resolved) => {
            let id = textureIds.get(resolved.texture);
            if (id === undefined) {
                id = textureIds.size;
                textureIds.set(resolved.texture, id);
            }
            return `${id}:${resolved.shapeId}`;
        });
        this.indexEntries = this.objects.map(resolved =>
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
}

/**
 * The draw-order comparator: ascending sortOrder (raytraced objects have no layer concept, unlike
 * sprites). Equal sortOrders return 0 - their relative order is unobserved by design, so it's
 * left to Array.sort's stability rather than an arbitrary tiebreak here. Exported so its ordering
 * behavior can be unit-tested directly, mirroring SpriteResources' compareDrawOrder.
 */
export function compareRaytracedDrawOrder(a: { sortOrder: number }, b: { sortOrder: number }): number {
    return a.sortOrder - b.sortOrder;
}

function writeIndexData(view: DataView, byteOffset: number, resolved: ResolvedRaytracedEntry): void {
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

function writePropertiesData(view: DataView, byteOffset: number, raytraced: RaytracedObject, shapeId: number): void {
    // albedo is authored/stored in sRGB (matching Unity's Inspector-authored Color, see
    // color_space.ts) - converted to linear here, at GPU-upload time, since RTObjectMat.shader
    // declares _Color as a Color-typed property (gets Unity's own automatic conversion).
    const albedo = srgbColorToLinear(raytraced.albedo);
    view.setFloat32(byteOffset + 0, albedo.r, true);
    view.setFloat32(byteOffset + 4, albedo.g, true);
    view.setFloat32(byteOffset + 8, albedo.b, true);
    view.setFloat32(byteOffset + 12, albedo.a, true);
    view.setFloat32(byteOffset + 16, Math.pow(10, raytraced.logDensity), true); // substrateDensity
    view.setFloat32(byteOffset + 20, 1 - raytraced.roughness, true); // particleAlignment
    view.setFloat32(byteOffset + 24, raytraced.heightScale, true);
    view.setUint32(byteOffset + 28, shapeId, true);
}
