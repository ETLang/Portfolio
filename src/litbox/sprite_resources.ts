import type { mat4 } from 'gl-matrix';
import type { Scene, SceneSprite } from './scene.ts';
import type { SceneGraph } from './scene_graph.ts';
import type { TextureCache } from './texture_cache.ts';
import type { SimulationResources } from './simulation.ts';
import { QUAD_VERTEX_COUNT, QUAD_VERTEX_BUFFER_LAYOUT, getQuadVertexBuffer } from './quad_mesh.ts';
import spriteShaderCode from './shaders/sprite.wgsl?raw';

// erasableSyntaxOnly forbids `enum` - matches shapeId encoding in sprite.wgsl.
const PRIMITIVE_SHAPE_ID: Record<string, number> = { '': 0, rect: 1, ellipse: 2 };

// Must match the SpriteTransform/SpriteProperties struct layouts in sprite.wgsl.
const SPRITE_TRANSFORM_STRIDE_BYTES = 64;
const SPRITE_PROPERTIES_STRIDE_BYTES = 80;

interface ResolvedSprite {
    ownerId: number;
    sprite: SceneSprite; // live reference - re-read on every properties update, not just captured at resolve time
    layer: number;
    isActive: boolean;
    transformBuffer: GPUBuffer;
    propertiesBuffer: GPUBuffer;
    bindGroup: GPUBindGroup;
    lastResolvedImage: string; // image currently baked into bindGroup's texture
    pendingImage: string | null; // image an in-flight refreshTexture() call is resolving, if any
}

/**
 * A single shared render pipeline draws every sprite (one draw call per
 * visible sprite - not one pipeline per shape, not per sprite). Per-sprite
 * transform and properties data is written once per scene rebuild by default;
 * a sprite (or its owner's transform) marked dynamic/dirty via LitboxScene
 * gets its corresponding buffer rewritten every affected frame instead. Only
 * the camera view-projection uniform (bind group 0, owned by
 * LitboxSceneRenderer) is unconditionally rewritten every frame.
 *
 * Transform and properties are separate uniform buffers (not one combined
 * struct) precisely so a transform-only update (cascaded from a dynamic
 * ancestor) never has to touch the properties buffer, and vice versa.
 */
export class SpriteResources {
    private device: GPUDevice;
    private pipeline: GPURenderPipeline | null = null;
    private instanceBindGroupLayout: GPUBindGroupLayout | null = null;
    private lightmapBindGroupLayout: GPUBindGroupLayout | null = null;
    private vertexBuffer: GPUBuffer;
    private sprites: ResolvedSprite[] = [];
    private lightmapBindGroup: GPUBindGroup | null = null;
    private textureCache: TextureCache | null = null;

    constructor(device: GPUDevice) {
        this.device = device;
        this.vertexBuffer = getQuadVertexBuffer(device);
    }

    public initialize(cameraBindGroupLayout: GPUBindGroupLayout, hdrFormat: GPUTextureFormat): void {
        this.instanceBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
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
                bindGroupLayouts: [cameraBindGroupLayout, this.instanceBindGroupLayout, this.lightmapBindGroupLayout],
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
    ): Promise<void> {
        if (!this.instanceBindGroupLayout || !this.lightmapBindGroupLayout) {
            throw new Error('SpriteResources.initialize() must be called before updateFromScene().');
        }
        this.textureCache = textureCache;

        for (const resolved of this.sprites) {
            resolved.transformBuffer.destroy();
            resolved.propertiesBuffer.destroy();
        }

        const lightmapView = simulationResources.getLightmapView();
        this.lightmapBindGroup = this.device.createBindGroup({
            layout: this.lightmapBindGroupLayout,
            entries: [
                { binding: 0, resource: lightmapView ?? textureCache.getBlackTexture().createView() },
                { binding: 1, resource: simulationResources.getSampler() },
            ],
        });

        this.sprites = await Promise.all(scene.sprites.map(sprite => this.resolveSprite(sprite, sceneGraph, textureCache)));
        this.sprites.sort((a, b) => a.layer - b.layer);
    }

    public draw(passEncoder: GPURenderPassEncoder, layerFilter: (layer: number) => boolean): void {
        if (!this.pipeline || !this.lightmapBindGroup) {
            return;
        }
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setVertexBuffer(0, this.vertexBuffer);
        passEncoder.setBindGroup(2, this.lightmapBindGroup);

        for (const resolved of this.sprites) {
            if (!resolved.isActive || !layerFilter(resolved.layer)) {
                continue;
            }
            passEncoder.setBindGroup(1, resolved.bindGroup);
            passEncoder.draw(QUAD_VERTEX_COUNT);
        }
    }

    /**
     * Destroys the GPU buffers for every sprite owned by an id in `ownerIds` and drops them from
     * the draw list. Unlike updateFromScene, this never touches any surviving sprite's buffers,
     * bind group, or texture - the targeted counterpart for structural destroy ops.
     */
    public removeByOwnerIds(ownerIds: Set<number>): void {
        const kept: ResolvedSprite[] = [];
        for (const resolved of this.sprites) {
            if (!ownerIds.has(resolved.ownerId)) {
                kept.push(resolved);
                continue;
            }
            resolved.transformBuffer.destroy();
            resolved.propertiesBuffer.destroy();
        }
        this.sprites = kept;
    }

    /** Targeted re-upload of the transform (and CPU-side active-flag) for every sprite owned by `ownerId`. */
    public refreshTransform(ownerId: number, sceneGraph: SceneGraph): void {
        for (const resolved of this.sprites) {
            if (resolved.ownerId !== ownerId) {
                continue;
            }
            // isActive is read directly by draw()'s cull check, not baked into a GPU buffer.
            resolved.isActive = sceneGraph.isActiveInHierarchy(ownerId);
            const worldTransform = sceneGraph.getWorldTransform(ownerId);
            this.writeTransformData(resolved.transformBuffer, worldTransform);
        }
    }

    /** Targeted re-upload of the properties (and, if changed, the texture) for every sprite owned by `ownerId`. */
    public refreshProperties(ownerId: number): void {
        for (const resolved of this.sprites) {
            if (resolved.ownerId !== ownerId) {
                continue;
            }
            this.writePropertiesData(resolved.propertiesBuffer, resolved.sprite, this.resolveShapeId(resolved.sprite));

            const targetImage = resolved.sprite.image;
            if (targetImage !== resolved.lastResolvedImage && resolved.pendingImage !== targetImage) {
                resolved.pendingImage = targetImage;
                void this.refreshTexture(resolved);
            }
        }
    }

    private async resolveSprite(sprite: SceneSprite, sceneGraph: SceneGraph, textureCache: TextureCache): Promise<ResolvedSprite> {
        const worldTransform = sceneGraph.getWorldTransform(sprite.ownerId);
        const isActive = sceneGraph.isActiveInHierarchy(sprite.ownerId);
        const texture = await textureCache.resolve(sprite.image, 'white');
        const shapeId = this.resolveShapeId(sprite);

        const transformBuffer = this.device.createBuffer({
            size: SPRITE_TRANSFORM_STRIDE_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.writeTransformData(transformBuffer, worldTransform);

        const propertiesBuffer = this.device.createBuffer({
            size: SPRITE_PROPERTIES_STRIDE_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.writePropertiesData(propertiesBuffer, sprite, shapeId);

        const bindGroup = this.device.createBindGroup({
            layout: this.instanceBindGroupLayout!,
            entries: [
                { binding: 0, resource: { buffer: transformBuffer } },
                { binding: 1, resource: { buffer: propertiesBuffer } },
                { binding: 2, resource: texture.createView() },
                { binding: 3, resource: textureCache.trilinearClamped },
            ],
        });

        return {
            ownerId: sprite.ownerId,
            sprite,
            layer: sprite.layer,
            isActive,
            transformBuffer,
            propertiesBuffer,
            bindGroup,
            lastResolvedImage: sprite.image,
            pendingImage: null,
        };
    }

    /**
     * Resolves a sprite's newly-assigned image and swaps its bind group in once ready.
     * Fire-and-forget from refreshProperties (not awaited by the render loop):
     * `draw()` keeps using the old bind group - still valid, still bound to a live
     * texture - until this completes, so there's no flicker or invalid-binding window.
     */
    private async refreshTexture(resolved: ResolvedSprite): Promise<void> {
        const targetImage = resolved.sprite.image;
        const texture = await this.textureCache!.resolve(targetImage, 'white');

        if (resolved.sprite.image !== targetImage) {
            // Superseded by a newer image change while this resolve was in flight;
            // refreshProperties will have kicked off (or will kick off) a fresh
            // refreshTexture for whatever the current target is.
            resolved.pendingImage = null;
            return;
        }

        resolved.bindGroup = this.device.createBindGroup({
            layout: this.instanceBindGroupLayout!,
            entries: [
                { binding: 0, resource: { buffer: resolved.transformBuffer } },
                { binding: 1, resource: { buffer: resolved.propertiesBuffer } },
                { binding: 2, resource: texture.createView() },
                { binding: 3, resource: this.textureCache!.trilinearClamped },
            ],
        });
        resolved.lastResolvedImage = targetImage;
        resolved.pendingImage = null;
    }

    private resolveShapeId(sprite: SceneSprite): number {
        let shapeId = PRIMITIVE_SHAPE_ID[sprite.primitiveShape];
        if (shapeId === undefined) {
            console.warn(`Litbox: unrecognized primitiveShape "${sprite.primitiveShape}" on sprite owner ${sprite.ownerId}; treating as unspecified.`);
            shapeId = PRIMITIVE_SHAPE_ID[''];
        }
        return shapeId;
    }

    private writeTransformData(buffer: GPUBuffer, worldTransform: mat4): void {
        const data = new Float32Array(SPRITE_TRANSFORM_STRIDE_BYTES / 4);
        data.set(worldTransform as Float32Array, 0);
        this.device.queue.writeBuffer(buffer, 0, data);
    }

    private writePropertiesData(buffer: GPUBuffer, sprite: SceneSprite, shapeId: number): void {
        const data = new ArrayBuffer(SPRITE_PROPERTIES_STRIDE_BYTES);
        const floats = new Float32Array(data);
        const view = new DataView(data);

        floats.set([sprite.ambient.r, sprite.ambient.g, sprite.ambient.b, sprite.ambient.a], 0);
        floats.set([sprite.emissive.r, sprite.emissive.g, sprite.emissive.b, sprite.emissive.a], 4);
        floats.set([sprite.simContribution.r, sprite.simContribution.g, sprite.simContribution.b, sprite.simContribution.a], 8);
        floats.set([sprite.colorMod.r, sprite.colorMod.g, sprite.colorMod.b, sprite.colorMod.a], 12);
        view.setFloat32(64, sprite.opacity, true);
        view.setFloat32(68, sprite.simBlur, true);
        view.setUint32(72, shapeId, true);
        // Bytes 76-79 are unused padding (WGSL rounds the struct up to a 16-byte multiple).

        this.device.queue.writeBuffer(buffer, 0, data);
    }
}
