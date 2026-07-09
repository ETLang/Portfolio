import type { mat4 } from 'gl-matrix';
import type { Scene, SceneSprite } from './scene.ts';
import type { SceneGraph } from './scene_graph.ts';
import type { TextureCache } from './texture_cache.ts';
import type { SimulationResources } from './simulation.ts';
import spriteShaderCode from './shaders/sprite.wgsl?raw';

// erasableSyntaxOnly forbids `enum` - matches shapeId encoding in sprite.wgsl.
const PRIMITIVE_SHAPE_ID: Record<string, number> = { '': 0, rect: 1, ellipse: 2 };

// Must match the SpriteInstance struct layout in sprite.wgsl.
const SPRITE_INSTANCE_STRIDE_BYTES = 144;

// Unit quad, two triangles, matching SceneObject.scale semantics ([-0.5, 0.5]^2 local space).
const QUAD_VERTICES = new Float32Array([
    -0.5, -0.5,
    0.5, -0.5,
    0.5, 0.5,
    -0.5, -0.5,
    0.5, 0.5,
    -0.5, 0.5,
]);

interface ResolvedSprite {
    layer: number;
    isActive: boolean;
    uniformBuffer: GPUBuffer;
    bindGroup: GPUBindGroup;
}

/**
 * A single shared render pipeline draws every sprite (one draw call per
 * visible sprite - not one pipeline per shape, not per sprite). Per-sprite
 * data is written once per scene rebuild, not per frame; only the camera
 * view-projection uniform (bind group 0, owned by LitboxSceneRenderer)
 * changes every frame.
 */
export class SpriteResources {
    private device: GPUDevice;
    private pipeline: GPURenderPipeline | null = null;
    private instanceBindGroupLayout: GPUBindGroupLayout | null = null;
    private lightmapBindGroupLayout: GPUBindGroupLayout | null = null;
    private vertexBuffer: GPUBuffer;
    private sprites: ResolvedSprite[] = [];
    private lightmapBindGroup: GPUBindGroup | null = null;

    constructor(device: GPUDevice) {
        this.device = device;
        this.vertexBuffer = device.createBuffer({
            size: QUAD_VERTICES.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.vertexBuffer.getMappedRange()).set(QUAD_VERTICES);
        this.vertexBuffer.unmap();
    }

    public initialize(cameraBindGroupLayout: GPUBindGroupLayout, hdrFormat: GPUTextureFormat): void {
        this.instanceBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
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
                buffers: [{ arrayStride: 4 * 2, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] }],
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

        for (const resolved of this.sprites) {
            resolved.uniformBuffer.destroy();
        }

        const lightmapView = simulationResources.getLightmapView();
        this.lightmapBindGroup = this.device.createBindGroup({
            layout: this.lightmapBindGroupLayout,
            entries: [
                { binding: 0, resource: lightmapView ?? textureCache.getBlackFallback().createView() },
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
            passEncoder.draw(6);
        }
    }

    private async resolveSprite(sprite: SceneSprite, sceneGraph: SceneGraph, textureCache: TextureCache): Promise<ResolvedSprite> {
        const worldTransform = sceneGraph.getWorldTransform(sprite.ownerId);
        const isActive = sceneGraph.isActiveInHierarchy(sprite.ownerId);
        const texture = await textureCache.resolve(sprite.image, 'white');

        let shapeId = PRIMITIVE_SHAPE_ID[sprite.primitiveShape];
        if (shapeId === undefined) {
            console.warn(`Litbox: unrecognized primitiveShape "${sprite.primitiveShape}" on sprite owner ${sprite.ownerId}; treating as unspecified.`);
            shapeId = PRIMITIVE_SHAPE_ID[''];
        }

        const uniformBuffer = this.device.createBuffer({
            size: SPRITE_INSTANCE_STRIDE_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.writeInstanceData(uniformBuffer, worldTransform, sprite, shapeId);

        const bindGroup = this.device.createBindGroup({
            layout: this.instanceBindGroupLayout!,
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: texture.createView() },
                { binding: 2, resource: textureCache.sampler },
            ],
        });

        return { layer: sprite.layer, isActive, uniformBuffer, bindGroup };
    }

    private writeInstanceData(buffer: GPUBuffer, worldTransform: mat4, sprite: SceneSprite, shapeId: number): void {
        const data = new ArrayBuffer(SPRITE_INSTANCE_STRIDE_BYTES);
        const floats = new Float32Array(data);
        const view = new DataView(data);

        floats.set(worldTransform as Float32Array, 0);
        floats.set([sprite.ambient.r, sprite.ambient.g, sprite.ambient.b, sprite.ambient.a], 16);
        floats.set([sprite.emissive.r, sprite.emissive.g, sprite.emissive.b, sprite.emissive.a], 20);
        floats.set([sprite.simContribution.r, sprite.simContribution.g, sprite.simContribution.b, sprite.simContribution.a], 24);
        floats.set([sprite.colorMod.r, sprite.colorMod.g, sprite.colorMod.b, sprite.colorMod.a], 28);
        view.setFloat32(128, sprite.opacity, true);
        view.setFloat32(132, sprite.simBlur, true);
        view.setUint32(136, shapeId, true);
        // Bytes 140-143 are unused padding (WGSL rounds the struct up to a 16-byte multiple).

        this.device.queue.writeBuffer(buffer, 0, data);
    }
}
