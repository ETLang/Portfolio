import { mat4 } from 'gl-matrix';
import type { Scene, SceneSimulation } from './scene.ts';
import type { SceneGraph } from './scene_graph.ts';
import { QUAD_VERTEX_COUNT, QUAD_VERTEX_BUFFER_LAYOUT, getQuadVertexBuffer } from './quad_mesh.ts';
import compositeShaderCode from './shaders/simulation_composite.wgsl?raw';

const LIGHTMAP_FORMAT: GPUTextureFormat = 'rgba16float';

/**
 * Owns the HDR mipmapped lightmap produced by the (currently stubbed)
 * raytraced light simulation, and the pipeline that additively composites
 * it into the HDR frame buffer as a world-space quad. The real photon
 * integration is deferred - run() just clears the lightmap each frame so
 * downstream consumers (sprites, the composite pass) have well-defined,
 * safely-samplable content across every mip level.
 */
export class SimulationResources {
    private device: GPUDevice;
    private lightmapTexture: GPUTexture | null = null;
    private lightmapView: GPUTextureView | null = null;
    private mipViews: GPUTextureView[] = [];
    private sampler: GPUSampler;

    private pipeline: GPURenderPipeline | null = null;
    private vertexBuffer: GPUBuffer;
    private compositeUniformBuffer: GPUBuffer | null = null;
    private compositeBindGroup: GPUBindGroup | null = null;
    private compositeBindGroupLayout: GPUBindGroupLayout | null = null;

    private simulation: SceneSimulation | null = null;
    private worldTransform: mat4 = mat4.create();

    constructor(device: GPUDevice) {
        this.device = device;
        this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
        this.vertexBuffer = getQuadVertexBuffer(device);
    }

    public initialize(cameraBindGroupLayout: GPUBindGroupLayout): void {
        this.compositeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ],
        });

        const shaderModule = this.device.createShaderModule({ code: compositeShaderCode });
        this.pipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [cameraBindGroupLayout, this.compositeBindGroupLayout] }),
            vertex: {
                module: shaderModule,
                entryPoint: 'vertex_main',
                buffers: [QUAD_VERTEX_BUFFER_LAYOUT],
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragment_main',
                targets: [{
                    format: LIGHTMAP_FORMAT,
                    blend: {
                        color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                    },
                }],
            },
            primitive: { topology: 'triangle-list' },
        });

        this.compositeUniformBuffer = this.device.createBuffer({
            size: 4 * 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    /** The lightmap's full mip chain, for sprites/composite to sample. */
    public getLightmapView(): GPUTextureView | null {
        return this.lightmapView;
    }

    public getSampler(): GPUSampler {
        return this.sampler;
    }

    /** World transform of the simulation's owner, used by sprites to derive their lightmap UV. */
    public getWorldTransform(): mat4 {
        return this.worldTransform;
    }

    public hasSimulation(): boolean {
        return this.simulation !== null;
    }

    /** Owner id of the current simulation, or null if none - lets callers cheaply check subtree membership. */
    public getOwnerId(): number | null {
        return this.simulation?.ownerId ?? null;
    }

    /** Targeted re-derivation of the composite uniform's world transform and its (already transform-only) GPU upload. */
    public refreshWorldTransform(sceneGraph: SceneGraph): void {
        if (!this.simulation || !this.compositeUniformBuffer) {
            return;
        }
        this.worldTransform = sceneGraph.getWorldTransform(this.simulation.ownerId);
        const worldTransformData = this.worldTransform as Float32Array;
        this.device.queue.writeBuffer(this.compositeUniformBuffer, 0, worldTransformData.buffer, worldTransformData.byteOffset, worldTransformData.byteLength);
    }

    public updateFromScene(scene: Scene, sceneGraph: SceneGraph): void {
        this.lightmapTexture?.destroy();
        this.lightmapTexture = null;
        this.lightmapView = null;
        this.mipViews = [];
        this.simulation = scene.simulations.length > 0 ? scene.simulations[0] : null;

        if (scene.simulations.length > 1) {
            console.warn(`Litbox: ${scene.simulations.length} simulations present; only the first is rendered.`);
        }
        if (!this.simulation) {
            return;
        }

        this.worldTransform = sceneGraph.getWorldTransform(this.simulation.ownerId);

        const mipLevelCount = Math.floor(Math.log2(Math.max(this.simulation.width, this.simulation.height))) + 1;
        this.lightmapTexture = this.device.createTexture({
            size: [this.simulation.width, this.simulation.height],
            format: LIGHTMAP_FORMAT,
            mipLevelCount,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.lightmapView = this.lightmapTexture.createView();
        for (let mip = 0; mip < mipLevelCount; mip++) {
            this.mipViews.push(this.lightmapTexture.createView({ baseMipLevel: mip, mipLevelCount: 1 }));
        }

        if (this.compositeBindGroupLayout && this.compositeUniformBuffer) {
            this.compositeBindGroup = this.device.createBindGroup({
                layout: this.compositeBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.compositeUniformBuffer } },
                    { binding: 1, resource: this.lightmapView },
                    { binding: 2, resource: this.sampler },
                ],
            });
            const worldTransformData = this.worldTransform as Float32Array;
            this.device.queue.writeBuffer(this.compositeUniformBuffer, 0, worldTransformData.buffer, worldTransformData.byteOffset, worldTransformData.byteLength);
        }
    }

    /** Stub: clears every mip of the lightmap. Real photon integration is deferred. */
    public run(encoder: GPUCommandEncoder): void {
        for (const view of this.mipViews) {
            const pass = encoder.beginRenderPass({
                colorAttachments: [{ view, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
            });
            pass.end();
        }
    }

    /** Additively blends the lightmap into the current render pass as a world-space quad. No exposure applied here. */
    public compositeInto(passEncoder: GPURenderPassEncoder): void {
        if (!this.pipeline || !this.compositeBindGroup || !this.simulation) {
            return;
        }
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(1, this.compositeBindGroup);
        passEncoder.setVertexBuffer(0, this.vertexBuffer);
        passEncoder.draw(QUAD_VERTEX_COUNT);
    }
}
