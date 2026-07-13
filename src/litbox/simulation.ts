import { mat4 } from 'gl-matrix';
import type { Scene, SceneSimulation } from './scene.ts';
import type { SceneGraph } from './scene_graph.ts';
import type { RaytracedResources } from './raytraced_resources.ts';
import { QUAD_VERTEX_COUNT, QUAD_VERTEX_BUFFER_LAYOUT, getQuadVertexBuffer } from './quad_mesh.ts';
import { ComputedDataManager, ComputedTexture, ComputedBuffer } from './computed_data_manager.ts';
import { ConvertPhotonIrradianceToHdrOperation } from './convert_photon_irradiance_to_hdr.ts';
import compositeShaderCode from './shaders/simulation_composite.wgsl?raw';

const LIGHTMAP_FORMAT: GPUTextureFormat = 'rgba16float';

/**
 * Owns the HDR mipmapped lightmap produced by the light simulation, and the pipeline that
 * additively composites it into the HDR frame buffer as a world-space quad.
 *
 * The photon-tracing pass itself (emission/bounce/accumulation) is still deferred - what run()
 * does today is the far end of that pipeline: it converts whatever is currently sitting in the
 * photon-receptor buffer (photonBuffer, an atomic accumulator a future tracer will write into)
 * into the lightmap's mip 0 via ConvertPhotonIrradianceToHdrOperation, combining it with the
 * albedo/density G-Buffer. Until the real tracer exists, loadFromScene() fills the buffer with a
 * deterministic test pattern (see fillTestPhotonPattern) so the conversion has something
 * non-trivial to work with. Higher mips have no real content yet (no mip-chain generation from
 * mip 0), so they're just cleared each frame, as the whole lightmap used to be.
 */
export class SimulationResources {
    private device: GPUDevice;
    private computedDataManager: ComputedDataManager;
    private lightmap: ComputedTexture | null = null;
    private sampler: GPUSampler;

    private pipeline: GPURenderPipeline | null = null;
    private vertexBuffer: GPUBuffer;
    private compositeUniformBuffer: GPUBuffer | null = null;
    private compositeBindGroup: GPUBindGroup | null = null;
    private compositeBindGroupLayout: GPUBindGroupLayout | null = null;

    private simulation: SceneSimulation | null = null;
    private worldTransform: mat4 = mat4.create();

    /** Atomic accumulator a future photon tracer will write into: width*height*3 u32 entries (3 consecutive slots per pixel: R, G, B). */
    private photonBuffer: ComputedBuffer | null = null;
    private convertToHdr: ConvertPhotonIrradianceToHdrOperation;

    constructor(device: GPUDevice, computedDataManager: ComputedDataManager) {
        this.device = device;
        this.computedDataManager = computedDataManager;
        this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
        this.vertexBuffer = getQuadVertexBuffer(device);
        this.convertToHdr = new ConvertPhotonIrradianceToHdrOperation(device);
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
        return this.lightmap?.view ?? null;
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

    /**
     * Full teardown-and-rebuild of the lightmap (and photon buffer) from `scene`. Called only on
     * an actual scene load/swap (see LitboxSceneRenderer.rebuildFromScene, its only caller) - never
     * per-frame; a transform-only change instead goes through refreshWorldTransform.
     */
    public loadFromScene(scene: Scene, sceneGraph: SceneGraph): void {
        if (this.lightmap) {
            this.computedDataManager.releaseTexture(this.lightmap);
            this.lightmap = null;
        }
        if (this.photonBuffer) {
            this.computedDataManager.releaseBuffer(this.photonBuffer);
            this.photonBuffer = null;
        }
        this.simulation = scene.simulations.length > 0 ? scene.simulations[0] : null;

        if (scene.simulations.length > 1) {
            console.warn(`Litbox: ${scene.simulations.length} simulations present; only the first is rendered.`);
        }
        if (!this.simulation) {
            return;
        }

        this.worldTransform = sceneGraph.getWorldTransform(this.simulation.ownerId);

        const { width, height } = this.simulation;
        const mipLevelCount = Math.floor(Math.log2(Math.max(width, height))) + 1;
        this.lightmap = this.computedDataManager.acquireTexture(
            width,
            height,
            LIGHTMAP_FORMAT,
            GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING,
            mipLevelCount,
        );

        this.photonBuffer = this.computedDataManager.acquireBuffer(
            width * height * 3 * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        );
        this.fillTestPhotonPattern(width, height);

        if (this.compositeBindGroupLayout && this.compositeUniformBuffer) {
            this.compositeBindGroup = this.device.createBindGroup({
                layout: this.compositeBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.compositeUniformBuffer } },
                    { binding: 1, resource: this.lightmap.view },
                    { binding: 2, resource: this.sampler },
                ],
            });
            const worldTransformData = this.worldTransform as Float32Array;
            this.device.queue.writeBuffer(this.compositeUniformBuffer, 0, worldTransformData.buffer, worldTransformData.byteOffset, worldTransformData.byteLength);
        }
    }

    /**
     * Temporary stand-in for the not-yet-built photon tracer: writes a deterministic radial energy
     * falloff directly into the freshly-acquired photon buffer so
     * ConvertPhotonIrradianceToHdrOperation has something non-zero, and distinguishable per
     * channel, to convert. Replace once real photon emission/tracing lands.
     */
    private fillTestPhotonPattern(width: number, height: number): void {
        if (!this.photonBuffer) {
            return;
        }
        // Comfortably inside u32 range, with headroom left for the atomic adds a real tracer would
        // eventually layer on top - not meant to be physically meaningful, just visually distinguishable.
        const peak = 2 ** 28;
        const centerX = width / 2;
        const centerY = height / 2;
        const maxDist = Math.hypot(centerX, centerY);
        const data = new Uint32Array(width * height * 3);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const falloff = Math.max(0, 1 - Math.hypot(x - centerX, y - centerY) / maxDist);
                const base = (y * width + x) * 3;
                data[base] = Math.round(peak * falloff);
                data[base + 1] = Math.round(peak * falloff * 0.6);
                data[base + 2] = Math.round(peak * falloff * 0.3);
            }
        }
        this.device.queue.writeBuffer(this.photonBuffer.buffer, 0, data);
    }

    /**
     * Clears the lightmap's higher mips (no mip-chain generation exists yet), then converts the
     * photon-receptor buffer into mip 0 via ConvertPhotonIrradianceToHdrOperation, sourcing albedo
     * and density from raytracedResources' G-Buffer (rendered earlier this same frame).
     */
    public run(encoder: GPUCommandEncoder, raytracedResources: RaytracedResources): void {
        if (!this.lightmap) {
            return;
        }
        for (let mip = 1; mip < this.lightmap.mipLevelCount; mip++) {
            const pass = encoder.beginRenderPass({
                colorAttachments: [{ view: this.lightmap.getMipView(mip), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
            });
            pass.end();
        }

        if (!this.simulation || !this.photonBuffer) {
            return;
        }
        const albedoView = raytracedResources.getAlbedoView();
        const densityView = raytracedResources.getDensityView();
        if (!albedoView || !densityView) {
            return;
        }

        const { width, height } = this.simulation;
        this.convertToHdr.updateUniforms({ hdrScale: (width * height) / 0xFFFFFFFF });
        this.convertToHdr.updateInputs(this.photonBuffer.buffer, albedoView, densityView);
        this.convertToHdr.updateOutputs(this.lightmap.getMipView(0), width, height);
        this.convertToHdr.execute(encoder);
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
