import { mat4 } from 'gl-matrix';
import type { AnyLight, LightKind, Scene, SceneSimulation } from './scene.ts';
import type { SceneGraph } from './scene_graph.ts';
import type { RaytracedResources } from './raytraced_resources.ts';
import type { LightResources } from './light_resources.ts';
import type { LutResources } from './lut_resources.ts';
import { QUAD_VERTEX_COUNT, QUAD_VERTEX_BUFFER_LAYOUT, getQuadVertexBuffer } from './quad_mesh.ts';
import { ComputedDataManager, ComputedTexture, ComputedBuffer } from './computed_data_manager.ts';
import { ConvertPhotonIrradianceToHdrOperation } from './convert_photon_irradiance_to_hdr.ts';
import {
    ForwardMonteCarloOperation,
    luminance,
    computeRayCount,
    resolveBounces,
    computeWorldToTargetPixels,
    computeLightToTarget,
    computeDirectionalLightDirection,
    combineWriteCount,
} from './forward_monte_carlo.ts';
import compositeShaderCode from './shaders/simulation_composite.wgsl?raw';
import { preprocessShader } from './shaders/shader_preprocessor.ts';

const LIGHTMAP_FORMAT: GPUTextureFormat = 'rgba16float';

/**
 * Owns the HDR mipmapped lightmap produced by the light simulation, and the pipeline that
 * additively composites it into the HDR frame buffer as a world-space quad.
 *
 * run() drives the whole per-frame simulation pipeline: tracePhotons() dispatches the
 * ForwardMonteCarloOperation tracer once per light instance (see forward_monte_carlo.ts/.wgsl) to
 * populate photonBuffer (an atomic accumulator), then ConvertPhotonIrradianceToHdrOperation
 * converts that into the lightmap's mip 0, combining it with the albedo/density G-Buffer. This is
 * a single-pass, non-accumulating integration - photonBuffer is cleared every frame, not
 * progressively converged across frames (see tracePhotons). Higher mips have no real content yet
 * (no mip-chain generation from mip 0), so they're just cleared each frame.
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

    /** Atomic accumulator the photon tracer writes into: width*height*3 u32 entries (3 consecutive slots per pixel: R, G, B). */
    private photonBuffer: ComputedBuffer | null = null;
    /** Cached zeroed data matching photonBuffer's size, reused every frame to clear it - see run(). */
    private photonBufferClearData: Uint32Array | null = null;
    private convertToHdr: ConvertPhotonIrradianceToHdrOperation;

    /** One ForwardMonteCarloOperation per light kind - see its class doc. Constructed in initialize() once lutResources exists. */
    private pointOperation: ForwardMonteCarloOperation | null = null;
    private spotOperation: ForwardMonteCarloOperation | null = null;
    private laserOperation: ForwardMonteCarloOperation | null = null;
    private directionalOperation: ForwardMonteCarloOperation | null = null;
    private ambientOperation: ForwardMonteCarloOperation | null = null;

    /**
     * Ported from Unity's _forwardWriteCounterBuffer/g_write_counter: a lifetime (never cleared by
     * a scene load or per-frame photon-buffer clear) manual-uint64 photon-writes counter, shared
     * by all 5 ForwardMonteCarloOperation instances - see getWriteCount() and
     * forward_monte_carlo.wgsl's file header.
     */
    private writeCounterBuffer: GPUBuffer;
    private writeCounterStaging: GPUBuffer;

    constructor(device: GPUDevice, computedDataManager: ComputedDataManager) {
        this.device = device;
        this.computedDataManager = computedDataManager;
        this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
        this.vertexBuffer = getQuadVertexBuffer(device);
        this.convertToHdr = new ConvertPhotonIrradianceToHdrOperation(device);

        this.writeCounterBuffer = device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        this.writeCounterStaging = device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
    }

    public initialize(cameraBindGroupLayout: GPUBindGroupLayout, lutResources: LutResources): void {
        this.pointOperation = new ForwardMonteCarloOperation(this.device, 'point', lutResources);
        this.spotOperation = new ForwardMonteCarloOperation(this.device, 'spot', lutResources);
        this.laserOperation = new ForwardMonteCarloOperation(this.device, 'laser', lutResources);
        this.directionalOperation = new ForwardMonteCarloOperation(this.device, 'directional', lutResources);
        this.ambientOperation = new ForwardMonteCarloOperation(this.device, 'ambient', lutResources);
        this.compositeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ],
        });

        const shaderModule = this.device.createShaderModule({ code: preprocessShader(compositeShaderCode) });
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
        this.photonBufferClearData = null;
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
        this.photonBufferClearData = new Uint32Array(width * height * 3);

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
     * Clears the lightmap's higher mips (no mip-chain generation exists yet), dispatches the
     * ForwardMonteCarlo photon tracer once per light instance (see tracePhotons), then converts the
     * resulting photon-receptor buffer into lightmap mip 0 via ConvertPhotonIrradianceToHdrOperation,
     * sourcing albedo/density from raytracedResources' G-Buffer (rendered earlier this same frame).
     */
    public run(
        encoder: GPUCommandEncoder,
        raytracedResources: RaytracedResources,
        lightResources: LightResources,
        lutResources: LutResources,
        sceneGraph: SceneGraph,
    ): void {
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
        const normalRoughnessView = raytracedResources.getNormalRoughnessView();
        if (!albedoView || !densityView || !normalRoughnessView) {
            return;
        }

        this.tracePhotons(encoder, lightResources, lutResources, sceneGraph, albedoView, densityView, normalRoughnessView);

        const { width, height } = this.simulation;
        this.convertToHdr.updateUniforms({ hdrScale: (width * height) / 0xFFFFFFFF });
        this.convertToHdr.updateInputs(this.photonBuffer.buffer, albedoView, densityView);
        this.convertToHdr.updateOutputs(this.lightmap.getMipView(0), width, height);
        this.convertToHdr.execute(encoder);
    }

    /**
     * Dispatches the ForwardMonteCarlo tracer once per light instance in the scene (not
     * deduplicated by kind - matches Unity's ForwardMonteCarlo.Integrate/SimulateLight), each ray
     * budget luminance-weighted from `simulation.raysPerFrame`. See forward_monte_carlo.ts for the
     * per-light math this mirrors, and forward_monte_carlo.wgsl for the actual integration.
     */
    private tracePhotons(
        encoder: GPUCommandEncoder,
        lightResources: LightResources,
        lutResources: LutResources,
        sceneGraph: SceneGraph,
        albedoView: GPUTextureView,
        densityView: GPUTextureView,
        normalRoughnessView: GPUTextureView,
    ): void {
        if (!this.simulation || !this.photonBuffer || !this.photonBufferClearData
            || !this.pointOperation || !this.spotOperation || !this.laserOperation
            || !this.directionalOperation || !this.ambientOperation) {
            return;
        }
        // Clear every frame - matches Unity's Realtime-mode per-frame Clear()/NewScene() (this is a
        // single-pass, non-accumulating integration - see this project's plan).
        this.device.queue.writeBuffer(this.photonBuffer.buffer, 0, this.photonBufferClearData);

        const { width, height, raysPerFrame, integrationInterval: integrationIntervalRatio, photonBounces } = this.simulation;

        interface Entry {
            kind: LightKind;
            light: AnyLight;
            operation: ForwardMonteCarloOperation;
            pinch: number;
        }
        const entries: Entry[] = [
            ...lightResources.getPointLights().filter((light)  => sceneGraph.isActiveInHierarchy(light.ownerId)).map((light): Entry => ({ kind: 'point', light, operation: this.pointOperation!, pinch: 0 })),
            ...lightResources.getSpotlights().filter((light)  => sceneGraph.isActiveInHierarchy(light.ownerId)).map((light): Entry => ({ kind: 'spot', light, operation: this.spotOperation!, pinch: light.pinch })),
            ...lightResources.getLaserLights().filter((light)  => sceneGraph.isActiveInHierarchy(light.ownerId)).map((light): Entry => ({ kind: 'laser', light, operation: this.laserOperation!, pinch: 0 })),
            ...lightResources.getDirectionalLights().filter((light)  => sceneGraph.isActiveInHierarchy(light.ownerId)).map((light): Entry => ({ kind: 'directional', light, operation: this.directionalOperation!, pinch: 0 })),
            ...lightResources.getAmbientLights().filter((light)  => sceneGraph.isActiveInHierarchy(light.ownerId)).map((light): Entry => ({ kind: 'ambient', light, operation: this.ambientOperation!, pinch: 0 })),
        ];
        if (entries.length === 0) {
            return;
        }

        const energies = entries.map((entry) => {
            const { color, intensity } = entry.light;
            // NOT intensity² here - the Unity scene exporter already writes intensity² into the
            // JSON (LitboxDemoSceneExporter.cs), so this project's `intensity` is already squared.
            const energyRgb: [number, number, number] = [color.r * intensity, color.g * intensity, color.b * intensity];
            return { luma: luminance(energyRgb), energyRgb };
        });
        const totalLuma = energies.reduce((sum, energy) => sum + energy.luma, 0);
        if (totalLuma === 0) {
            return;
        }

        const integrationInterval = Math.max(1, integrationIntervalRatio * height);
        const integrationIntervalSquared = integrationInterval * integrationInterval;

        // Resolve every light's ray budget and cumulative seedBase first, so the shared random
        // seed buffer can be sized to the sum across all lights before any dispatch touches it.
        const rays: number[] = [];
        const seedBases: number[] = [];
        let seedBase = 0;
        for (const { luma } of energies) {
            const lightRays = computeRayCount(luma, totalLuma, raysPerFrame);
            rays.push(lightRays);
            seedBases.push(seedBase);
            seedBase += lightRays;
        }
        const seedBuffer = this.computedDataManager.acquireRandomSeedBuffer(seedBase);

        const worldToTargetPixels = computeWorldToTargetPixels(sceneGraph.getWorldTransform(this.simulation.ownerId), width, height);

        for (let i = 0; i < entries.length; i++) {
            const { kind, light, operation, pinch } = entries[i];
            const { energyRgb } = energies[i];

            const lightWorldTransform = sceneGraph.getWorldTransform(light.ownerId);
            const lightToTarget = computeLightToTarget(worldToTargetPixels, lightWorldTransform);
            const directionalLightDirection: [number, number] = kind === 'directional'
                ? computeDirectionalLightDirection(lightToTarget)
                : [0, 0];
            const pinchSquared = pinch * pinch;
            const lightPinch: [number, number] = kind === 'spot' ? [pinchSquared, Math.atan(pinchSquared)] : [0, 0];

            const photonEnergyScale = 0xFFFFFFFF / rays[i] / integrationInterval;
            const lightEnergy: [number, number, number] = [
                energyRgb[0] * photonEnergyScale,
                energyRgb[1] * photonEnergyScale,
                energyRgb[2] * photonEnergyScale,
            ];
            const bounces = resolveBounces(photonBounces, light.bounces);

            operation.updateUniforms({
                lightToTarget,
                lightEnergy,
                bounces,
                seedBase: seedBases[i],
                directionalLightDirection,
                lightPinch,
                integrationInterval,
                integrationIntervalSquared,
                rays: rays[i],
            });
            operation.updateInputs(seedBuffer.buffer, albedoView, densityView, normalRoughnessView, lutResources);
            operation.updateOutputs(this.photonBuffer.buffer, this.writeCounterBuffer);
            operation.execute(encoder);
        }
    }

    /**
     * Reads the lifetime photon-writes counter back from the GPU (see writeCounterBuffer's doc
     * comment) - self-contained (its own tiny command encoder/submit), not driven by the main
     * render loop. Intended for a future portfolio-page "MWrites/s" display: callers poll this at
     * whatever cadence they want and compute their own rate from consecutive reads, exactly like
     * Unity's UpdatePerformanceMetrics() does.
     */
    public async getWriteCount(): Promise<bigint> {
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.writeCounterBuffer, 0, this.writeCounterStaging, 0, 8);
        this.device.queue.submit([encoder.finish()]);

        await this.device.queue.onSubmittedWorkDone();
        await this.writeCounterStaging.mapAsync(GPUMapMode.READ);
        const data = new Uint32Array(this.writeCounterStaging.getMappedRange());
        const result = combineWriteCount(data[0], data[1]);
        this.writeCounterStaging.unmap();
        return result;
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
