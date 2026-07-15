import { mat4 } from 'gl-matrix';
import type { AnyLight, LightKind, Scene, SceneSimulation } from './scene.ts';
import type { SceneGraph } from './scene_graph.ts';
import type { RaytracedResources } from './raytraced_resources.ts';
import type { LightResources } from './light_resources.ts';
import type { LutResources } from './lut_resources.ts';
import { QUAD_VERTEX_COUNT, QUAD_VERTEX_BUFFER_LAYOUT, getQuadVertexBuffer } from './quad_mesh.ts';
import { ComputedDataManager, ComputedTexture, ComputedBuffer } from './computed_data_manager.ts';
import { ConvertPhotonIrradianceToHdrOperation } from './convert_photon_irradiance_to_hdr.ts';
import { MipDownsampleOperation } from './mip_downsample.ts';
import { DensityMipBlitResources } from './density_mip_blit.ts';
import { ComputeVarianceAndMipsOperation } from './compute_variance_and_mips.ts';
import { FilterVarianceOperation } from './filter_variance.ts';
import { DenoiseOperation } from './denoise_operation.ts';
import { ComputeVolatilityOperation } from './compute_volatility.ts';
import { BuildDenoiserQuadtreeOperation } from './build_denoiser_quadtree.ts';
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
import { getPlatform, isRandomAccessFriendlyGpu, type Platform } from './device_environment.ts';
import { srgbToLinear } from './color_space.ts';

const LIGHTMAP_FORMAT: GPUTextureFormat = 'rgba16float';
// r32float, not r16float - r16float isn't a valid WGSL storage-texture texel format (see
// ComputeVarianceAndMipsOperation/FilterVarianceOperation, both of which textureStore into this).
const VARIANCE_FORMAT: GPUTextureFormat = 'r32float';

/**
 * How a scene's raw SceneSimulation config gets scaled down for a given device - see
 * deriveEffectiveSimulation and CLAUDE.md/mobile-perf-tuning notes for the reasoning behind each
 * number. Desktop is always the identity profile (resolutionScale/raysPerFrameScale of 1,
 * bilinear splat on) - only mobile gets scaled down, and how much depends on whether the GPU is
 * expected to handle this integrator's scattered/incoherent access pattern well:
 *
 * - Mobile, GPU NOT known random-access-friendly (the default assumption for mobile): halve
 *   resolution (small screen, so the loss is barely visible), quarter raysPerFrame (a lower
 *   resolution needs proportionally fewer photon writes for similar apparent fidelity - see
 *   maxIntegrationSteps's own resolution-derived scaling for why the step budget doesn't need an
 *   independent scale factor), and disable the bilinear photon splat (measured ~1.5x photons/s win
 *   from cutting scattered global atomicAdds 4x on the Pixel 10 Pro's PowerVR GPU).
 * - Mobile, GPU known random-access-friendly (e.g. Apple Silicon): still halve resolution (same
 *   small-screen argument), but only halve raysPerFrame (not quarter) and keep the bilinear splat
 *   on, since the GPU shouldn't be paying the same atomic-contention tax the default profile is
 *   working around.
 */
export interface SimulationDeviceProfile {
    resolutionScale: number;
    raysPerFrameScale: number;
    bilinearPhotonDistribution: boolean;
}

export function getSimulationDeviceProfile(platform: Platform, gpuRandomAccessFriendly: boolean): SimulationDeviceProfile {
    if (platform === 'desktop') {
        return { resolutionScale: 1, raysPerFrameScale: 1, bilinearPhotonDistribution: true };
    }
    return gpuRandomAccessFriendly
        ? { resolutionScale: 0.5, raysPerFrameScale: 0.5, bilinearPhotonDistribution: true }
        : { resolutionScale: 0.5, raysPerFrameScale: 0.25, bilinearPhotonDistribution: false };
}

/**
 * Per-bounce-phase ray-march step cap: one domain-diagonal march. forward_monte_carlo.wgsl's
 * integrate() runs its search phase and refine phase as two *separate* invocations of the same
 * steps-for-loop (each resets steps to 0), each independently bounded by that phase's own uEscape
 * (a ray-vs-box exit distance) - so a single diagonal covers either phase's worst case, not both
 * combined. Always derived from whatever width/height are actually in play (not a stored field)
 * so it automatically tracks resolutionScale - a fixed constant would either waste budget (too
 * high for a scaled-down domain) or truncate a legitimate search/refine phase (too low).
 */
export function computeMaxIntegrationSteps(width: number, height: number): number {
    return Math.sqrt(width * width + height * height);
}

/** Applies `profile` to a scene's raw simulation config - pure, doesn't mutate `simulation`. */
export function deriveEffectiveSimulation(simulation: SceneSimulation, profile: SimulationDeviceProfile): SceneSimulation {
    return {
        ...simulation,
        width: Math.max(1, Math.round(simulation.width * profile.resolutionScale)),
        height: Math.max(1, Math.round(simulation.height * profile.resolutionScale)),
        raysPerFrame: Math.max(1, Math.round(simulation.raysPerFrame * profile.raysPerFrameScale)),
    };
}

/**
 * Every tunable threshold in the denoiser pipeline (both DenoiseOperation's guided-blur weights
 * and BuildDenoiserQuadtreeOperation's split-quadtree bake), bundled into one flat object so the
 * portfolio page's config UI (see main.ts's denoiser tunables panel) has a single source of truth
 * to read/write - mutate a field directly (e.g. `simulationResources.denoiserTunables.varianceScale
 * = 4`); run()/buildDenoiserQuadtree() read this fresh every frame, so there's no separate "apply"
 * step. None of these are solved constants - see this project's denoiser plan for what each one
 * does and why its current default was chosen as a starting point, not a final value.
 */
export interface DenoiserTunables {
    // DenoiseOperation (denoise.wgsl) - see DenoiseUniforms in denoise_operation.ts for what each does.
    varianceScale: number;
    darknessNoiseFloor: number;
    maxBlurMip: number;
    albedoSensitivity: number;
    densitySensitivity: number;
    normalSensitivity: number;
    sigmaLuminanceTight: number;
    sigmaLuminanceLoose: number;
    kLuminance: number;
    maxSplitDistance: number;
    // BuildDenoiserQuadtreeOperation (build_denoiser_quadtree.wgsl) - see
    // BuildDenoiserQuadtreeUniforms in build_denoiser_quadtree.ts. currentGBufferMip is excluded -
    // that field is derived per-dispatch (which quadtree level is being built), not a tunable.
    albedoLuminanceThreshold: number;
    albedoChromaThreshold: number;
    logDensityThreshold: number;
    volatilityThreshold: number;
    detailThreshold: number;
    varianceGateScale: number;
}

export const DEFAULT_DENOISER_TUNABLES: DenoiserTunables = {
    varianceScale: 8.0,
    darknessNoiseFloor: 0.002,
    maxBlurMip: 5.0,
    albedoSensitivity: 0.3,
    densitySensitivity: 1.0,
    normalSensitivity: 8.0,
    // Matches filter_variance.wgsl's SIGMA_LUMINANCE_TIGHT/LOOSE/K_LUMINANCE exactly - same
    // adaptive-sigma pattern, reused rather than re-derived (see denoise.wgsl).
    sigmaLuminanceTight: 0.05,
    sigmaLuminanceLoose: 2.5,
    kLuminance: 2.0,
    // Distance-bias split cutoff (this project's denoiser plan) - see denoise.wgsl's shouldSplit()
    // doc comment for the node-relative-texels normalization this is measured in.
    maxSplitDistance: 2.0,
    albedoLuminanceThreshold: 0.1,
    albedoChromaThreshold: 0.2,
    logDensityThreshold: 0.05,
    volatilityThreshold: 0.05,
    detailThreshold: 0.1,
    varianceGateScale: 20.0,
};

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

    /**
     * Atomic accumulator the photon tracer writes into: width*height*6 u32 entries - two
     * interleaved 3-wide (R,G,B) halves per pixel (base, base+3), one per independent half of
     * this frame's ray budget - see tracePhotons and this project's denoiser plan. Splitting the
     * ray budget this way lets the two halves' disagreement serve as a variance estimate; their
     * average is the same-quality signal a single undivided buffer would have produced.
     */
    private photonBuffer: ComputedBuffer | null = null;
    /** Cached zeroed data matching photonBuffer's size, reused every frame to clear it - see run(). */
    private photonBufferClearData: Uint32Array | null = null;
    private convertToHdr: ConvertPhotonIrradianceToHdrOperation;

    // --- Denoiser pipeline (this project's denoiser plan): a hierarchical guided blur over
    // combinedIrradiance, backed by a baked min/max-range quadtree over the G-Buffer. Phase 1
    // (forceFullSplit) validated the guided blur's weight quality in isolation; Phase 2 (the
    // quadtree below) is what makes ShouldSplit() actually prune the traversal.

    /** Live-editable copy of every denoiser threshold - see DenoiserTunables' own doc comment. */
    public denoiserTunables: DenoiserTunables = { ...DEFAULT_DENOISER_TUNABLES };

    /** Per-half HDR conversion of photonBuffer, mip0 only - see ConvertPhotonIrradianceToHdrOperation. */
    private irradianceA: ComputedTexture | null = null;
    private irradianceB: ComputedTexture | null = null;
    /** mean(irradianceA, irradianceB) pre-denoise signal, full mip chain (0..lightmap.mipLevelCount-1) - kept separate from `lightmap` since denoise() reads mip0 of this while writing lightmap's mip0. */
    private combinedIrradiance: ComputedTexture | null = null;
    /** Relative variance from the (irradianceA, irradianceB) pair, quarter resolution (matches combinedIrradiance's mip2) - see ComputeVarianceAndMipsOperation. */
    private rawVariance: ComputedTexture | null = null;
    private filteredVariance: ComputedTexture | null = null;

    /** Normal-based edge detector feeding the quadtree bake, full G-Buffer resolution, mip0 only - see ComputeVolatilityOperation. */
    private volatility: ComputedTexture | null = null;
    /** Baked min/max-range quadtree textures (see BuildDenoiserQuadtreeOperation) - half G-Buffer resolution, own 0-indexed mip chain of (combinedIrradiance.mipLevelCount - 1) levels; level i answers "should G-Buffer mip (i+1) split into mip-i children" (see denoise.wgsl's shouldSplit()). */
    private albedoMin: ComputedTexture | null = null;
    private albedoMax: ComputedTexture | null = null;
    private densityMinMaxVolatility: ComputedTexture | null = null;
    private quadtreeMustSplit: ComputedTexture | null = null;

    private computeVarianceAndMips: ComputeVarianceAndMipsOperation;
    private filterVariance: FilterVarianceOperation;
    private denoise: DenoiseOperation;
    private computeVolatility: ComputeVolatilityOperation;
    /** Permanently LEVEL0=true / LEVEL0=false respectively - never toggled per-frame, see build_denoiser_quadtree.wgsl's file header (a switch change is a full pipeline recompile). */
    private buildQuadtreeLevel0: BuildDenoiserQuadtreeOperation;
    private buildQuadtreeIterate: BuildDenoiserQuadtreeOperation;
    /** Fixed to 'rgba8unorm' at construction (Albedo's format) - never switched at runtime, so mipDownsampleAlbedo/mipDownsample never fight over recompiling the same pipeline for two different formats every frame. */
    private mipDownsampleAlbedo: MipDownsampleOperation;
    /** Fixed to 'rgba16float' (its default) - used for NormalRoughness, combinedIrradiance, and the final lightmap's own post-denoise mip chain. */
    private mipDownsample: MipDownsampleOperation;
    private densityMipBlit: DensityMipBlitResources;

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

    /**
     * Unity's BILINEAR_PHOTON_DISTRIBUTION toggle (see ForwardMonteCarloOperation.updateSwitches
     * and forward_monte_carlo.wgsl's writeSample): smooth bilinear photon splat vs. a single-tap
     * nearest write that measurably reduces scattered global-atomic pressure on mobile GPUs weak
     * at that pattern - see CLAUDE.md. Reset from the active SimulationDeviceProfile on every
     * loadFromScene() call (see getSimulationDeviceProfile); still free to reassign afterward for
     * a manual override, picked up by every light-kind operation on the next tracePhotons() call.
     */
    public bilinearPhotonDistribution = true;

    constructor(device: GPUDevice, computedDataManager: ComputedDataManager) {
        this.device = device;
        this.computedDataManager = computedDataManager;
        this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
        this.vertexBuffer = getQuadVertexBuffer(device);
        this.convertToHdr = new ConvertPhotonIrradianceToHdrOperation(device);

        this.computeVarianceAndMips = new ComputeVarianceAndMipsOperation(device);
        this.filterVariance = new FilterVarianceOperation(device);
        this.denoise = new DenoiseOperation(device);
        this.computeVolatility = new ComputeVolatilityOperation(device);
        this.buildQuadtreeLevel0 = new BuildDenoiserQuadtreeOperation(device);
        this.buildQuadtreeIterate = new BuildDenoiserQuadtreeOperation(device);
        this.buildQuadtreeIterate.updateSwitches({ level0: false });
        this.mipDownsampleAlbedo = new MipDownsampleOperation(device);
        this.mipDownsampleAlbedo.updateSwitches({ outputFormat: 'rgba8unorm' });
        this.mipDownsample = new MipDownsampleOperation(device);
        this.densityMipBlit = new DensityMipBlitResources(device);

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

    // --- Denoiser evidence debug-view plumbing (see debug_view.ts's DebugView, registered by
    // LitboxSceneRenderer) - not used by the normal render path.

    public getIrradianceAView(): GPUTextureView | null {
        return this.irradianceA?.view ?? null;
    }

    public getIrradianceBView(): GPUTextureView | null {
        return this.irradianceB?.view ?? null;
    }

    /** Pre-denoise mean(A,B) signal - contrast with getLightmapView(), the post-denoise final (albedo/density-combined) image. */
    public getCombinedIrradianceView(): GPUTextureView | null {
        return this.combinedIrradiance?.view ?? null;
    }

    public getRawVarianceView(): GPUTextureView | null {
        return this.rawVariance?.view ?? null;
    }

    public getFilteredVarianceView(): GPUTextureView | null {
        return this.filteredVariance?.view ?? null;
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

    /**
     * The simulation's actual (device-profile-scaled) target resolution, or null if there's no
     * active simulation - see deriveEffectiveSimulation. RaytracedResources' G-Buffer must match
     * this exactly (not the scene's raw, unscaled SceneSimulation.width/height), since
     * forward_monte_carlo.wgsl samples both at the same target pixel coordinates.
     */
    public getEffectiveResolution(): { width: number; height: number } | null {
        return this.simulation ? { width: this.simulation.width, height: this.simulation.height } : null;
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
        for (const texture of [
            this.irradianceA, this.irradianceB, this.combinedIrradiance, this.rawVariance, this.filteredVariance,
            this.volatility, this.albedoMin, this.albedoMax, this.densityMinMaxVolatility, this.quadtreeMustSplit,
        ]) {
            if (texture) {
                this.computedDataManager.releaseTexture(texture);
            }
        }
        this.irradianceA = null;
        this.irradianceB = null;
        this.combinedIrradiance = null;
        this.rawVariance = null;
        this.filteredVariance = null;
        this.volatility = null;
        this.albedoMin = null;
        this.albedoMax = null;
        this.densityMinMaxVolatility = null;
        this.quadtreeMustSplit = null;
        const rawSimulation = scene.simulations.length > 0 ? scene.simulations[0] : null;

        if (scene.simulations.length > 1) {
            console.warn(`Litbox: ${scene.simulations.length} simulations present; only the first is rendered.`);
        }
        if (!rawSimulation) {
            this.simulation = null;
            return;
        }

        const deviceProfile = getSimulationDeviceProfile(getPlatform(), isRandomAccessFriendlyGpu());
        this.simulation = deriveEffectiveSimulation(rawSimulation, deviceProfile);
        this.bilinearPhotonDistribution = deviceProfile.bilinearPhotonDistribution;
        // Visible via the on-screen console overlay on mobile - lets a device profile be verified
        // without attaching devtools (see CDP-over-adb mobile debugging notes: some GPU readback
        // paths hang under remote debugging, so an in-page log is the more reliable check anyway).
        console.log(`Litbox: device profile - platform=${getPlatform()} gpuRandomAccessFriendly=${isRandomAccessFriendlyGpu()} `
            + `resolution=${this.simulation.width}x${this.simulation.height} raysPerFrame=${this.simulation.raysPerFrame} `
            + `maxIntegrationSteps=${computeMaxIntegrationSteps(this.simulation.width, this.simulation.height)} bilinear=${this.bilinearPhotonDistribution}`);

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
            width * height * 6 * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        );
        this.photonBufferClearData = new Uint32Array(width * height * 6);

        // Denoiser evidence-gathering textures (this project's denoiser plan) - see their field
        // doc comments above for what each holds.
        const irradianceUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        this.irradianceA = this.computedDataManager.acquireTexture(width, height, LIGHTMAP_FORMAT, irradianceUsage);
        this.irradianceB = this.computedDataManager.acquireTexture(width, height, LIGHTMAP_FORMAT, irradianceUsage);
        this.combinedIrradiance = this.computedDataManager.acquireTexture(width, height, LIGHTMAP_FORMAT, irradianceUsage, mipLevelCount);
        const mip2Width = Math.max(1, width >> 2);
        const mip2Height = Math.max(1, height >> 2);
        this.rawVariance = this.computedDataManager.acquireTexture(mip2Width, mip2Height, VARIANCE_FORMAT, irradianceUsage);
        this.filteredVariance = this.computedDataManager.acquireTexture(mip2Width, mip2Height, VARIANCE_FORMAT, irradianceUsage);

        // Baked denoiser quadtree (this project's denoiser plan, Phase 2) - see field doc comments
        // above. albedoMin/Max/densityMinMaxVolatility/quadtreeMustSplit are allocated at HALF the
        // G-Buffer's resolution with their own (mipLevelCount - 1)-level chain (own 0-indexed mip
        // space, offset by -1 from G-Buffer/irradiance mip space - see denoise.wgsl's
        // shouldSplit()); skipped entirely (guarded again in run()) if the simulation is too small
        // to have any levels for it.
        this.volatility = this.computedDataManager.acquireTexture(width, height, VARIANCE_FORMAT, irradianceUsage);
        const quadtreeMipLevelCount = mipLevelCount - 1;
        if (quadtreeMipLevelCount > 0) {
            const quadtreeWidth = Math.max(1, width >> 1);
            const quadtreeHeight = Math.max(1, height >> 1);
            this.albedoMin = this.computedDataManager.acquireTexture(quadtreeWidth, quadtreeHeight, LIGHTMAP_FORMAT, irradianceUsage, quadtreeMipLevelCount);
            this.albedoMax = this.computedDataManager.acquireTexture(quadtreeWidth, quadtreeHeight, LIGHTMAP_FORMAT, irradianceUsage, quadtreeMipLevelCount);
            this.densityMinMaxVolatility = this.computedDataManager.acquireTexture(quadtreeWidth, quadtreeHeight, LIGHTMAP_FORMAT, irradianceUsage, quadtreeMipLevelCount);
            this.quadtreeMustSplit = this.computedDataManager.acquireTexture(quadtreeWidth, quadtreeHeight, VARIANCE_FORMAT, irradianceUsage, quadtreeMipLevelCount);
        }

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
     * Drives the full per-frame simulation + denoiser evidence-gathering pipeline (this project's
     * denoiser plan): tracePhotons() dispatches the ForwardMonteCarlo tracer twice per light
     * instance (once per half of the two-way variance-estimation ray split) into the shared
     * stereo photonBuffer; ConvertPhotonIrradianceToHdrOperation converts each half into its own
     * HDR irradiance texture; ComputeVarianceAndMipsOperation fuses mean(A,B)/variance/mip
     * generation into combinedIrradiance (mip0-4) and rawVariance (quarter res); mip generation
     * then continues for the G-Buffer and combinedIrradiance's deeper mips;
     * FilterVarianceOperation bilateral-filters the variance evidence; denoise (currently a
     * passthrough stub - the real size-argument/guided-blur algorithm is a separate, later step)
     * produces the final lightmap mip0, whose own higher mips are then regenerated from it (for
     * whatever later samples the lightmap across mips, e.g. sprites).
     */
    public run(
        encoder: GPUCommandEncoder,
        raytracedResources: RaytracedResources,
        lightResources: LightResources,
        lutResources: LutResources,
        sceneGraph: SceneGraph,
    ): void {
        if (!this.lightmap || !this.irradianceA || !this.irradianceB || !this.combinedIrradiance || !this.rawVariance || !this.filteredVariance) {
            return;
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

        // Photon buffer -> two independent HDR irradiance textures. Deferred here:
        // albedo/density combination now happens later, in denoise() - see this project's
        // denoiser plan (combination must happen after variance computation and after denoising).
        this.convertToHdr.updateSwitches({ combineAlbedoDensity: false });
        this.convertToHdr.updateUniforms({ hdrScale: (width * height) / 0xFFFFFFFF });
        this.convertToHdr.updateInputs(this.photonBuffer.buffer, albedoView, densityView);
        this.convertToHdr.updateOutputs(this.irradianceA.getMipView(0), this.irradianceB.getMipView(0), width, height);
        this.convertToHdr.execute(encoder);

        // Fused mean/variance/mip pass - combinedIrradiance mip0-2 + rawVariance at mip2 (quarter
        // resolution). Capped at mip2, not the 5 levels a 16x16 tile could in principle reduce to
        // in one pass, by WebGPU's guaranteed-minimum maxStorageTexturesPerShaderStage of 4 (see
        // ComputeVarianceAndMipsOperation) - only attempt it when combinedIrradiance actually has
        // that many mip levels (tiny simulation resolutions might not; a real fix for that edge
        // case is deferred, not a concern for realistic scene sizes).
        if (this.combinedIrradiance.mipLevelCount >= 3) {
            this.computeVarianceAndMips.updateInputs(this.irradianceA.getMipView(0), this.irradianceB.getMipView(0));
            this.computeVarianceAndMips.updateOutputs(
                this.combinedIrradiance.getMipView(0),
                this.combinedIrradiance.getMipView(1),
                this.combinedIrradiance.getMipView(2),
                this.rawVariance.getMipView(0),
                width,
                height,
            );
            this.computeVarianceAndMips.execute(encoder);
        }

        // Extend combinedIrradiance past mip2 (irradianceA/B have already served their purpose -
        // the variance evidence was captured at mip2 above) and generate the G-Buffer's own mip
        // chains - all evidence the eventual guided blur will want.
        this.generateEvidenceMips(encoder, raytracedResources);

        // Bilateral-filter rawVariance using G-Buffer mip2 evidence (albedo similarity, luminance
        // closeness) - structurally matches Unity's confirmed-live FilterVariance kernel.
        const albedoMip2View = raytracedResources.getAlbedoMipView(2);
        if (albedoMip2View) {
            const mip2Width = Math.max(1, width >> 2);
            const mip2Height = Math.max(1, height >> 2);
            this.filterVariance.updateInputs(this.rawVariance.getMipView(0), albedoMip2View, this.combinedIrradiance.getMipView(2));
            this.filterVariance.updateOutputs(this.filteredVariance.getMipView(0), mip2Width, mip2Height);
            this.filterVariance.execute(encoder);
        }

        // Baked denoiser quadtree (this project's denoiser plan, Phase 2) - backs denoise.wgsl's
        // real ShouldSplit(). Must run after filterVariance above (the irradiance-detail trigger
        // gates on filteredVariance) and before denoise below. See build_denoiser_quadtree.wgsl.
        this.buildDenoiserQuadtree(encoder, raytracedResources);

        // Hierarchical guided blur (see this project's denoiser plan). forceFullSplit is off -
        // ShouldSplit() now consults the baked quadtree above instead of always splitting to mip
        // 0. Also where albedo/density finally get combined into the final lit image.
        this.denoise.updateSwitches({ combineAlbedoDensity: true, forceFullSplit: false });
        // denoiserTunables has more fields than DenoiseUniforms needs (the quadtree-bake ones) -
        // structurally fine to pass straight through, see DenoiserTunables' own doc comment.
        this.denoise.updateUniforms(this.denoiserTunables);
        this.denoise.updateInputs(
            this.combinedIrradiance.view, albedoView, normalRoughnessView, densityView, this.filteredVariance.view,
            this.quadtreeMustSplit?.view ?? this.filteredVariance.view,
        );
        this.denoise.updateOutputs(this.lightmap.getMipView(0), width, height);
        this.denoise.execute(encoder);

        // Regenerate the final lightmap's own higher mips from its just-denoised mip0 (whatever
        // later samples the lightmap across mips, e.g. sprites, needs a real chain, not a clear) -
        // independent of combinedIrradiance's own chain, since this one reflects the final lit
        // (albedo/density-combined) image, not raw irradiance.
        for (let mip = 1; mip < this.lightmap.mipLevelCount; mip++) {
            const mipWidth = Math.max(1, width >> mip);
            const mipHeight = Math.max(1, height >> mip);
            this.mipDownsample.updateInputs(this.lightmap.getMipView(mip - 1));
            this.mipDownsample.updateOutputs(this.lightmap.getMipView(mip), mipWidth, mipHeight);
            this.mipDownsample.execute(encoder);
        }
    }

    /**
     * G-Buffer mip chains (Albedo/NormalRoughness via MipDownsampleOperation, Density via its
     * dedicated render-attachment blit - see DensityMipBlitResources for why) and combinedIrradiance's
     * mip3+ (past what ComputeVarianceAndMipsOperation's fused pass produces) - all
     * structural/signal evidence the eventual guided blur will want. See this project's denoiser
     * plan.
     */
    private generateEvidenceMips(encoder: GPUCommandEncoder, raytracedResources: RaytracedResources): void {
        if (!this.combinedIrradiance || !this.simulation) {
            return;
        }
        const { width, height } = this.simulation;

        const gBufferMipLevelCount = raytracedResources.getGBufferMipLevelCount();
        for (let mip = 1; mip < gBufferMipLevelCount; mip++) {
            const mipWidth = Math.max(1, width >> mip);
            const mipHeight = Math.max(1, height >> mip);

            const albedoSource = raytracedResources.getAlbedoMipView(mip - 1);
            const albedoDest = raytracedResources.getAlbedoMipView(mip);
            if (albedoSource && albedoDest) {
                this.mipDownsampleAlbedo.updateInputs(albedoSource);
                this.mipDownsampleAlbedo.updateOutputs(albedoDest, mipWidth, mipHeight);
                this.mipDownsampleAlbedo.execute(encoder);
            }

            const normalRoughnessSource = raytracedResources.getNormalRoughnessMipView(mip - 1);
            const normalRoughnessDest = raytracedResources.getNormalRoughnessMipView(mip);
            if (normalRoughnessSource && normalRoughnessDest) {
                this.mipDownsample.updateInputs(normalRoughnessSource);
                this.mipDownsample.updateOutputs(normalRoughnessDest, mipWidth, mipHeight);
                this.mipDownsample.execute(encoder);
            }

            const densitySource = raytracedResources.getDensityMipView(mip - 1);
            const densityDest = raytracedResources.getDensityMipView(mip);
            if (densitySource && densityDest) {
                this.densityMipBlit.updateInputs(densitySource);
                this.densityMipBlit.execute(encoder, densityDest);
            }
        }

        for (let mip = 3; mip < this.combinedIrradiance.mipLevelCount; mip++) {
            const mipWidth = Math.max(1, width >> mip);
            const mipHeight = Math.max(1, height >> mip);
            this.mipDownsample.updateInputs(this.combinedIrradiance.getMipView(mip - 1));
            this.mipDownsample.updateOutputs(this.combinedIrradiance.getMipView(mip), mipWidth, mipHeight);
            this.mipDownsample.execute(encoder);
        }
    }

    /**
     * Builds the baked min/max-range quadtree (this project's denoiser plan, Phase 2) that backs
     * denoise.wgsl's ShouldSplit() - see build_denoiser_quadtree.wgsl for the algorithm.
     * computeVolatility runs once (normalRoughness mip0 -> volatility, the only point this touches
     * the normal texture); buildQuadtreeLevel0 then buildQuadtreeIterate (looped) build the
     * quadtree's own mip chain level by level, each level's output feeding the next. Two
     * permanently-switched operation instances (not one instance toggled per frame) - see
     * BuildDenoiserQuadtreeOperation's doc comment for why.
     */
    private buildDenoiserQuadtree(encoder: GPUCommandEncoder, raytracedResources: RaytracedResources): void {
        if (!this.combinedIrradiance || !this.filteredVariance || !this.volatility
            || !this.albedoMin || !this.albedoMax || !this.densityMinMaxVolatility || !this.quadtreeMustSplit
            || !this.simulation) {
            return;
        }
        const { width, height } = this.simulation;
        const albedoView = raytracedResources.getAlbedoView();
        const densityView = raytracedResources.getDensityView();
        const normalRoughnessView = raytracedResources.getNormalRoughnessView();
        if (!albedoView || !densityView || !normalRoughnessView) {
            return;
        }

        this.computeVolatility.updateInputs(normalRoughnessView);
        this.computeVolatility.updateOutputs(this.volatility.getMipView(0), width, height);
        this.computeVolatility.execute(encoder);

        // Thresholds come from denoiserTunables (TBD/tunable - see this project's denoiser plan),
        // shared across every level of the chain (both the LEVEL0 and iterative passes apply the
        // same comparison, per the Unity reference this was ported from) - only currentGBufferMip
        // changes per level, so it's overridden fresh on each call below.
        const level0Width = Math.max(1, width >> 1);
        const level0Height = Math.max(1, height >> 1);
        this.buildQuadtreeLevel0.updateUniforms({ ...this.denoiserTunables, currentGBufferMip: 1 });
        this.buildQuadtreeLevel0.updateInputsLevel0(
            albedoView, densityView, this.volatility.getMipView(0), this.combinedIrradiance.view, this.filteredVariance.view,
        );
        this.buildQuadtreeLevel0.updateOutputs(
            this.albedoMin.getMipView(0), this.albedoMax.getMipView(0), this.densityMinMaxVolatility.getMipView(0), this.quadtreeMustSplit.getMipView(0),
            level0Width, level0Height,
        );
        this.buildQuadtreeLevel0.execute(encoder);

        for (let level = 1; level < this.quadtreeMustSplit.mipLevelCount; level++) {
            const levelWidth = Math.max(1, width >> (level + 1));
            const levelHeight = Math.max(1, height >> (level + 1));
            this.buildQuadtreeIterate.updateUniforms({ ...this.denoiserTunables, currentGBufferMip: level + 1 });
            this.buildQuadtreeIterate.updateInputsIterate(
                this.albedoMin.getMipView(level - 1), this.albedoMax.getMipView(level - 1),
                this.densityMinMaxVolatility.getMipView(level - 1), this.quadtreeMustSplit.getMipView(level - 1),
                this.combinedIrradiance.view, this.filteredVariance.view,
            );
            this.buildQuadtreeIterate.updateOutputs(
                this.albedoMin.getMipView(level), this.albedoMax.getMipView(level), this.densityMinMaxVolatility.getMipView(level), this.quadtreeMustSplit.getMipView(level),
                levelWidth, levelHeight,
            );
            this.buildQuadtreeIterate.execute(encoder);
        }
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
        const maxIntegrationSteps = computeMaxIntegrationSteps(width, height);

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
            // color.r/g/b converted sRGB->linear (see color_space.ts) to match Unity's
            // RTLightSource.Energy, which now reads `.color.linear * intensity * intensity` -
            // this is that same Energy computation, just with the intensity² already folded into
            // `intensity` upstream.
            const energyRgb: [number, number, number] = [srgbToLinear(color.r) * intensity, srgbToLinear(color.g) * intensity, srgbToLinear(color.b) * intensity];
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
            const bounces = resolveBounces(photonBounces, light.bounces);

            operation.updateSwitches({
                bilinearPhotonDistribution: this.bilinearPhotonDistribution,
                maxIntegrationSteps,
            });
            operation.updateInputs(seedBuffer.buffer, albedoView, densityView, normalRoughnessView, lutResources);
            operation.updateOutputs(this.photonBuffer.buffer, this.writeCounterBuffer);

            // Split this light's ray budget into two independent halves - disjoint seedBase
            // sub-ranges within its own seedBases[i]..seedBases[i]+rays[i] slice, each writing
            // into photonBuffer's own half (see its doc comment and this project's denoiser
            // plan). Split at workgroup (64-ray) granularity, not just in two arbitrarily: rays[i]
            // is always a multiple of 64 (computeRayCount), and ForwardMonteCarloOperation's
            // dispatch extent rounds *up* to whole workgroups - an unaligned split would let one
            // half's "overflow" threads run past its intended ray count and collide with the
            // other half's seed sub-range and photon writes. A light with the minimum single
            // workgroup (rays[i] === 64) can't be split at all this way - all 64 rays go to half
            // 0 and half 1 gets none for that light this frame, a rare, low-stakes edge case.
            const workgroupCount = rays[i] / 64;
            const workgroupCountA = Math.floor(workgroupCount / 2);
            const halves: { rays: number; seedBase: number; halfIndex: number }[] = [
                { rays: workgroupCountA * 64, seedBase: seedBases[i], halfIndex: 0 },
                { rays: (workgroupCount - workgroupCountA) * 64, seedBase: seedBases[i] + workgroupCountA * 64, halfIndex: 1 },
            ];
            for (const half of halves) {
                if (half.rays === 0) {
                    continue;
                }
                // Normalized from this half's own ray count, not rays[i] - so each half
                // independently is an unbiased estimator of the same signal on its own, matching
                // Unity's TracerPostProcessing.compute's mean(sample_a, sample_b) equivalence.
                const photonEnergyScale = 0xFFFFFFFF / half.rays / integrationInterval;
                const lightEnergy: [number, number, number] = [
                    energyRgb[0] * photonEnergyScale,
                    energyRgb[1] * photonEnergyScale,
                    energyRgb[2] * photonEnergyScale,
                ];
                operation.updateUniforms({
                    lightToTarget,
                    lightEnergy,
                    bounces,
                    seedBase: half.seedBase,
                    halfIndex: half.halfIndex,
                    directionalLightDirection,
                    lightPinch,
                    integrationInterval,
                    integrationIntervalSquared,
                    rays: half.rays,
                });
                operation.execute(encoder);
            }
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
