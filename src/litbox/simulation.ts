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
import { LightmapBlurCascade } from './lightmap_blur_cascade.ts';
import { DensityMipBlitResources } from './density_mip_blit.ts';
import { ComputeVarianceAndMipsOperation } from './compute_variance_and_mips.ts';
import { FilterVarianceOperation } from './filter_variance.ts';
import { DenoiseOperation } from './denoise_operation.ts';
import { DitherFilterOperation } from './dither_filter.ts';
import { ComputeVolatilityOperation } from './compute_volatility.ts';
import { BuildDenoiserQuadtreeOperation } from './build_denoiser_quadtree.ts';
import {
    ForwardMonteCarloOperation,
    luminance,
    computeRayCount,
    resolveBounces,
    computeWorldToTargetPixels,
    computeLightToTarget,
    computeDirectionalLightSegment,
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
 *
 * maxBlurMip, unlike the resolutionScale/raysPerFrameScale/bilinear split above, trims off
 * Desktop's default (DEFAULT_DENOISER_TUNABLES.maxBlurMip) by a flat amount rather than scaling -
 * but *how much* it trims does still depend on gpuRandomAccessFriendly, for two additive reasons
 * that apply independently:
 * - Both mobile profiles get a free -1 from resolutionScale's halved resolution alone (fewer real
 *   mip levels exist in the texture chain at half res, so the equivalent-quality starting mip is
 *   one level lower without cutting anything - this is a parity correction, not a quality cut).
 * - GPU NOT random-access-friendly (PowerVR) gets a further -1 on top of that: denoise.wgsl's
 *   traversal cost is dominated by how deep it's allowed to descend per seed, and a deeper descent
 *   means more divergent per-pixel stack work on PowerVR's SIMT model specifically (see CLAUDE.md's
 *   PowerVR-divergent-SIMT-loop finding) - confirmed NOT to be a problem on Apple GPUs (measured:
 *   the denoiser "runs amazingly well" on iPhone, see this project's mobile-perf-tuning notes), so
 *   random-access-friendly mobile only pays the resolution-parity -1, not this extra cut.
 */
export interface SimulationDeviceProfile {
    resolutionScale: number;
    raysPerFrameScale: number;
    bilinearPhotonDistribution: boolean;
    maxBlurMip: number;
}

export function getSimulationDeviceProfile(platform: Platform, gpuRandomAccessFriendly: boolean): SimulationDeviceProfile {
    if (platform === 'desktop') {
        return { resolutionScale: 1, raysPerFrameScale: 1, bilinearPhotonDistribution: true, maxBlurMip: DEFAULT_DENOISER_TUNABLES.maxBlurMip };
    }
    return gpuRandomAccessFriendly
        ? { resolutionScale: 0.5, raysPerFrameScale: 0.5, bilinearPhotonDistribution: true, maxBlurMip: DEFAULT_DENOISER_TUNABLES.maxBlurMip - 1 }
        : { resolutionScale: 0.5, raysPerFrameScale: 0.25, bilinearPhotonDistribution: false, maxBlurMip: DEFAULT_DENOISER_TUNABLES.maxBlurMip - 2 };
}

/**
 * Blur levels to subtract from every sprite's authored simBlur (see sprite.wgsl's SpriteProperties
 * and LightmapBlurCascade) before it's used to select the lightmap's blur level. simBlur is
 * authored as a level index assuming the lightmap's mip 0 sits at the scene's natural, unscaled
 * resolution - but LightmapBlurCascade's levels are built on top of the device-profile-scaled
 * simulation resolution (see deriveEffectiveSimulation), so a halved resolutionScale means each of
 * *its* levels already covers twice the world-space extent the same-numbered level would at full
 * resolution. Left uncorrected, a scene's authored blur sizes (tuned against desktop's unscaled
 * lightmap) would render roughly 1/resolutionScale times too blurry on a scaled-down device.
 * resolutionScale is always a power of two across every current/future device profile, so this is
 * always a whole number - no rounding needed before it's subtracted from simBlur (see
 * SpriteResources' use of this).
 */
export function computeSpriteBlurMipOffset(resolutionScale: number): number {
    return -Math.log2(resolutionScale);
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
    densityBlurFalloff: number;
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
    // LightmapBlurCascade (lightmap_blur_cascade.ts) - see this project's plan: "Fix lightmap
    // sprite-blur pixelation on high-contrast content" for why these exist and what each
    // calibrates. Grouped with the denoiser tunables (not SimulationTunables) since this is
    // squarely a lightmap post-process concern, same category as maxBlurMip above.
    /** How many full-resolution, never-decimated blur levels sprite.wgsl can select before falling back to the decimated mip chain - a structural tunable: changing it reallocates lightmapCascade (see SimulationResources.ensureBlurCascadeAllocated). */
    blurCascadeLevelCount: number;
    /** Target sigma (in texels) of the first cascade level; each subsequent level's target cumulative sigma doubles the previous one's - see LightmapBlurCascade's doc comment. Cheap to change live, no reallocation. */
    blurCascadeSigma: number;
}

export const DEFAULT_DENOISER_TUNABLES: DenoiserTunables = {
    // Re-tuned (live, via the tunables panel) after two fixes landed together: the
    // ForwardMonteCarloOperation shared-uniform-buffer clobber (forward_monte_carlo.ts's
    // uniformBufferPool/beginFrame - centerVariance used to read saturated almost everywhere from
    // that bug alone, not real scene noise) and the quadtree's irradiance-detail signal switching
    // from luminance(mean(A,B)) to min(lumaA, lumaB) (build_denoiser_quadtree.wgsl - stops a
    // one-sided firefly from reading as confirmed "real detail"). Both fixes change what
    // centerVariance/hasNearbyDetail actually mean, so the old defaults (tuned against the
    // pre-fix, corrupted signals) no longer apply - these values reflect the post-fix scene.
    varianceScale: 10.0,
    darknessNoiseFloor: 0.1,
    maxBlurMip: 5.0,
    // Starting point, not a solved value (see this struct's own doc comment). Measured in
    // log2(1+opticalDepth), not raw optical depth (see denoise.wgsl's decideBlurSize doc comment
    // for why) - 2.0 reaches full suppression around opticalDepth 3 (about 95% dense), the same
    // ceiling a raw-optical-depth falloff of 3.0 would have had, but ramps up far more gradually
    // below that instead of the whole transition being compressed into the last few % of density.
    densityBlurFalloff: 0.001,
    albedoSensitivity: 0.3,
    densitySensitivity: 1.0,
    normalSensitivity: 8.0,
    // Matches filter_variance.wgsl's SIGMA_LUMINANCE_TIGHT/LOOSE/K_LUMINANCE exactly - same
    // adaptive-sigma pattern, reused rather than re-derived (see denoise.wgsl). Both sigmas are
    // compared against a log-luminance difference (see denoise.wgsl's decideWeight), not raw linear
    // luminance. sigmaLuminanceTight no longer needs to be razor-thin now that the quadtree's
    // min(A,B) detail signal (see the comment above) rejects one-sided fireflies before
    // hasNearbyDetail ever forces decideWeight onto this branch - false positives it used to have to
    // guard against on its own are structurally gone, so it can sit much looser without fireflies
    // creeping back in.
    sigmaLuminanceTight: 1.0,
    sigmaLuminanceLoose: 2.5,
    kLuminance: 2.0,
    // Distance-bias split cutoff (this project's denoiser plan) - see denoise.wgsl's shouldSplit()
    // doc comment for the seed-relative-texels normalization this is measured in.
    maxSplitDistance: 1.5,
    albedoLuminanceThreshold: 0.1,
    albedoChromaThreshold: 0.2,
    logDensityThreshold: 0.05,
    volatilityThreshold: 0.05,
    detailThreshold: 0.1,
    varianceGateScale: 20.0,
    // Starting points, not solved values (see LightmapBlurCascade's doc comment for the
    // doubling-target formula these feed). blurCascadeLevelCount default of 2 matches this
    // project's original educated guess at how many full-res levels a laser beam needs to survive
    // before decimation is safe - tunable specifically because that guess wasn't derived.
    blurCascadeLevelCount: 2,
    blurCascadeSigma: 3.0,
};

/**
 * Every tunable value in the photon-tracing simulation itself (as opposed to DenoiserTunables,
 * which tunes the post-process blur) - bundled the same way for the same reason: a single source
 * of truth for the portfolio page's config UI (see main.ts's simulation tunables panel). Mutate a
 * field directly (e.g. `simulationResources.simulationTunables.surfaceBias = 4`); tracePhotons
 * reads this fresh every frame, so there's no separate "apply" step.
 *
 * raysPerFrame/integrationInterval/photonBounces mirror the identically-named SceneSimulation
 * fields (the scene's exported JSON) - they start out equal to that scene's own (already
 * device-profile-scaled) values every time loadFromScene runs (see its doc comment), then become
 * independently live-editable from that point on, same as every other field here. Simulation
 * resolution (width/height) is deliberately NOT part of this struct - resizing needs a full GPU
 * resource reallocation (LitboxSceneRenderer.resizeSimulation), not a per-frame uniform write, so
 * it doesn't fit this "mutate and reread" shape.
 */
export interface SimulationTunables {
    raysPerFrame: number;
    integrationInterval: number;
    /** -1 = use each light's own bounce count (SceneSimulation's OverrideBounceCount sentinel - see resolveBounces). */
    photonBounces: number;
    /** forward_monte_carlo.wgsl's scatterMaterially self-intersection pushback after a bounce - fine-geometry light leaks vs. self-shadowing artifacts. */
    surfaceBias: number;
    /** forward_monte_carlo.wgsl's ambient emitLight direction-perturbation divisor - how directional vs. uniform ambient scatter looks. */
    ambientScatterSoftness: number;
}

/**
 * raysPerFrame/integrationInterval/photonBounces here are placeholder fallbacks only - loadFromScene
 * always immediately overwrites them from the loaded scene's own (device-profile-scaled)
 * SceneSimulation, so no real scene ends up relying on these three. surfaceBias/
 * ambientScatterSoftness are the actual defaults (the Unity-ported hardcoded literals this
 * project inherited - starting points, not solved values, same as DEFAULT_DENOISER_TUNABLES).
 */
export const DEFAULT_SIMULATION_TUNABLES: SimulationTunables = {
    raysPerFrame: 100000,
    integrationInterval: 0.01,
    photonBounces: -1,
    surfaceBias: 2.5,
    ambientScatterSoftness: 1.44,
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
 * progressively converged across frames (see tracePhotons). Higher blur levels are regenerated
 * every frame from mip 0 by LightmapBlurCascade (a real, full-resolution Gaussian cascade feeding
 * a decimated tail, not a plain mip chain) - see lightmap_blur_cascade.ts - since sprite.wgsl's
 * blur-size selection is their only consumer.
 */
export class SimulationResources {
    private device: GPUDevice;
    private computedDataManager: ComputedDataManager;
    private lightmap: ComputedTexture | null = null;
    /** Full-resolution HDR-safe blur cascade (see lightmap_blur_cascade.ts) - entries [0, length-2] are sprite-selectable blur buckets, the last entry is an internal bridging level never exposed to sprites. Reallocated lazily whenever denoiserTunables.blurCascadeLevelCount changes - see ensureBlurCascadeAllocated. */
    private lightmapCascade: ComputedTexture[] = [];
    /** The blurCascadeLevelCount lightmapCascade is currently sized for (-1 = not yet allocated) - compared against denoiserTunables.blurCascadeLevelCount each frame to detect a tunable change. */
    private allocatedBlurCascadeLevelCount = -1;
    /** Full-resolution scratch texture shared by every LightmapBlurCascade pass (horizontal-pass intermediate) - fixed size from scene load, independent of blurCascadeLevelCount, so unlike lightmapCascade it never needs lazy reallocation. */
    private lightmapCascadeScratch: ComputedTexture | null = null;
    /** Real, halving-resolution decimated mip chain continuing past the blur cascade (see lightmap_blur_cascade.ts) - own mip0 is half of lightmap's resolution; fixed size/level count from scene load, independent of blurCascadeLevelCount. */
    private lightmapMipChain: ComputedTexture | null = null;
    private sampler: GPUSampler;

    private pipeline: GPURenderPipeline | null = null;
    private vertexBuffer: GPUBuffer;
    private compositeUniformBuffer: GPUBuffer | null = null;
    private compositeBindGroup: GPUBindGroup | null = null;
    private compositeBindGroupLayout: GPUBindGroupLayout | null = null;

    private simulation: SceneSimulation | null = null;
    private worldTransform: mat4 = mat4.create();
    /** This session's SimulationDeviceProfile.resolutionScale, set by loadFromScene - see getSpriteBlurMipOffset. */
    private resolutionScale = 1;

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

    /** Live-editable copy of every simulation tunable - see SimulationTunables' own doc comment. */
    public simulationTunables: SimulationTunables = { ...DEFAULT_SIMULATION_TUNABLES };

    public denoiserEnabled = true;
    private frameIndex = 0;

    /** Per-half HDR conversion of photonBuffer, mip0 only - see ConvertPhotonIrradianceToHdrOperation. */
    private irradianceA: ComputedTexture | null = null;
    private irradianceB: ComputedTexture | null = null;
    /** mean(irradianceA, irradianceB) pre-denoise signal, full mip chain (loadFromScene's own mipLevelCount, independent of lightmap's now-single-mip allocation) - kept separate from `lightmap` since denoise() reads mip0 of this while writing lightmap's mip0. */
    private combinedIrradiance: ComputedTexture | null = null;
    /** Relative variance from the (irradianceA, irradianceB) pair, quarter resolution (matches combinedIrradiance's mip2) - see ComputeVarianceAndMipsOperation. */
    private rawVariance: ComputedTexture | null = null;
    private filteredVariance: ComputedTexture | null = null;

    /** DenoiseOperation's output when DitherFilterOperation is about to run over it (see run()'s use of this) - the not-yet-final, jittery denoised image, full G-Buffer resolution, single mip. DitherFilterOperation reads this and writes lightmap's mip0 directly, so no copy is ever needed to get the real final image into lightmap. */
    private denoiseScratch: ComputedTexture | null = null;
    /** DenoiseOperation's per-pixel blurSize (see denoise.wgsl's decideBlurSize/blurSizeOutput), full G-Buffer resolution, single mip - diagnostic-only, feeds the 'blur-size' debug view, never consumed by anything downstream in the normal render path. */
    private blurSizeDebug: ComputedTexture | null = null;

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
    private ditherFilter: DitherFilterOperation;
    private computeVolatility: ComputeVolatilityOperation;
    /** Permanently LEVEL0=true / LEVEL0=false respectively - never toggled per-frame, see build_denoiser_quadtree.wgsl's file header (a switch change is a full pipeline recompile). */
    private buildQuadtreeLevel0: BuildDenoiserQuadtreeOperation;
    private buildQuadtreeIterate: BuildDenoiserQuadtreeOperation;
    /** Fixed to 'rgba8unorm' at construction (Albedo's format) - never switched at runtime, so mipDownsampleAlbedo/mipDownsample never fight over recompiling the same pipeline for two different formats every frame. */
    private mipDownsampleAlbedo: MipDownsampleOperation;
    /** Fixed to 'rgba16float' (its default) - used for NormalRoughness and combinedIrradiance's own post-mip2 chain (denoiser evidence, not sprite-facing). */
    private mipDownsample: MipDownsampleOperation;
    /** Regenerates the lightmap's blur levels for sprite.wgsl's blur-size selection - see lightmap_blur_cascade.ts for why this replaced LightmapBlurPyramid (still present, untouched, and still used nowhere - kept for near-LDR scenes where its cheap decimation pyramid works well, per this project's plan). */
    private lightmapBlurCascade: LightmapBlurCascade;
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
        // mipmapFilter is 'nearest', not 'linear': sprite.wgsl's textureSampleLevel(...,
        // props.simBlur) selects a discrete blur-size level, not a continuous LOD - 'nearest'
        // makes "no blending between adjacent blur sizes" a structural guarantee instead of an
        // authoring convention (relying on simBlur always being a whole number). Shared with
        // compositeInto below, which always samples level 0.0 exactly - no fractional level, so
        // this is a no-op there.
        this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'nearest' });
        this.vertexBuffer = getQuadVertexBuffer(device);
        this.convertToHdr = new ConvertPhotonIrradianceToHdrOperation(device);

        this.computeVarianceAndMips = new ComputeVarianceAndMipsOperation(device);
        this.filterVariance = new FilterVarianceOperation(device);
        this.denoise = new DenoiseOperation(device);
        this.ditherFilter = new DitherFilterOperation(device);
        this.computeVolatility = new ComputeVolatilityOperation(device);
        this.buildQuadtreeLevel0 = new BuildDenoiserQuadtreeOperation(device);
        this.buildQuadtreeIterate = new BuildDenoiserQuadtreeOperation(device);
        this.buildQuadtreeIterate.updateSwitches({ level0: false });
        this.mipDownsampleAlbedo = new MipDownsampleOperation(device);
        this.mipDownsampleAlbedo.updateSwitches({ outputFormat: 'rgba8unorm' });
        this.mipDownsample = new MipDownsampleOperation(device);
        this.lightmapBlurCascade = new LightmapBlurCascade(device);
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

    /** lightmapCascade's exposed (non-bridging) entries' views, for sprite_resources.ts's bind group - see lightmap_blur_cascade.ts. Length is getBlurCascadeLevelCount(). */
    public getLightmapCascadeViews(): GPUTextureView[] {
        return this.lightmapCascade.slice(0, -1).map((texture) => texture.view);
    }

    public getLightmapMipChainView(): GPUTextureView | null {
        return this.lightmapMipChain?.view ?? null;
    }

    /** How many of lightmapCascade's entries are currently allocated as sprite-selectable buckets (excludes the internal bridging level) - see ensureBlurCascadeAllocated. SpriteResources compares this against its own last-seen value each frame to know when to rebuild its lightmap bind group and reupload sprite properties. */
    public getBlurCascadeLevelCount(): number {
        return this.allocatedBlurCascadeLevelCount;
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

    /** Diagnostic-only (see blurSizeDebug's own doc comment) - feeds the 'blur-size' debug view. */
    public getBlurSizeDebugView(): GPUTextureView | null {
        return this.blurSizeDebug?.view ?? null;
    }

    public getSampler(): GPUSampler {
        return this.sampler;
    }

    /** See computeSpriteBlurMipOffset - SpriteResources subtracts this from every sprite's authored simBlur before upload. */
    public getSpriteBlurMipOffset(): number {
        return computeSpriteBlurMipOffset(this.resolutionScale);
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
     *
     * `sceneTunables` (the just-loaded scene's own LitboxScene.getDenoiserTunables()) resets
     * denoiserTunables to DEFAULT_DENOISER_TUNABLES merged with that override - any tunables-panel
     * tweak made against the previous scene is discarded, not carried over (see LitboxScene's own
     * doc comment for why). The device profile's maxBlurMip cut below is then re-applied on top,
     * same as before this parameter existed - that one stays device-derived, never scene-derived.
     *
     * `sceneSimulationTunables` (LitboxScene.getSimulationTunables()) gets the same treatment for
     * simulationTunables - see SimulationTunables' own doc comment for why raysPerFrame/
     * integrationInterval/photonBounces are seeded from this scene's own (already device-scaled)
     * SceneSimulation first, with sceneSimulationTunables layered on top of that.
     */
    public loadFromScene(
        scene: Scene,
        sceneGraph: SceneGraph,
        sceneTunables: Partial<DenoiserTunables> | null = null,
        sceneSimulationTunables: Partial<SimulationTunables> | null = null,
    ): void {
        if (this.lightmap) {
            this.computedDataManager.releaseTexture(this.lightmap);
            this.lightmap = null;
        }
        for (const texture of this.lightmapCascade) {
            this.computedDataManager.releaseTexture(texture);
        }
        this.lightmapCascade = [];
        this.allocatedBlurCascadeLevelCount = -1;
        if (this.lightmapCascadeScratch) {
            this.computedDataManager.releaseTexture(this.lightmapCascadeScratch);
            this.lightmapCascadeScratch = null;
        }
        if (this.lightmapMipChain) {
            this.computedDataManager.releaseTexture(this.lightmapMipChain);
            this.lightmapMipChain = null;
        }
        if (this.photonBuffer) {
            this.computedDataManager.releaseBuffer(this.photonBuffer);
            this.photonBuffer = null;
        }
        this.photonBufferClearData = null;
        for (const texture of [
            this.irradianceA, this.irradianceB, this.combinedIrradiance, this.rawVariance, this.filteredVariance,
            this.volatility, this.albedoMin, this.albedoMax, this.densityMinMaxVolatility, this.quadtreeMustSplit,
            this.denoiseScratch, this.blurSizeDebug,
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
        this.denoiseScratch = null;
        this.blurSizeDebug = null;
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
        this.resolutionScale = deviceProfile.resolutionScale;
        this.denoiserTunables = { ...DEFAULT_DENOISER_TUNABLES, ...sceneTunables };
        this.denoiserTunables.maxBlurMip = deviceProfile.maxBlurMip;
        this.simulationTunables = {
            ...DEFAULT_SIMULATION_TUNABLES,
            raysPerFrame: this.simulation.raysPerFrame,
            integrationInterval: this.simulation.integrationInterval,
            photonBounces: this.simulation.photonBounces,
            ...sceneSimulationTunables,
        };
        // Visible via the on-screen console overlay on mobile - lets a device profile be verified
        // without attaching devtools (see CDP-over-adb mobile debugging notes: some GPU readback
        // paths hang under remote debugging, so an in-page log is the more reliable check anyway).
        console.log(`Litbox: device profile - platform=${getPlatform()} gpuRandomAccessFriendly=${isRandomAccessFriendlyGpu()} `
            + `resolution=${this.simulation.width}x${this.simulation.height} raysPerFrame=${this.simulation.raysPerFrame} `
            + `maxIntegrationSteps=${computeMaxIntegrationSteps(this.simulation.width, this.simulation.height)} bilinear=${this.bilinearPhotonDistribution} `
            + `maxBlurMip=${this.denoiserTunables.maxBlurMip}`);

        this.worldTransform = sceneGraph.getWorldTransform(this.simulation.ownerId);

        const { width, height } = this.simulation;
        const mipLevelCount = Math.floor(Math.log2(Math.max(width, height))) + 1;
        // Single mip now - LightmapBlurCascade's exposed/bridging levels live in lightmapCascade
        // (full resolution, never decimated) and lightmapMipChain (the decimated tail) instead of
        // this texture's own mip1+, unlike the old LightmapBlurPyramid design - see
        // lightmap_blur_cascade.ts.
        this.lightmap = this.computedDataManager.acquireTexture(
            width,
            height,
            LIGHTMAP_FORMAT,
            GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
            1,
        );
        const cascadeUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        this.lightmapCascadeScratch = this.computedDataManager.acquireTexture(width, height, LIGHTMAP_FORMAT, cascadeUsage);
        // Own mip0 is freshly half of lightmap's resolution (the first real decimation, one step
        // past the blur cascade's last/bridging level) - NOT a reuse of any absolute mip-index
        // sizing from the old single-texture pyramid. See lightmap_blur_cascade.ts's "Calibration"
        // doc comment for why that distinction matters.
        const mipChainWidth = Math.max(1, width >> 1);
        const mipChainHeight = Math.max(1, height >> 1);
        const mipChainLevelCount = Math.max(1, mipLevelCount - 1);
        this.lightmapMipChain = this.computedDataManager.acquireTexture(mipChainWidth, mipChainHeight, LIGHTMAP_FORMAT, cascadeUsage, mipChainLevelCount);

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
        // DenoiseOperation's output when DitherFilterOperation is about to run over it (see run()) -
        // STORAGE_BINDING for denoise's write, TEXTURE_BINDING for the filter's textureLoad read.
        // No COPY_SRC/COPY_DST - the filter writes its result straight to lightmap's mip0 (its real
        // final destination), so this texture is never the source or destination of a GPU copy.
        this.denoiseScratch = this.computedDataManager.acquireTexture(
            width, height, LIGHTMAP_FORMAT, GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING);
        // Diagnostic-only (see this field's own doc comment) - same STORAGE_BINDING|TEXTURE_BINDING
        // usage/no-copy rationale as denoiseScratch above, just single-channel.
        this.blurSizeDebug = this.computedDataManager.acquireTexture(
            width, height, VARIANCE_FORMAT, GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING);

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
     * produces the final lightmap mip0, from which LightmapBlurCascade regenerates the rest of the
     * blur levels sprite.wgsl's blur-size selection samples - the only consumer of lightmapCascade/
     * lightmapMipChain.
     */
    public run(
        encoder: GPUCommandEncoder,
        raytracedResources: RaytracedResources,
        lightResources: LightResources,
        lutResources: LutResources,
        sceneGraph: SceneGraph,
    ): void {
        if (!this.lightmap || !this.irradianceA || !this.irradianceB || !this.combinedIrradiance || !this.rawVariance || !this.filteredVariance || !this.denoiseScratch || !this.blurSizeDebug || !this.lightmapCascadeScratch || !this.lightmapMipChain) {
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

        // Fused mean/variance/mip pass. Capped at mip2, not the 5 levels a 16x16 tile could in
        // principle reduce to in one pass, by WebGPU's guaranteed-minimum
        // maxStorageTexturesPerShaderStage of 4 (see ComputeVarianceAndMipsOperation) - only
        // attempt it when there are actually that many mip levels to write (tiny simulation
        // resolutions might not; a real fix for that edge case is deferred, not a concern for
        // realistic scene sizes - when the denoiser is also disabled, this means lightmap's mip0
        // goes unwritten this frame too, same tolerance).
        //
        // combineAlbedoDensity switches combinedMip0/1/2's target AND meaning when the denoiser is
        // fully off (see ComputeVarianceAndMipsSwitches' own doc comment): irradiance itself is
        // never the wanted output in that mode, so this dispatch writes directly into lightmap's
        // mip0 plus lightmapMipChain's first two mips, each holding mean * albedo * density (with
        // albedo/density box-filtered down through the same shared-memory reduction the mean
        // itself uses) rather than combinedIrradiance's raw irradiance mips - no separate dispatch
        // (DenoiseOperation's combine-only fast path) or later blur-cascade work is spent on what
        // this pass already produces for free. lightmapMipChain's mip0/mip1 are exactly half/
        // quarter of lightmap's resolution, matching this fused pass's own mip1/mip2 sizing, so the
        // redirect needs no adjustment - see this project's plan: denoiser-disabled is a debug/perf
        // toggle, not the path the blur-cascade fix targets, so it deliberately keeps this
        // box-filtered fast path instead of also running LightmapBlurCascade.
        this.computeVarianceAndMips.updateSwitches({ combineAlbedoDensity: !this.denoiserEnabled });
        const combinedMip0 = this.denoiserEnabled ? this.combinedIrradiance.getMipView(0) : this.lightmap.getMipView(0);
        const combinedMip1 = this.denoiserEnabled ? this.combinedIrradiance.getMipView(1) : this.lightmapMipChain.getMipView(0);
        const combinedMip2 = this.denoiserEnabled ? this.combinedIrradiance.getMipView(2) : this.lightmapMipChain.getMipView(1);
        if (this.combinedIrradiance.mipLevelCount >= 3) {
            this.computeVarianceAndMips.updateInputs(
                this.irradianceA.getMipView(0), this.irradianceB.getMipView(0),
                this.denoiserEnabled ? undefined : albedoView,
                this.denoiserEnabled ? undefined : densityView,
            );
            this.computeVarianceAndMips.updateOutputs(
                combinedMip0,
                combinedMip1,
                combinedMip2,
                width,
                height,
                this.denoiserEnabled ? this.rawVariance.getMipView(0) : undefined,
            );
            this.computeVarianceAndMips.execute(encoder);
        }

        if (this.denoiserEnabled) {
            // Extend combinedIrradiance past mip2 (irradianceA/B have already served their purpose
            // - the variance evidence was captured at mip2 above) and generate the G-Buffer's own
            // mip chains - all evidence the eventual guided blur will want.
            this.generateEvidenceMips(encoder, raytracedResources);

            // Bilateral-filter rawVariance using G-Buffer mip2 evidence (albedo similarity,
            // luminance closeness) - structurally matches Unity's confirmed-live FilterVariance
            // kernel.
            const albedoMip2View = raytracedResources.getAlbedoMipView(2);
            if (albedoMip2View) {
                const mip2Width = Math.max(1, width >> 2);
                const mip2Height = Math.max(1, height >> 2);
                this.filterVariance.updateInputs(this.rawVariance.getMipView(0), albedoMip2View, this.combinedIrradiance.getMipView(2));
                this.filterVariance.updateOutputs(this.filteredVariance.getMipView(0), mip2Width, mip2Height);
                this.filterVariance.execute(encoder);
            }

            // Baked denoiser quadtree (this project's denoiser plan, Phase 2) - backs
            // denoise.wgsl's real ShouldSplit(). Must run after filterVariance above (the
            // irradiance-detail trigger gates on filteredVariance) and before denoise below. See
            // build_denoiser_quadtree.wgsl.
            this.buildDenoiserQuadtree(encoder, raytracedResources);

            // Hierarchical guided blur (see this project's denoiser plan). forceFullSplit is off -
            // ShouldSplit() now consults the baked quadtree above instead of always splitting to
            // mip 0. Writes denoiseScratch, not lightmap directly - the guided post-filter below
            // always runs immediately after (both are only ever reached inside this same
            // denoiserEnabled branch), and its output is what actually belongs in lightmap.
            this.denoise.updateSwitches({ combineAlbedoDensity: true, forceFullSplit: false });
            this.frameIndex = (this.frameIndex + 1) % 4096;
            this.denoise.updateUniforms({ ...this.denoiserTunables, frameIndex: this.frameIndex });
            this.denoise.updateInputs(
                this.combinedIrradiance.view, albedoView, normalRoughnessView, densityView, this.filteredVariance.view,
                this.quadtreeMustSplit?.view ?? this.filteredVariance.view,
            );
            this.denoise.updateOutputs(this.denoiseScratch.view, this.blurSizeDebug.view, width, height);
            this.denoise.execute(encoder);

            // Guided post-filter over denoise's just-written result (see dither_filter.wgsl) -
            // EXPERIMENTAL, see this project's denoiser plan/conversation history: temporal jitter
            // alone already looked good enough that this may prove unnecessary. Reads
            // denoiseScratch (denoise's just-written, not-yet-final result) and writes directly to
            // lightmap's mip0 - its real final destination, not a staging copy - so this frame
            // never needs a copyTextureToTexture: each pass writes straight to wherever its result
            // actually belongs.
            this.ditherFilter.updateInputs(this.denoiseScratch.view, albedoView, normalRoughnessView, densityView);
            this.ditherFilter.updateOutputs(this.lightmap.getMipView(0), width, height);
            this.ditherFilter.execute(encoder);
        }
        // When denoiserEnabled is false, denoise/ditherFilter never dispatch at all - lightmap's
        // mip0 and lightmapMipChain's first two mips were already written directly above by
        // computeVarianceAndMips's combined output; lightmapCascade's own levels are left
        // unregenerated in that mode (see the comment on that redirect above).
        if (this.denoiserEnabled) {
            this.ensureBlurCascadeAllocated(width, height);
            this.lightmapBlurCascade.regenerate(
                encoder,
                this.lightmap.getMipView(0),
                this.lightmapCascade,
                this.lightmapCascadeScratch,
                this.lightmapMipChain,
                width,
                height,
                this.denoiserTunables.blurCascadeSigma,
            );
        }
    }

    /**
     * Lazily (re)allocates lightmapCascade to match denoiserTunables.blurCascadeLevelCount - one
     * extra entry beyond the exposed count for LightmapBlurCascade's internal bridging level (see
     * its doc comment). A no-op once already sized correctly, so this is cheap to call every frame;
     * only actually reallocates right after blurCascadeLevelCount changes (including the first call
     * after a scene load, since loadFromScene resets allocatedBlurCascadeLevelCount to -1).
     */
    private ensureBlurCascadeAllocated(width: number, height: number): void {
        const desiredLevelCount = Math.max(0, Math.round(this.denoiserTunables.blurCascadeLevelCount));
        if (desiredLevelCount === this.allocatedBlurCascadeLevelCount) {
            return;
        }
        for (const texture of this.lightmapCascade) {
            this.computedDataManager.releaseTexture(texture);
        }
        const cascadeUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        this.lightmapCascade = [];
        for (let i = 0; i < desiredLevelCount + 1; i++) {
            this.lightmapCascade.push(this.computedDataManager.acquireTexture(width, height, LIGHTMAP_FORMAT, cascadeUsage));
        }
        this.allocatedBlurCascadeLevelCount = desiredLevelCount;
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
        // same comparison, per the Unity reference this was ported from) - unlike before, nothing
        // here needs to change per level, since the irradiance range check no longer depends on
        // which absolute mip is being examined (see build_denoiser_quadtree.wgsl).
        // Reset each instance's per-frame uniform-buffer-slot cursor - see
        // BuildDenoiserQuadtreeOperation.beginFrame's doc comment. buildQuadtreeLevel0 only ever
        // dispatches once per frame, but resetting it too keeps both instances' usage uniform.
        this.buildQuadtreeLevel0.beginFrame();
        this.buildQuadtreeIterate.beginFrame();

        const level0Width = Math.max(1, width >> 1);
        const level0Height = Math.max(1, height >> 1);
        this.buildQuadtreeLevel0.updateUniforms({ ...this.denoiserTunables });
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
            this.buildQuadtreeIterate.updateUniforms({ ...this.denoiserTunables });
            this.buildQuadtreeIterate.updateInputsIterate(
                this.albedoMin.getMipView(level - 1), this.albedoMax.getMipView(level - 1),
                this.densityMinMaxVolatility.getMipView(level - 1), this.quadtreeMustSplit.getMipView(level - 1),
                this.filteredVariance.view,
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

        // Reset each light kind's per-frame uniform-buffer-slot cursor - see
        // ForwardMonteCarloOperation.beginFrame's doc comment. Unconditional (even for a kind with
        // no active lights this frame) since it's just a counter reset, no allocation.
        this.pointOperation.beginFrame();
        this.spotOperation.beginFrame();
        this.laserOperation.beginFrame();
        this.directionalOperation.beginFrame();
        this.ambientOperation.beginFrame();

        const { width, height } = this.simulation;
        const { raysPerFrame, integrationInterval: integrationIntervalRatio, photonBounces, surfaceBias, ambientScatterSoftness } = this.simulationTunables;
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
            const directionalSegment = kind === 'directional'
                ? computeDirectionalLightSegment(lightToTarget, width, height)
                : null;
            const directionalLightDirection: readonly [number, number] = directionalSegment?.direction ?? [0, 0];
            const directionalLightSegmentStart: readonly [number, number] = directionalSegment?.segmentStart ?? [0, 0];
            const directionalLightSegmentVector: readonly [number, number] = directionalSegment?.segmentVector ?? [0, 0];
            // Divides out the segment's own length so this light's total emitted power stays
            // independent of the target's resolution (see computeDirectionalLightSegment) - the
            // segment is only a sampling aperture, not a physical property of the light.
            const directionalEnergyScale = directionalSegment ? 1 / directionalSegment.length : 1;
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
                const photonEnergyScale = 0xFFFFFFFF / half.rays / integrationInterval * directionalEnergyScale;
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
                    directionalLightSegmentStart,
                    directionalLightSegmentVector,
                    lightPinch,
                    integrationInterval,
                    integrationIntervalSquared,
                    rays: half.rays,
                    surfaceBias,
                    ambientScatterSoftness,
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
