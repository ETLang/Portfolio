import { ComputeOperation } from './compute_operation.ts';
import shaderCode from './shaders/denoise.wgsl?raw';
import { preprocessShader, type ShaderDefines } from './shaders/shader_preprocessor.ts';

export interface DenoiseSwitches {
    /**
     * Whether this pass multiplies by albedo/density to produce the final lit image, or leaves
     * its output as plain irradiance - see denoise.wgsl's file header. Normally true (this is the
     * final stage); false is for debugging only.
     */
    combineAlbedoDensity: boolean;
    /**
     * Debug mode (see this project's denoiser plan): forces ShouldSplit to always split, ignoring
     * the baked quadtree entirely, so every seed fully descends to mip 0. Slow by design - useful
     * for re-isolating DecideBlurSize/DecideWeight's own quality from a split-heuristic bug.
     * Normally false (SimulationResources.run() only sets it true for debugging).
     */
    forceFullSplit: boolean;
}

/** Historical default - always overridden by the caller's real value before the first dispatch, see updateSwitches. */
const DEFAULT_SWITCHES: DenoiseSwitches = { combineAlbedoDensity: true, forceFullSplit: true };

/** Mirrors denoise.wgsl's DenoiseUniforms struct field-for-field, offset 0. */
export interface DenoiseUniforms {
    /** Multiplies filteredVariance before feeding DecideBlurSize's max() - see denoise.wgsl. */
    varianceScale: number;
    /** Mean-luminance floor below which a pixel is treated as under-sampled/dark regardless of its (possibly deceptively low) variance. */
    darknessNoiseFloor: number;
    /** Ceiling on the starting mip DecideBlurSize can choose - also clamped in-shader to combinedIrradiance's actual mip count. */
    maxBlurMip: number;
    /** DecideWeight's albedo-distance tolerance - larger allows blending across more different albedos. */
    albedoSensitivity: number;
    /** DecideWeight's optical-depth-distance tolerance. */
    densitySensitivity: number;
    /** DecideWeight's normal-similarity exponent (pow(dot, this)) - larger makes the normal term more selective. */
    normalSensitivity: number;
    /** Adaptive radiance-similarity sigma when the center pixel's own variance is near zero (tight = selective). */
    sigmaLuminanceTight: number;
    /** Adaptive radiance-similarity sigma when the center pixel's own variance is high (loose = permissive). */
    sigmaLuminanceLoose: number;
    /** Denominator of the smoothstep gating sigmaLuminanceTight/Loose by centerVariance - see filter_variance.wgsl's identical K_LUMINANCE. */
    kLuminance: number;
    /**
     * Distance-bias split cutoff, in node-relative texels (see denoise.wgsl's shouldSplit() doc
     * comment for the exact normalization and why it's node-relative, not seed-relative like
     * decideWeight's own spatial falloff). Smaller values cut off splitting farther from the query
     * pixel more aggressively (faster, coarser far-field detail); larger values let more distant
     * branches keep resolving fine structure the final weighted average will barely use.
     */
    maxSplitDistance: number;
}

const UNIFORM_FIELD_COUNT = 10;

/**
 * Hierarchical guided blur over combinedIrradiance - see denoise.wgsl and this project's denoiser
 * plan for the full algorithm/argument. Also where the final albedo/density combination happens
 * (moved here from ConvertPhotonIrradianceToHdrOperation, since combination must happen after
 * variance computation and after denoising).
 */
export class DenoiseOperation extends ComputeOperation {
    private uniformBuffer: GPUBuffer;
    private lastUniforms: DenoiseUniforms | null = null;
    /** Material-identity sampling (albedo/normalRoughness/density) - never blend across a boundary. */
    private nearestSampler: GPUSampler;
    /** Smoother reconstruction of the signal actually being blurred (combinedIrradiance/filteredVariance). */
    private linearSampler: GPUSampler;

    constructor(device: GPUDevice) {
        super(device, preprocessShader(shaderCode, toDefines(DEFAULT_SWITCHES)), 'main');
        this.nearestSampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
        this.linearSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
        this.uniformBuffer = device.createBuffer({
            size: UNIFORM_FIELD_COUNT * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.setUniforms([{ binding: 0, resource: { buffer: this.uniformBuffer } }]);
    }

    /** See CLAUDE.md's updateSwitches/pipelineDirty pattern - a no-op (via setShaderCode) if `switches` matches what's already compiled. */
    public updateSwitches(switches: DenoiseSwitches): void {
        this.setShaderCode(preprocessShader(shaderCode, toDefines(switches)));
    }

    public updateUniforms(uniforms: DenoiseUniforms): void {
        if (this.lastUniforms
            && this.lastUniforms.varianceScale === uniforms.varianceScale
            && this.lastUniforms.darknessNoiseFloor === uniforms.darknessNoiseFloor
            && this.lastUniforms.maxBlurMip === uniforms.maxBlurMip
            && this.lastUniforms.albedoSensitivity === uniforms.albedoSensitivity
            && this.lastUniforms.densitySensitivity === uniforms.densitySensitivity
            && this.lastUniforms.normalSensitivity === uniforms.normalSensitivity
            && this.lastUniforms.sigmaLuminanceTight === uniforms.sigmaLuminanceTight
            && this.lastUniforms.sigmaLuminanceLoose === uniforms.sigmaLuminanceLoose
            && this.lastUniforms.kLuminance === uniforms.kLuminance
            && this.lastUniforms.maxSplitDistance === uniforms.maxSplitDistance) {
            return;
        }
        // Snapshot into a fresh object, not a reference to `uniforms` itself - callers may pass a
        // persistent, externally-mutable object (e.g. SimulationResources.denoiserTunables, live-
        // edited by the portfolio page's config panel) rather than a fresh literal each call. If
        // this stored the reference directly, `this.lastUniforms` and a later `uniforms` argument
        // would end up being the *same* mutated object, making every field-by-field comparison
        // above trivially true (an object always equals its own current fields) - silently
        // freezing the GPU-side uniforms at whatever was first uploaded, no matter what the caller
        // changes afterward.
        this.lastUniforms = { ...uniforms };
        this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([
            uniforms.varianceScale,
            uniforms.darknessNoiseFloor,
            uniforms.maxBlurMip,
            uniforms.albedoSensitivity,
            uniforms.densitySensitivity,
            uniforms.normalSensitivity,
            uniforms.sigmaLuminanceTight,
            uniforms.sigmaLuminanceLoose,
            uniforms.kLuminance,
            uniforms.maxSplitDistance,
        ]));
    }

    /**
     * Each texture's full mip chain (not just mip0) - the guided blur samples arbitrary levels of
     * each. filteredVariance has only one (quarter-res) level; quadtreeMustSplit has its own
     * 0-indexed chain at half G-Buffer resolution (see build_denoiser_quadtree.wgsl) - both are
     * passed the same way for a uniform call shape. quadtreeMustSplit is bound even while
     * forceFullSplit is on (unused in that mode, but the WGSL declaration is unconditional - see
     * denoise.wgsl).
     */
    public updateInputs(
        combinedIrradiance: GPUTextureView,
        albedo: GPUTextureView,
        normalRoughness: GPUTextureView,
        density: GPUTextureView,
        filteredVariance: GPUTextureView,
        quadtreeMustSplit: GPUTextureView,
    ): void {
        this.setInputs([
            { binding: 0, resource: combinedIrradiance },
            { binding: 1, resource: albedo },
            { binding: 2, resource: normalRoughness },
            { binding: 3, resource: density },
            { binding: 4, resource: filteredVariance },
            { binding: 5, resource: this.linearSampler },
            { binding: 6, resource: this.nearestSampler },
            { binding: 7, resource: quadtreeMustSplit },
        ]);
    }

    public updateOutputs(output: GPUTextureView, width: number, height: number): void {
        this.setOutputs([{ binding: 0, resource: output }]);
        this.setDispatchExtent(width, height);
    }
}

function toDefines(switches: DenoiseSwitches): ShaderDefines {
    const defines: ShaderDefines = {};
    if (switches.combineAlbedoDensity) {
        defines.COMBINE_ALBEDO_DENSITY = true;
    }
    if (switches.forceFullSplit) {
        defines.FORCE_FULL_SPLIT = true;
    }
    return defines;
}
