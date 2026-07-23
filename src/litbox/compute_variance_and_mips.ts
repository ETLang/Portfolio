import { ComputeOperation } from './compute_operation.ts';
import shaderCode from './shaders/compute_variance_and_mips.wgsl?raw';
import { preprocessShader, type ShaderDefines } from './shaders/shader_preprocessor.ts';

export interface ComputeVarianceAndMipsSwitches {
    /**
     * When true, combinedMip0/1/2 hold centerIrradiance * albedo * density (the final lit result)
     * instead of raw mean(A,B) irradiance, and rawVariance stops existing entirely - see
     * compute_variance_and_mips.wgsl. Set when SimulationResources.denoiserEnabled is false: in
     * that mode combinedMip0/1/2 are bound directly to `lightmap`'s own top 3 mip levels (see
     * SimulationResources.run()), since irradiance itself is never the wanted output when the
     * denoiser is off - only the combined lit image is, and this pass already computes each
     * pixel's mean anyway, so folding albedo/density in here (with albedo/density themselves
     * box-filtered down through the same shared-memory reduction as the mean) is far cheaper than
     * a dedicated dispatch just for a trivial per-pixel multiply.
     */
    combineAlbedoDensity: boolean;
}

/** Historical default - always overridden by the caller's real value before the first dispatch, see updateSwitches. */
const DEFAULT_SWITCHES: ComputeVarianceAndMipsSwitches = { combineAlbedoDensity: false };

/**
 * Fused mean/variance/mip-generation pass - a structural port of Unity's confirmed-live
 * ComputeVarianceAndNMipsFromSamplePair (TracerPostProcessing.compute) using workgroup shared
 * memory, not a workaround for it - see compute_variance_and_mips.wgsl and this project's
 * denoiser plan. One dispatch produces combinedMip0-2 (mean(A,B) irradiance when the denoiser is
 * enabled, or the final mean*albedo*density lit result - written directly into lightmap's own top
 * 3 mip levels - when it's disabled, see ComputeVarianceAndMipsSwitches) and, only in the enabled
 * case, rawVariance at mip2 (quarter resolution). Capped at mip2 (4 simultaneous storage-texture
 * outputs) to stay within WebGPU's guaranteed-minimum maxStorageTexturesPerShaderStage of 4 - see
 * the shader's file header; MipDownsampleOperation continues combinedIrradiance past mip2 when
 * the denoiser is enabled.
 */
export class ComputeVarianceAndMipsOperation extends ComputeOperation {
    constructor(device: GPUDevice) {
        super(device, preprocessShader(shaderCode, toDefines(DEFAULT_SWITCHES)), 'main');
    }

    /** See CLAUDE.md's updateSwitches/pipelineDirty pattern - a no-op (via setShaderCode) if `switches` matches what's already compiled. */
    public updateSwitches(switches: ComputeVarianceAndMipsSwitches): void {
        this.setShaderCode(preprocessShader(shaderCode, toDefines(switches)));
    }

    /** albedo/density are only meaningful - and only bound - when the combineAlbedoDensity switch is on (see updateSwitches); omit them otherwise. */
    public updateInputs(irradianceA: GPUTextureView, irradianceB: GPUTextureView, albedo?: GPUTextureView, density?: GPUTextureView): void {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: irradianceA },
            { binding: 1, resource: irradianceB },
        ];
        if (albedo && density) {
            entries.push({ binding: 2, resource: albedo }, { binding: 3, resource: density });
        }
        this.setInputs(entries);
    }

    /**
     * combinedMip0/1/2: pass combinedIrradiance's own mip views when the denoiser is enabled, or
     * lightmap's mip0/1/2 views directly when it's disabled (see ComputeVarianceAndMipsSwitches
     * and compute_variance_and_mips.wgsl) - no intermediate/staging texture either way. rawVariance
     * only exists - and should only be passed - when combineAlbedoDensity is off; the shader
     * doesn't declare that binding at all in the combine-mode variant, so passing it there would be
     * silently ignored.
     */
    public updateOutputs(
        combinedMip0: GPUTextureView,
        combinedMip1: GPUTextureView,
        combinedMip2: GPUTextureView,
        mip0Width: number,
        mip0Height: number,
        rawVariance?: GPUTextureView,
    ): void {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: combinedMip0 },
            { binding: 1, resource: combinedMip1 },
            { binding: 2, resource: combinedMip2 },
        ];
        if (rawVariance) {
            entries.push({ binding: 3, resource: rawVariance });
        }
        this.setOutputs(entries);
        this.setDispatchExtent(mip0Width, mip0Height);
    }
}

function toDefines(switches: ComputeVarianceAndMipsSwitches): ShaderDefines {
    const defines: ShaderDefines = {};
    if (switches.combineAlbedoDensity) {
        defines.COMBINE_ALBEDO_DENSITY = true;
    }
    return defines;
}
