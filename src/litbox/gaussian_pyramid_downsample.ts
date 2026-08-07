import { ComputeOperation } from './compute_operation.ts';
import shaderCode from './shaders/gaussian_pyramid_downsample.wgsl?raw';
import { preprocessShader, type ShaderDefines } from './shaders/shader_preprocessor.ts';

export interface GaussianPyramidDownsampleSwitches {
    /**
     * Re-weights each tap by 1/(1+luminance) (Karis average) instead of using the plain binomial
     * weight - see gaussian_pyramid_downsample.wgsl's #ifdef KARIS_WEIGHTED doc comment for why:
     * bounds the worst-case HDR-spike/thin-feature aliasing step regardless of brightness,
     * without meaningfully dimming wide, locally-uniform bright regions. LightmapBlurPyramid sets
     * this true unconditionally, for every level - false is kept only as the other half of this
     * switch for isolating/comparing the two later (see denoise.wgsl's FORCE_FULL_SPLIT for the
     * same pattern), not currently used by anything.
     */
    karisWeighted: boolean;
}

const DEFAULT_SWITCHES: GaussianPyramidDownsampleSwitches = { karisWeighted: false };

/**
 * One dispatch's worth of the lightmap Gaussian blur pyramid - see gaussian_pyramid_downsample.wgsl
 * for the 5x5 binomial kernel and lightmap_blur_pyramid.ts (this operation's only caller) for how
 * a whole mip chain is built from repeated calls to it. Fixed to rgba16float (lightmap's own
 * format) - unlike MipDownsampleOperation, nothing else is expected to reuse this, so there's no
 * OUTPUT_FORMAT switch to keep it format-agnostic.
 */
export class GaussianPyramidDownsampleOperation extends ComputeOperation {
    constructor(device: GPUDevice) {
        super(device, preprocessShader(shaderCode, toDefines(DEFAULT_SWITCHES)), 'main');
    }

    /** See CLAUDE.md's updateSwitches/pipelineDirty pattern - a no-op (via setShaderCode) if `switches` matches what's already compiled. */
    public updateSwitches(switches: GaussianPyramidDownsampleSwitches): void {
        this.setShaderCode(preprocessShader(shaderCode, toDefines(switches)));
    }

    public updateInputs(sourceMipView: GPUTextureView): void {
        this.setInputs([{ binding: 0, resource: sourceMipView }]);
    }

    public updateOutputs(destMipView: GPUTextureView, width: number, height: number): void {
        this.setOutputs([{ binding: 0, resource: destMipView }]);
        this.setDispatchExtent(width, height);
    }
}

function toDefines(switches: GaussianPyramidDownsampleSwitches): ShaderDefines {
    return switches.karisWeighted ? { KARIS_WEIGHTED: true } : {};
}
