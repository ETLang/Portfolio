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
}

/** Historical default - always overridden by the caller's real value before the first dispatch, see updateSwitches. */
const DEFAULT_SWITCHES: DenoiseSwitches = { combineAlbedoDensity: true };

/**
 * Denoiser stub - see this project's denoiser plan. Algorithmically a passthrough (copies mip0 of
 * the combined irradiance straight through, no blur) so the pipeline keeps rendering a correct
 * image while the actual size-argument/guided-blur algorithm gets designed separately; also where
 * the final albedo/density combination now happens (moved here from
 * ConvertPhotonIrradianceToHdrOperation - see denoise.wgsl).
 */
export class DenoiseOperation extends ComputeOperation {
    constructor(device: GPUDevice) {
        super(device, preprocessShader(shaderCode, toDefines(DEFAULT_SWITCHES)), 'main');
    }

    /** See CLAUDE.md's updateSwitches/pipelineDirty pattern - a no-op (via setShaderCode) if `switches` matches what's already compiled. */
    public updateSwitches(switches: DenoiseSwitches): void {
        this.setShaderCode(preprocessShader(shaderCode, toDefines(switches)));
    }

    public updateInputs(combinedIrradianceMip0: GPUTextureView, albedo: GPUTextureView, density: GPUTextureView): void {
        this.setInputs([
            { binding: 0, resource: combinedIrradianceMip0 },
            { binding: 1, resource: albedo },
            { binding: 2, resource: density },
        ]);
    }

    public updateOutputs(output: GPUTextureView, width: number, height: number): void {
        this.setOutputs([{ binding: 0, resource: output }]);
        this.setDispatchExtent(width, height);
    }
}

function toDefines(switches: DenoiseSwitches): ShaderDefines {
    return switches.combineAlbedoDensity ? { COMBINE_ALBEDO_DENSITY: true } : {};
}
