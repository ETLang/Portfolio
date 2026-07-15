import { ComputeOperation } from './compute_operation.ts';
import shaderCode from './shaders/compute_variance_and_mips.wgsl?raw';
import { preprocessShader } from './shaders/shader_preprocessor.ts';

/**
 * Fused mean/variance/mip-generation pass - a structural port of Unity's confirmed-live
 * ComputeVarianceAndNMipsFromSamplePair (TracerPostProcessing.compute) using workgroup shared
 * memory, not a workaround for it - see compute_variance_and_mips.wgsl and this project's
 * denoiser plan. One dispatch produces combinedIrradiance's mip0-mip2 (the mean(A,B) signal used
 * everywhere downstream) and rawVariance at mip2 (quarter resolution). Capped at mip2 (4
 * simultaneous storage-texture outputs) to stay within WebGPU's guaranteed-minimum
 * maxStorageTexturesPerShaderStage of 4 - see the shader's file header; MipDownsampleOperation
 * continues combinedIrradiance past mip2.
 *
 * No uniforms - dispatch extent and all mip resolutions are derived entirely from the output
 * textures given to updateOutputs.
 */
export class ComputeVarianceAndMipsOperation extends ComputeOperation {
    constructor(device: GPUDevice) {
        super(device, preprocessShader(shaderCode), 'main');
    }

    public updateInputs(irradianceA: GPUTextureView, irradianceB: GPUTextureView): void {
        this.setInputs([
            { binding: 0, resource: irradianceA },
            { binding: 1, resource: irradianceB },
        ]);
    }

    public updateOutputs(
        combinedMip0: GPUTextureView,
        combinedMip1: GPUTextureView,
        combinedMip2: GPUTextureView,
        rawVariance: GPUTextureView,
        mip0Width: number,
        mip0Height: number,
    ): void {
        this.setOutputs([
            { binding: 0, resource: combinedMip0 },
            { binding: 1, resource: combinedMip1 },
            { binding: 2, resource: combinedMip2 },
            { binding: 3, resource: rawVariance },
        ]);
        this.setDispatchExtent(mip0Width, mip0Height);
    }
}
