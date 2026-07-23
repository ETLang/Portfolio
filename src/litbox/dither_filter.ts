import { ComputeOperation } from './compute_operation.ts';
import shaderCode from './shaders/dither_filter.wgsl?raw';
import { preprocessShader } from './shaders/shader_preprocessor.ts';

/**
 * Small guided post-filter cleaning up denoise.wgsl's residual per-pixel dither noise (the
 * tradeoff its temporal seedJitter makes for the coherent lattice-alignment Moire it fixes) - see
 * dither_filter.wgsl and this project's denoiser plan/conversation history for the full argument.
 * No uniforms - every weighting constant is fixed in the shader (experimental pass, not yet wired
 * into the portfolio page's tunables panel).
 */
export class DitherFilterOperation extends ComputeOperation {
    constructor(device: GPUDevice) {
        super(device, preprocessShader(shaderCode), 'main');
    }

    public updateInputs(lightmap: GPUTextureView, albedo: GPUTextureView, normalRoughness: GPUTextureView, density: GPUTextureView): void {
        this.setInputs([
            { binding: 0, resource: lightmap },
            { binding: 1, resource: albedo },
            { binding: 2, resource: normalRoughness },
            { binding: 3, resource: density },
        ]);
    }

    public updateOutputs(output: GPUTextureView, width: number, height: number): void {
        this.setOutputs([{ binding: 0, resource: output }]);
        this.setDispatchExtent(width, height);
    }
}
