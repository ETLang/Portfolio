import { ComputeOperation } from './compute_operation.ts';
import shaderCode from './shaders/compute_volatility.wgsl?raw';
import { preprocessShader } from './shaders/shader_preprocessor.ts';

/**
 * Normal-based edge detector feeding the baked denoiser quadtree - see compute_volatility.wgsl and
 * this project's denoiser plan. No uniforms/switches; one dispatch, full G-Buffer resolution,
 * mip0 only (every quadtree level above propagates it via max-reduction).
 */
export class ComputeVolatilityOperation extends ComputeOperation {
    constructor(device: GPUDevice) {
        super(device, preprocessShader(shaderCode), 'main');
    }

    public updateInputs(normalRoughness: GPUTextureView): void {
        this.setInputs([{ binding: 0, resource: normalRoughness }]);
    }

    public updateOutputs(volatility: GPUTextureView, width: number, height: number): void {
        this.setOutputs([{ binding: 0, resource: volatility }]);
        this.setDispatchExtent(width, height);
    }
}
