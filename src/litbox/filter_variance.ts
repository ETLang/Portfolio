import { ComputeOperation } from './compute_operation.ts';
import shaderCode from './shaders/filter_variance.wgsl?raw';
import { preprocessShader } from './shaders/shader_preprocessor.ts';

/**
 * Bilateral-filters rawVariance (quarter resolution) using G-Buffer/irradiance mip2 evidence -
 * structurally matches Unity's confirmed-live FilterVariance kernel, see filter_variance.wgsl and
 * this project's denoiser plan. Thresholds are TBD/tunable - this is evidence-gathering plumbing
 * feeding the not-yet-designed size argument, not the final algorithm.
 */
export class FilterVarianceOperation extends ComputeOperation {
    constructor(device: GPUDevice) {
        super(device, preprocessShader(shaderCode), 'main');
    }

    public updateInputs(rawVariance: GPUTextureView, albedoMip2: GPUTextureView, combinedIrradianceMip2: GPUTextureView): void {
        this.setInputs([
            { binding: 0, resource: rawVariance },
            { binding: 1, resource: albedoMip2 },
            { binding: 2, resource: combinedIrradianceMip2 },
        ]);
    }

    public updateOutputs(filteredVariance: GPUTextureView, width: number, height: number): void {
        this.setOutputs([{ binding: 0, resource: filteredVariance }]);
        this.setDispatchExtent(width, height);
    }
}
