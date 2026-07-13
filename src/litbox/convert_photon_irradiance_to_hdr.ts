import { ComputeOperation } from './compute_operation.ts';
import shaderCode from './shaders/convert_photon_irradiance_to_hdr.wgsl?raw';

export interface ConvertPhotonIrradianceToHdrUniforms {
    /** (width * height) / 4294967295 - converts the raw atomic accumulator back into a float range. See the wgsl file for the full explanation. */
    hdrScale: number;
}

/**
 * Converts the photon-receptor buffer's raw irradiance into the final HDR lightmap by multiplying
 * by albedo and density - see convert_photon_irradiance_to_hdr.wgsl for the exact math and its
 * relationship to Unity's ConvertToHDR kernel.
 */
export class ConvertPhotonIrradianceToHdrOperation extends ComputeOperation {
    private uniformBuffer: GPUBuffer;

    constructor(device: GPUDevice) {
        super(device, shaderCode, 'main');
        this.uniformBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.setUniforms([{ binding: 0, resource: { buffer: this.uniformBuffer } }]);
    }

    public updateUniforms(uniforms: ConvertPhotonIrradianceToHdrUniforms): void {
        this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([uniforms.hdrScale]));
    }

    public updateInputs(photonBuffer: GPUBuffer, albedo: GPUTextureView, density: GPUTextureView): void {
        this.setInputs([
            { binding: 0, resource: { buffer: photonBuffer } },
            { binding: 1, resource: albedo },
            { binding: 2, resource: density },
        ]);
    }

    public updateOutputs(output: GPUTextureView, width: number, height: number): void {
        this.setOutputs([{ binding: 0, resource: output }]);
        this.setDispatchExtent(width, height);
    }
}
