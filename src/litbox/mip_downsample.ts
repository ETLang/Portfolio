import { ComputeOperation } from './compute_operation.ts';
import shaderCode from './shaders/mip_downsample.wgsl?raw';
import { preprocessShader, type ShaderDefines } from './shaders/shader_preprocessor.ts';

export interface MipDownsampleSwitches {
    /**
     * Must match the destination texture's actual format - WGSL storage-texture bindings need a
     * compile-time texel format (see mip_downsample.wgsl). Density (rg16float) isn't a valid
     * storage-texture format at all - see DensityMipBlitResources for that one instead.
     */
    outputFormat: 'rgba16float' | 'rgba8unorm';
}

const DEFAULT_SWITCHES: MipDownsampleSwitches = { outputFormat: 'rgba16float' };

/**
 * Generic single-mip-level box-filter downsample, reused for every mip chain in this project's
 * denoiser evidence-gathering pipeline (the G-Buffer's Albedo/NormalRoughness, and the combined
 * HDR irradiance/lightmap past mip4) - see mip_downsample.wgsl and this project's denoiser plan.
 * One execute() call downsamples exactly one source mip level into the next; callers loop this
 * for a whole chain.
 */
export class MipDownsampleOperation extends ComputeOperation {
    private linearSampler: GPUSampler;

    constructor(device: GPUDevice) {
        super(device, preprocessShader(shaderCode, toDefines(DEFAULT_SWITCHES)), 'main');
        this.linearSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    }

    /** See CLAUDE.md's updateSwitches/pipelineDirty pattern - a no-op (via setShaderCode) if `switches` matches what's already compiled. */
    public updateSwitches(switches: MipDownsampleSwitches): void {
        this.setShaderCode(preprocessShader(shaderCode, toDefines(switches)));
    }

    public updateInputs(sourceMipView: GPUTextureView): void {
        this.setInputs([
            { binding: 0, resource: sourceMipView },
            { binding: 1, resource: this.linearSampler },
        ]);
    }

    public updateOutputs(destMipView: GPUTextureView, width: number, height: number): void {
        this.setOutputs([{ binding: 0, resource: destMipView }]);
        this.setDispatchExtent(width, height);
    }
}

function toDefines(switches: MipDownsampleSwitches): ShaderDefines {
    return { OUTPUT_FORMAT: switches.outputFormat };
}
