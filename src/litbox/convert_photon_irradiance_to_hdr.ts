import { ComputeOperation } from './compute_operation.ts';
import shaderCode from './shaders/convert_photon_irradiance_to_hdr.wgsl?raw';
import { preprocessShader, type ShaderDefines } from './shaders/shader_preprocessor.ts';

export interface ConvertPhotonIrradianceToHdrUniforms {
    /** (width * height) / 4294967295 - converts the raw atomic accumulator back into a float range. See the wgsl file for the full explanation. */
    hdrScale: number;
}

export interface ConvertPhotonIrradianceToHdrSwitches {
    /**
     * Whether this pass multiplies by albedo/density itself, or leaves the output as plain
     * irradiance - see convert_photon_irradiance_to_hdr.wgsl's file header. Normally false: the
     * combination now happens later, in the denoiser stage, after variance computation and
     * denoising (this project's denoiser plan) - true is for debugging only.
     */
    combineAlbedoDensity: boolean;
}

/** Historical Unity-ported value, used only for the constructor's placeholder pre-updateSwitches shader compile - see updateSwitches, always called before the first real dispatch. */
const DEFAULT_SWITCHES: ConvertPhotonIrradianceToHdrSwitches = { combineAlbedoDensity: false };

/**
 * Converts the photon-receptor buffer's raw irradiance into two HDR textures, one per half of
 * this frame's two-way variance-estimation split (see this project's denoiser plan) - both halves
 * converted in one dispatch, since this pass is per-pixel, not ray-parallel, unlike the tracer
 * dispatches that actually populate the two halves (see forward_monte_carlo.ts/simulation.ts). See
 * convert_photon_irradiance_to_hdr.wgsl for the exact math and its relationship to Unity's
 * ConvertToHDR kernel.
 */
export class ConvertPhotonIrradianceToHdrOperation extends ComputeOperation {
    private uniformBuffer: GPUBuffer;
    private lastHdrScale: number | null = null;

    constructor(device: GPUDevice) {
        super(device, preprocessShader(shaderCode, toDefines(DEFAULT_SWITCHES)), 'main');
        this.uniformBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.setUniforms([{ binding: 0, resource: { buffer: this.uniformBuffer } }]);
    }

    /** See CLAUDE.md's updateSwitches/pipelineDirty pattern - a no-op (via setShaderCode) if `switches` matches what's already compiled. */
    public updateSwitches(switches: ConvertPhotonIrradianceToHdrSwitches): void {
        this.setShaderCode(preprocessShader(shaderCode, toDefines(switches)));
    }

    public updateUniforms(uniforms: ConvertPhotonIrradianceToHdrUniforms): void {
        if (this.lastHdrScale === uniforms.hdrScale) {
            return;
        }
        this.lastHdrScale = uniforms.hdrScale;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([uniforms.hdrScale]));
    }

    public updateInputs(photonBuffer: GPUBuffer, albedo: GPUTextureView, density: GPUTextureView): void {
        this.setInputs([
            { binding: 0, resource: { buffer: photonBuffer } },
            { binding: 1, resource: albedo },
            { binding: 2, resource: density },
        ]);
    }

    public updateOutputs(outputA: GPUTextureView, outputB: GPUTextureView, width: number, height: number): void {
        this.setOutputs([
            { binding: 0, resource: outputA },
            { binding: 1, resource: outputB },
        ]);
        this.setDispatchExtent(width, height);
    }
}

function toDefines(switches: ConvertPhotonIrradianceToHdrSwitches): ShaderDefines {
    return switches.combineAlbedoDensity ? { COMBINE_ALBEDO_DENSITY: true } : {};
}
