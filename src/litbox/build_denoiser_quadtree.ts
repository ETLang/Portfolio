import { ComputeOperation } from './compute_operation.ts';
import shaderCode from './shaders/build_denoiser_quadtree.wgsl?raw';
import { preprocessShader, type ShaderDefines } from './shaders/shader_preprocessor.ts';

export interface BuildDenoiserQuadtreeSwitches {
    /** See build_denoiser_quadtree.wgsl - reads raw G-Buffer data if true, the previous level's own outputs if false. */
    level0: boolean;
}

export interface BuildDenoiserQuadtreeUniforms {
    albedoLuminanceThreshold: number;
    albedoChromaThreshold: number;
    logDensityThreshold: number;
    volatilityThreshold: number;
    detailThreshold: number;
    varianceGateScale: number;
    /** G-Buffer/irradiance-space mip this dispatch is building evidence for (this operation's own output mip + 1). */
    currentGBufferMip: number;
}

const UNIFORM_FIELD_COUNT = 7;

/** Historical default - callers construct one instance per switch value and never toggle it afterward, see the shader's file header for why. */
const DEFAULT_SWITCHES: BuildDenoiserQuadtreeSwitches = { level0: true };

/**
 * Builds one level of the baked min/max-range quadtree that backs denoise.wgsl's ShouldSplit -
 * see build_denoiser_quadtree.wgsl and this project's denoiser plan. One execute() call builds
 * exactly one level; callers loop this for the whole chain - same reuse pattern as
 * MipDownsampleOperation. Construct two instances (one per switch value) rather than toggling a
 * single instance's switch every frame - see the shader's file header.
 */
export class BuildDenoiserQuadtreeOperation extends ComputeOperation {
    private uniformBuffer: GPUBuffer;
    private linearSampler: GPUSampler;

    constructor(device: GPUDevice) {
        super(device, preprocessShader(shaderCode, toDefines(DEFAULT_SWITCHES)), 'main');
        this.linearSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
        this.uniformBuffer = device.createBuffer({
            size: UNIFORM_FIELD_COUNT * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.setUniforms([{ binding: 0, resource: { buffer: this.uniformBuffer } }]);
    }

    /** Intended to be called once, right after construction - not part of the per-frame hot path, see the shader's file header. */
    public updateSwitches(switches: BuildDenoiserQuadtreeSwitches): void {
        this.setShaderCode(preprocessShader(shaderCode, toDefines(switches)));
    }

    /** currentGBufferMip changes every call in the per-frame chain loop, so this always writes (no dedupe cache, unlike most other operations' updateUniforms). */
    public updateUniforms(uniforms: BuildDenoiserQuadtreeUniforms): void {
        this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([
            uniforms.albedoLuminanceThreshold,
            uniforms.albedoChromaThreshold,
            uniforms.logDensityThreshold,
            uniforms.volatilityThreshold,
            uniforms.detailThreshold,
            uniforms.varianceGateScale,
            uniforms.currentGBufferMip,
        ]));
    }

    /** LEVEL0 variant's inputs - raw G-Buffer mip0 + volatility mip0. */
    public updateInputsLevel0(
        albedoMip0: GPUTextureView,
        densityMip0: GPUTextureView,
        volatilityMip0: GPUTextureView,
        combinedIrradiance: GPUTextureView,
        filteredVariance: GPUTextureView,
    ): void {
        this.setInputs([
            { binding: 0, resource: albedoMip0 },
            { binding: 1, resource: densityMip0 },
            { binding: 2, resource: volatilityMip0 },
            { binding: 3, resource: combinedIrradiance },
            { binding: 4, resource: filteredVariance },
            { binding: 5, resource: this.linearSampler },
        ]);
    }

    /** Iterative variant's inputs - the previous level's own min/max/volatility/mustSplit outputs. */
    public updateInputsIterate(
        prevAlbedoMin: GPUTextureView,
        prevAlbedoMax: GPUTextureView,
        prevDensityMinMaxVolatility: GPUTextureView,
        prevQuadtreeMustSplit: GPUTextureView,
        combinedIrradiance: GPUTextureView,
        filteredVariance: GPUTextureView,
    ): void {
        this.setInputs([
            { binding: 0, resource: prevAlbedoMin },
            { binding: 1, resource: prevAlbedoMax },
            { binding: 2, resource: prevDensityMinMaxVolatility },
            { binding: 3, resource: prevQuadtreeMustSplit },
            { binding: 4, resource: combinedIrradiance },
            { binding: 5, resource: filteredVariance },
            { binding: 6, resource: this.linearSampler },
        ]);
    }

    public updateOutputs(
        albedoMin: GPUTextureView,
        albedoMax: GPUTextureView,
        densityMinMaxVolatility: GPUTextureView,
        quadtreeMustSplit: GPUTextureView,
        width: number,
        height: number,
    ): void {
        this.setOutputs([
            { binding: 0, resource: albedoMin },
            { binding: 1, resource: albedoMax },
            { binding: 2, resource: densityMinMaxVolatility },
            { binding: 3, resource: quadtreeMustSplit },
        ]);
        this.setDispatchExtent(width, height);
    }
}

function toDefines(switches: BuildDenoiserQuadtreeSwitches): ShaderDefines {
    return switches.level0 ? { LEVEL0: true } : {};
}
