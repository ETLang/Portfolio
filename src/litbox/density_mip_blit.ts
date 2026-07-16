import densityMipBlitShaderCode from './shaders/density_mip_blit.wgsl?raw';
import { preprocessShader } from './shaders/shader_preprocessor.ts';

const DENSITY_FORMAT: GPUTextureFormat = 'rg16float';

/**
 * Density's mip-chain generation - a render-pass sibling of MipDownsampleOperation (see this
 * project's denoiser plan and density_mip_blit.wgsl's file header for why Density needs its own
 * mechanism: rg16float isn't a valid WGSL storage-texture texel format, so a compute pass can't
 * write its mips via textureStore).
 *
 * Not a ComputeOperation subclass - that base is built around GPUComputePipeline/
 * dispatchWorkgroups, none of which apply to a render pipeline with a draw() call (same rationale
 * as TonemapResources/DebugViewBlitResources) - follows the same conventions by hand instead: a
 * bespoke updateInputs method and a dirty-tracked bind group.
 */
export class DensityMipBlitResources {
    private device: GPUDevice;
    private pipeline: GPURenderPipeline;
    private bindGroupLayout: GPUBindGroupLayout;
    private sampler: GPUSampler;
    private cachedBindGroup: GPUBindGroup | null = null;
    private cachedSourceView: GPUTextureView | null = null;

    constructor(device: GPUDevice) {
        this.device = device;
        this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ],
        });

        const shaderModule = device.createShaderModule({ code: preprocessShader(densityMipBlitShaderCode) });
        this.pipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
            vertex: { module: shaderModule, entryPoint: 'vertex_main' },
            fragment: { module: shaderModule, entryPoint: 'fragment_main', targets: [{ format: DENSITY_FORMAT }] },
            primitive: { topology: 'triangle-list' },
        });
    }

    /** A no-op if `sourceMipView` is the same view already bound. */
    public updateInputs(sourceMipView: GPUTextureView): void {
        if (this.cachedSourceView === sourceMipView) {
            return;
        }
        this.cachedSourceView = sourceMipView;
        this.cachedBindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: sourceMipView },
                { binding: 1, resource: this.sampler },
            ],
        });
    }

    /** Opens its own render pass targeting `destMipView` and draws the fullscreen quad - one call per mip level. */
    public execute(encoder: GPUCommandEncoder, destMipView: GPUTextureView): void {
        const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: destMipView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.cachedBindGroup!);
        pass.draw(6);
        pass.end();
    }
}
