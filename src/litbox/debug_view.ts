import debugViewBlitShaderCode from './shaders/debug_view_blit.wgsl?raw';
import { preprocessShader } from './shaders/shader_preprocessor.ts';

/**
 * One named, selectable debug view: a live source texture plus which display mode
 * debug_view_blit.wgsl should use to interpret it. `getSourceView` is a closure, not a captured
 * `GPUTextureView`, because the underlying texture can be reacquired (e.g. on scene reload) -
 * callers must look it up fresh each time a view is displayed, never cache the result.
 */
export interface DebugView {
    getSourceView(): GPUTextureView | null;
    mode: number;
}

/** debug_view_blit.wgsl's fragment shader modes - see its file header for what each one does. */
export const DEBUG_VIEW_MODE = {
    PASSTHROUGH: 0,
    DENSITY: 1,
    NORMAL_REMAP: 2,
    ALPHA_AS_LUMINANCE: 3,
    HDR_SCALED: 4,
} as const;

/**
 * Debug-only pass: blits whichever single DebugView is currently selected (see
 * LitboxSceneRenderer.debugView) to the swapchain, in place of the normal render - not part of
 * the normal render path. Structurally identical to TonemapResources (a fullscreen-triangle
 * blit), but samples via textureSampleLevel + a non-filtering sampler so it works uniformly
 * regardless of which source's texture format is bound - see the shader's file header for why.
 * Knows nothing about where a DebugView's texture comes from; LitboxSceneRenderer owns the
 * registry mapping view names to DebugViews, so this class stays reusable for any future debug
 * view, not just the G-Buffer's.
 */
export class DebugViewBlitResources {
    private device: GPUDevice;
    private pipeline: GPURenderPipeline;
    private bindGroupLayout: GPUBindGroupLayout;
    private uniformBuffer: GPUBuffer;
    private sampler: GPUSampler;
    private cachedBindGroup: GPUBindGroup | null = null;
    private cachedSourceView: GPUTextureView | null = null;

    constructor(device: GPUDevice, presentationFormat: GPUTextureFormat) {
        this.device = device;
        this.sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
            ],
        });

        const shaderModule = device.createShaderModule({ code: preprocessShader(debugViewBlitShaderCode) });
        this.pipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
            vertex: { module: shaderModule, entryPoint: 'vertex_main' },
            fragment: { module: shaderModule, entryPoint: 'fragment_main', targets: [{ format: presentationFormat }] },
            primitive: { topology: 'triangle-list' },
        });

        this.uniformBuffer = device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    public apply(passEncoder: GPURenderPassEncoder, sourceView: GPUTextureView, mode: number, scale: number): void {
        const uniformData = new ArrayBuffer(8);
        new DataView(uniformData).setUint32(0, mode, true);
        new DataView(uniformData).setFloat32(4, scale, true);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        if (this.cachedSourceView !== sourceView) {
            this.cachedBindGroup = this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: sourceView },
                    { binding: 2, resource: this.sampler },
                ],
            });
            this.cachedSourceView = sourceView;
        }

        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.cachedBindGroup!);
        passEncoder.draw(6);
    }
}
