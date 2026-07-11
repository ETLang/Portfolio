import tonemapShaderCode from './shaders/tonemap.wgsl?raw';

/**
 * Final pass: HDR frame texture -> swapchain. This is the one pass that is
 * genuinely fullscreen/screen-aligned (unlike the simulation composite),
 * so it draws a fullscreen quad (2 triangles - see tonemap.wgsl for why not
 * a single oversized triangle). Applies a UE5-style filmic tonemap curve;
 * exposure is added in log10 space (not an exp2 pre-multiply), so its
 * effective scale differs from a plain exposure stop.
 */
export class TonemapResources {
    private device: GPUDevice;
    private pipeline: GPURenderPipeline;
    private bindGroupLayout: GPUBindGroupLayout;
    private uniformBuffer: GPUBuffer;
    private sampler: GPUSampler;
    private cachedBindGroup: GPUBindGroup | null = null;
    private cachedHdrView: GPUTextureView | null = null;

    constructor(device: GPUDevice, presentationFormat: GPUTextureFormat) {
        this.device = device;
        this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                // rgba16float is filterable by default in WebGPU (unlike rgba32float).
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ],
        });

        const shaderModule = device.createShaderModule({ code: tonemapShaderCode });
        this.pipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
            vertex: { module: shaderModule, entryPoint: 'vertex_main' },
            fragment: { module: shaderModule, entryPoint: 'fragment_main', targets: [{ format: presentationFormat }] },
            primitive: { topology: 'triangle-list' },
        });

        this.uniformBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    public apply(passEncoder: GPURenderPassEncoder, hdrView: GPUTextureView, exposure: number): void {
        this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([exposure]));

        if (this.cachedHdrView !== hdrView) {
            this.cachedBindGroup = this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: hdrView },
                    { binding: 2, resource: this.sampler },
                ],
            });
            this.cachedHdrView = hdrView;
        }

        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.cachedBindGroup!);
        passEncoder.draw(6);
    }
}
