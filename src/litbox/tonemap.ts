import tonemapShaderCode from './shaders/tonemap.wgsl?raw';
import { preprocessShader } from './shaders/shader_preprocessor.ts';

export interface TonemapUniforms {
    /** Added in log10 space before the filmic curve - see tonemap.wgsl for why its effective scale differs from a plain exposure stop. */
    exposure: number;
    /** When false, the filmic curve is bypassed entirely and the raw HDR value is written straight to the swapchain (clipping to whatever the presentation format's range allows). */
    enabled: boolean;
}

/**
 * Final pass: HDR frame texture -> swapchain. This is the one pass that is
 * genuinely fullscreen/screen-aligned (unlike the simulation composite),
 * so it draws a fullscreen quad (2 triangles - see tonemap.wgsl for why not
 * a single oversized triangle). Applies a UE5-style filmic tonemap curve;
 * exposure is added in log10 space (not an exp2 pre-multiply), so its
 * effective scale differs from a plain exposure stop.
 *
 * Not a ComputeOperation subclass - that base (see CLAUDE.md's "Compute-shader operation
 * architecture") is built around GPUComputePipeline/dispatchWorkgroups/@workgroup_size, none of
 * which apply to a render pipeline with a draw() call. This instead follows the same conventions
 * by hand: bespoke updateUniforms/updateInputs methods, a dirty-tracked bind group, and skipping
 * the uniform buffer write entirely when the value hasn't changed.
 */
export class TonemapResources {
    private device: GPUDevice;
    private pipeline: GPURenderPipeline;
    private bindGroupLayout: GPUBindGroupLayout;
    private uniformBuffer: GPUBuffer;
    private sampler: GPUSampler;

    private lastExposure: number | null = null;
    private lastEnabled: boolean | null = null;
    private hdrView: GPUTextureView | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private bindGroupDirty = true;

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

        const shaderModule = device.createShaderModule({ code: preprocessShader(tonemapShaderCode) });
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

    /** A no-op if `uniforms` describes the same exposure/enabled state already written. */
    public updateUniforms(uniforms: TonemapUniforms): void {
        if (this.lastExposure === uniforms.exposure && this.lastEnabled === uniforms.enabled) {
            return;
        }
        this.lastExposure = uniforms.exposure;
        this.lastEnabled = uniforms.enabled;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([uniforms.exposure, uniforms.enabled ? 1.0 : 0.0]));
    }

    /** A no-op if `hdrView` is the same view already bound. */
    public updateInputs(hdrView: GPUTextureView): void {
        if (this.hdrView === hdrView) {
            return;
        }
        this.hdrView = hdrView;
        this.bindGroupDirty = true;
    }

    /** Rebuilds the bind group if dirty, then draws the fullscreen quad. */
    public execute(passEncoder: GPURenderPassEncoder): void {
        if (this.bindGroupDirty) {
            this.bindGroup = this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.uniformBuffer } },
                    { binding: 1, resource: this.hdrView! },
                    { binding: 2, resource: this.sampler },
                ],
            });
            this.bindGroupDirty = false;
        }

        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.bindGroup!);
        passEncoder.draw(6);
    }
}
