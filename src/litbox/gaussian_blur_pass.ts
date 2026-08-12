import { ComputeOperation } from './compute_operation.ts';
import shaderCode from './shaders/gaussian_blur_pass.wgsl?raw';
import { preprocessShader } from './shaders/shader_preprocessor.ts';

export interface GaussianBlurPassUniforms {
    /** Texel step direction for this pass: (1,0) for horizontal, (0,1) for vertical. */
    direction: [number, number];
    /** Tap radius in texels - the shader loops [-radius, radius]. */
    radius: number;
    /** Gaussian sigma for this pass, in texels. */
    sigma: number;
}

const UNIFORM_BUFFER_SIZE = 16; // direction: vec2<f32> (8) + radius: i32 (4) + sigma: f32 (4)

/**
 * One axis of a real, constant-resolution separable Gaussian blur - see gaussian_blur_pass.wgsl
 * and lightmap_blur_cascade.ts (this operation's only caller) for the full design. Unlike
 * GaussianPyramidDownsampleOperation, this never changes resolution between input and output -
 * radius/sigma are runtime uniforms, not compile-time switches, so a single instance is reused for
 * every axis and every cascade/bridging pass, re-parameterized per dispatch.
 */
export class GaussianBlurPassOperation extends ComputeOperation {
    /**
     * One small uniform buffer per dispatch within the current frame, not a single buffer reused
     * across calls - see CLAUDE.md's "Compute-shader operation architecture" for the general
     * pattern this fixes: LightmapBlurCascade calls updateUniforms()+
     * execute() on this SAME instance many times per frame (two passes per cascade/bridging
     * level), all recorded into one shared GPUCommandEncoder that isn't submitted until the very
     * end of the frame. GPUQueue.writeBuffer() is ordered strictly by JS call order on the queue's
     * timeline, but recording a compute pass into an encoder does not snapshot a bound buffer's
     * contents - every dispatch would silently read whichever write happened last (confirmed: this
     * produced blur that only ever looked vertical, since the last-written direction uniform,
     * (0,1), was the one every dispatch actually read at submit time). Grows lazily and is capped
     * at whatever the largest per-frame call count has been so far - beginFrame() resets nextSlot
     * to 0 without shrinking the pool.
     */
    private uniformBufferPool: GPUBuffer[] = [];
    private nextSlot = 0;

    constructor(device: GPUDevice) {
        super(device, preprocessShader(shaderCode), 'main');
    }

    public beginFrame(): void {
        this.nextSlot = 0;
    }

    public updateUniforms(uniforms: GaussianBlurPassUniforms): void {
        const data = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
        const view = new DataView(data);
        view.setFloat32(0, uniforms.direction[0], true);
        view.setFloat32(4, uniforms.direction[1], true);
        view.setInt32(8, Math.round(uniforms.radius), true);
        view.setFloat32(12, uniforms.sigma, true);

        let buffer = this.uniformBufferPool[this.nextSlot];
        if (!buffer) {
            buffer = this.device.createBuffer({
                size: UNIFORM_BUFFER_SIZE,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this.uniformBufferPool[this.nextSlot] = buffer;
        }
        this.nextSlot++;
        this.device.queue.writeBuffer(buffer, 0, data);
        // A distinct buffer object per slot (not a shared one at varying offsets) so
        // ComputeOperation's own entriesEqual/resourceIdentity dirty-check - which only compares
        // buffer object identity, not offset - correctly rebuilds the bind group every time the
        // slot actually changes, with no changes needed to that shared base-class logic.
        this.setUniforms([{ binding: 0, resource: { buffer } }]);
    }

    public updateInputs(sourceView: GPUTextureView): void {
        this.setInputs([{ binding: 0, resource: sourceView }]);
    }

    public updateOutputs(destView: GPUTextureView, width: number, height: number): void {
        this.setOutputs([{ binding: 0, resource: destView }]);
        this.setDispatchExtent(width, height);
    }
}
