/**
 * A uniform buffer ring backed by a single GPUBuffer sized N * frameCount,
 * where N is the (alignment-padded) per-frame stride. Binding uses one
 * GPUBindGroup with a dynamic offset - write() targets the current frame's
 * slot, and callers pass getCurrentOffset() to setBindGroup() to read it
 * back, so writing this frame's data can't race the GPU still reading an
 * earlier frame's. Call write() once per frame before drawing, then
 * advance() once per frame after submission.
 */
export class RingBufferedUniform {
    private device: GPUDevice;
    private buffer: GPUBuffer;
    private bindGroup: GPUBindGroup;
    private frameCount: number;
    private frameIndex = 0;
    private strideBytes: number;

    constructor(device: GPUDevice, bindGroupLayout: GPUBindGroupLayout, sizeBytes: number, frameCount = 3) {
        this.device = device;
        this.frameCount = frameCount;

        const alignment = device.limits.minUniformBufferOffsetAlignment;
        this.strideBytes = Math.ceil(sizeBytes / alignment) * alignment;

        this.buffer = device.createBuffer({
            size: this.strideBytes * frameCount,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.buffer, size: sizeBytes } }],
        });
    }

    /** Writes to the current frame's slot. */
    public write(data: BufferSource): void {
        this.device.queue.writeBuffer(this.buffer, this.frameIndex * this.strideBytes, data);
    }

    /** The single bind group backing every slot; bind together with getCurrentOffset(). */
    public getBindGroup(): GPUBindGroup {
        return this.bindGroup;
    }

    /** Byte offset of the current frame's slot; pass as the dynamic offset when binding. */
    public getCurrentOffset(): number {
        return this.frameIndex * this.strideBytes;
    }

    /** Advances to the next slot in the ring. Call once per frame, after submission. */
    public advance(): void {
        this.frameIndex = (this.frameIndex + 1) % this.frameCount;
    }
}
