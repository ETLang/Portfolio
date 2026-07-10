// Minimal WebGPU stand-ins for unit-testing GPU resource managers' own JS logic
// (buffer sizing, offsets, per-owner indexing, live-reference reads) in Node under
// Vitest, without a real WebGPU context. None of this simulates actual GPU behavior -
// it just records what was asked for so tests can assert on it.
//
// @webgpu/types (this project's devDependency) only supplies ambient TypeScript
// declarations, not runtime globals - GPUBufferUsage/GPUShaderStage/GPUTextureUsage
// are real values in a browser but don't exist in Node, so they must be stubbed
// before any code that references them (inside function bodies, not at module load
// time) actually runs.

export function installGpuGlobalStubs(): void {
    const globalAny = globalThis as unknown as Record<string, unknown>;
    globalAny.GPUBufferUsage ??= {
        MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
        INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
        INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
    };
    globalAny.GPUShaderStage ??= { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 };
    globalAny.GPUTextureUsage ??= {
        COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04,
        STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10,
    };
}

export interface WriteBufferCall {
    buffer: FakeGpuBuffer;
    bufferOffset: number;
    data: ArrayBuffer;
}

export class FakeGpuBuffer {
    public size: number;
    public destroyed = false;
    private mapped: ArrayBuffer | null = null;

    constructor(size: number) {
        this.size = size;
    }

    public getMappedRange(): ArrayBuffer {
        this.mapped = new ArrayBuffer(this.size);
        return this.mapped;
    }

    public unmap(): void {
        this.mapped = null;
    }

    public destroy(): void {
        this.destroyed = true;
    }
}

class FakeGpuTexture {
    public createView(): object {
        return {};
    }

    public destroy(): void {}
}

/** Records every queue.writeBuffer call so tests can assert target buffer/offset/bytes. */
export class FakeGpuDevice {
    public writeCalls: WriteBufferCall[] = [];

    public queue = {
        writeBuffer: (buffer: FakeGpuBuffer, bufferOffset: number, data: BufferSource): void => {
            const arrayBuffer = data instanceof ArrayBuffer
                ? data.slice(0)
                : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
            this.writeCalls.push({ buffer, bufferOffset, data: arrayBuffer });
        },
        writeTexture: (): void => {},
        copyExternalImageToTexture: (): void => {},
        submit: (): void => {},
    };

    public createBuffer(descriptor: { size: number }): FakeGpuBuffer {
        return new FakeGpuBuffer(descriptor.size);
    }

    public createTexture(): FakeGpuTexture {
        return new FakeGpuTexture();
    }

    public createSampler(): object {
        return {};
    }

    public createBindGroupLayout(): object {
        return {};
    }

    public createBindGroup(): object {
        return {};
    }

    public createShaderModule(): object {
        return {};
    }

    public createRenderPipeline(): object {
        return {};
    }

    public createPipelineLayout(): object {
        return {};
    }
}

export function createFakeGpuDevice(): FakeGpuDevice {
    installGpuGlobalStubs();
    return new FakeGpuDevice();
}
