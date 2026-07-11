/**
 * Pools GPUTextures and GPUBuffers by descriptor so equivalently-shaped resources can be
 * reused across frames/owners instead of repeatedly created and destroyed. Each pooled
 * resource is wrapped (ComputedTexture/ComputedBuffer) together with the low-level companion
 * objects every consumer needs alongside it - a GPUTextureView for textures - so callers
 * don't have to separately track "the view for this texture" the way SimulationResources
 * currently does with its sibling lightmapTexture/lightmapView fields.
 *
 * Named for the narrow role this manager plays: pooling compute-adjacent scratch resources,
 * not general-purpose graphics texture/buffer management (TextureCache and the various
 * *Resources classes own their resources directly and aren't expected to route through this).
 *
 * Release does not destroy the underlying GPU object - it's handed back for a future
 * acquire() with a matching descriptor. Only purge() actually calls destroy(). Callers must
 * treat a released ComputedTexture/ComputedBuffer (and any view obtained from it) as invalid
 * immediately after releasing it - null out any reference you're holding.
 */

function textureKey(width: number, height: number, format: GPUTextureFormat, usage: GPUTextureUsageFlags, mipLevelCount: number): string {
    return `${width}x${height}x${format}x${usage}x${mipLevelCount}`;
}

function bufferKey(size: number, usage: GPUBufferUsageFlags): string {
    return `${size}x${usage}`;
}

/** WebGPU requires certain buffer operations' sizes to be a multiple of 4 bytes; round up defensively for every pooled buffer. */
function align4(size: number): number {
    return Math.ceil(size / 4) * 4;
}

/** A pooled 2D GPUTexture plus its default full view and any per-mip views taken from it. */
export class ComputedTexture {
    public readonly texture: GPUTexture;
    public readonly view: GPUTextureView;
    public readonly width: number;
    public readonly height: number;
    public readonly format: GPUTextureFormat;
    public readonly usage: GPUTextureUsageFlags;
    public readonly mipLevelCount: number;

    private mipViews = new Map<number, GPUTextureView>();

    constructor(texture: GPUTexture, width: number, height: number, format: GPUTextureFormat, usage: GPUTextureUsageFlags, mipLevelCount: number) {
        this.texture = texture;
        this.width = width;
        this.height = height;
        this.format = format;
        this.usage = usage;
        this.mipLevelCount = mipLevelCount;
        this.view = texture.createView();
    }

    /** Lazily creates and caches a single-mip view of this texture - e.g. for rendering into one mip of a mipmapped render target. */
    public getMipView(level: number): GPUTextureView {
        let view = this.mipViews.get(level);
        if (!view) {
            view = this.texture.createView({ baseMipLevel: level, mipLevelCount: 1 });
            this.mipViews.set(level, view);
        }
        return view;
    }
}

/** A pooled GPUBuffer. */
export class ComputedBuffer {
    public readonly buffer: GPUBuffer;
    public readonly size: number;
    public readonly usage: GPUBufferUsageFlags;

    constructor(buffer: GPUBuffer, size: number, usage: GPUBufferUsageFlags) {
        this.buffer = buffer;
        this.size = size;
        this.usage = usage;
    }
}

export class ComputedDataManager {
    private device: GPUDevice;
    private texturePool = new Map<string, ComputedTexture[]>();
    private bufferPool = new Map<string, ComputedBuffer[]>();

    constructor(device: GPUDevice) {
        this.device = device;
    }

    /** Acquires a pooled 2D texture matching this exact descriptor, reusing a previously-released one if one is available. */
    public acquireTexture(width: number, height: number, format: GPUTextureFormat, usage: GPUTextureUsageFlags, mipLevelCount = 1): ComputedTexture {
        const key = textureKey(width, height, format, usage, mipLevelCount);
        const reused = this.texturePool.get(key)?.pop();
        if (reused) {
            return reused;
        }

        const texture = this.device.createTexture({ size: [width, height], format, usage, mipLevelCount });
        return new ComputedTexture(texture, width, height, format, usage, mipLevelCount);
    }

    /** Acquires a texture with the same width/height/mip count as `match`, optionally in a different format. */
    public acquireTextureLike(match: ComputedTexture, format: GPUTextureFormat = match.format, usage: GPUTextureUsageFlags = match.usage): ComputedTexture {
        return this.acquireTexture(match.width, match.height, format, usage, match.mipLevelCount);
    }

    /** Returns `pooled` to the pool for a future acquireTexture() with a matching descriptor. Do not use `pooled` (or any view taken from it) again afterward. */
    public releaseTexture(pooled: ComputedTexture): void {
        const key = textureKey(pooled.width, pooled.height, pooled.format, pooled.usage, pooled.mipLevelCount);
        let pool = this.texturePool.get(key);
        if (!pool) {
            pool = [];
            this.texturePool.set(key, pool);
        }
        pool.push(pooled);
    }

    /** Acquires a pooled buffer of at least `size` bytes with the given usage flags, reusing a previously-released one if one is available. Its contents are whatever was last written to it - callers needing a specific initial value should use acquireBufferWithData or write it themselves. */
    public acquireBuffer(size: number, usage: GPUBufferUsageFlags): ComputedBuffer {
        const alignedSize = align4(size);
        const key = bufferKey(alignedSize, usage);
        const reused = this.bufferPool.get(key)?.pop();
        if (reused) {
            return reused;
        }

        const buffer = this.device.createBuffer({ size: alignedSize, usage });
        return new ComputedBuffer(buffer, alignedSize, usage);
    }

    /** Acquires a pooled buffer sized to `data` and immediately uploads it. */
    public acquireBufferWithData(data: BufferSource, usage: GPUBufferUsageFlags): ComputedBuffer {
        const pooled = this.acquireBuffer(data.byteLength, usage);
        this.device.queue.writeBuffer(pooled.buffer, 0, data);
        return pooled;
    }

    /** Returns `pooled` to the pool for a future acquireBuffer()/acquireBufferWithData() with a matching size/usage. Do not use `pooled` again afterward. */
    public releaseBuffer(pooled: ComputedBuffer): void {
        const key = bufferKey(pooled.size, pooled.usage);
        let pool = this.bufferPool.get(key);
        if (!pool) {
            pool = [];
            this.bufferPool.set(key, pool);
        }
        pool.push(pooled);
    }

    /** Destroys every pooled (released, not currently acquired) resource and empties the pools - e.g. on device loss or teardown. */
    public purge(): void {
        for (const pool of this.texturePool.values()) {
            for (const pooled of pool) {
                pooled.texture.destroy();
            }
        }
        this.texturePool.clear();

        for (const pool of this.bufferPool.values()) {
            for (const pooled of pool) {
                pooled.buffer.destroy();
            }
        }
        this.bufferPool.clear();
    }
}
