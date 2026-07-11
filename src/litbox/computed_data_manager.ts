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
 * acquire() with a matching descriptor. Only purge() and the idle sweep (see purgeStale())
 * actually call destroy(). Callers must treat a released ComputedTexture/ComputedBuffer (and
 * any view obtained from it) as invalid immediately after releasing it - null out any
 * reference you're holding.
 *
 * Released resources that sit unused for more than maxIdleMs (5 seconds by default) are
 * destroyed automatically. This sweep piggybacks on acquire/release calls rather than running
 * on a timer, so it only actually runs (and only at most once per second) when the manager is
 * in active use - a manager that goes quiet simply stops sweeping until activity resumes.
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

/** Resources released and left unused for longer than this are destroyed by the idle sweep. */
const DEFAULT_MAX_IDLE_MS = 5000;

/** The idle sweep is opportunistic (piggybacked on acquire/release calls), so throttle it to run at most this often rather than on every call. */
const MIN_SWEEP_INTERVAL_MS = 1000;

/** A pooled resource plus the timestamp it was released at, used by the idle sweep to find resources that have sat unused too long. */
interface PoolEntry<T> {
    resource: T;
    releasedAt: number;
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
    private texturePool = new Map<string, PoolEntry<ComputedTexture>[]>();
    private bufferPool = new Map<string, PoolEntry<ComputedBuffer>[]>();
    private maxIdleMs: number;
    private now: () => number;
    private lastSweepAt = 0;

    /** `now` defaults to performance.now() and is only overridden in tests, to drive the idle sweep without relying on real elapsed time. */
    constructor(device: GPUDevice, maxIdleMs = DEFAULT_MAX_IDLE_MS, now: () => number = () => performance.now()) {
        this.device = device;
        this.maxIdleMs = maxIdleMs;
        this.now = now;
    }

    /** Acquires a pooled 2D texture matching this exact descriptor, reusing a previously-released one if one is available. */
    public acquireTexture(width: number, height: number, format: GPUTextureFormat, usage: GPUTextureUsageFlags, mipLevelCount = 1): ComputedTexture {
        this.sweepIfDue(this.now());

        const key = textureKey(width, height, format, usage, mipLevelCount);
        const reused = this.texturePool.get(key)?.pop()?.resource;
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
        const now = this.now();
        this.sweepIfDue(now);

        const key = textureKey(pooled.width, pooled.height, pooled.format, pooled.usage, pooled.mipLevelCount);
        let pool = this.texturePool.get(key);
        if (!pool) {
            pool = [];
            this.texturePool.set(key, pool);
        }
        pool.push({ resource: pooled, releasedAt: now });
    }

    /** Acquires a pooled buffer of at least `size` bytes with the given usage flags, reusing a previously-released one if one is available. Its contents are whatever was last written to it - callers needing a specific initial value should use acquireBufferWithData or write it themselves. */
    public acquireBuffer(size: number, usage: GPUBufferUsageFlags): ComputedBuffer {
        this.sweepIfDue(this.now());

        const alignedSize = align4(size);
        const key = bufferKey(alignedSize, usage);
        const reused = this.bufferPool.get(key)?.pop()?.resource;
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
        const now = this.now();
        this.sweepIfDue(now);

        const key = bufferKey(pooled.size, pooled.usage);
        let pool = this.bufferPool.get(key);
        if (!pool) {
            pool = [];
            this.bufferPool.set(key, pool);
        }
        pool.push({ resource: pooled, releasedAt: now });
    }

    /** Runs purgeStale() if it hasn't run in the last MIN_SWEEP_INTERVAL_MS - called opportunistically from acquire/release so idle resources get reclaimed without a dedicated timer. */
    private sweepIfDue(now: number): void {
        if (now - this.lastSweepAt < MIN_SWEEP_INTERVAL_MS) {
            return;
        }
        this.lastSweepAt = now;
        this.purgeStale(now);
    }

    /** Destroys and evicts every pooled resource that has been released for longer than maxIdleMs (defaults to the value passed to the constructor). Normally triggered automatically; exposed so callers can force an immediate sweep, e.g. in tests. */
    public purgeStale(now: number = this.now(), maxIdleMs: number = this.maxIdleMs): void {
        for (const pool of this.texturePool.values()) {
            this.evictStale(pool, now, maxIdleMs, (entry) => entry.resource.texture.destroy());
        }
        for (const pool of this.bufferPool.values()) {
            this.evictStale(pool, now, maxIdleMs, (entry) => entry.resource.buffer.destroy());
        }
    }

    private evictStale<T>(pool: PoolEntry<T>[], now: number, maxIdleMs: number, destroy: (entry: PoolEntry<T>) => void): void {
        let writeIndex = 0;
        for (let readIndex = 0; readIndex < pool.length; readIndex++) {
            const entry = pool[readIndex];
            if (now - entry.releasedAt > maxIdleMs) {
                destroy(entry);
            } else {
                pool[writeIndex++] = entry;
            }
        }
        pool.length = writeIndex;
    }

    /** Destroys every pooled (released, not currently acquired) resource and empties the pools - e.g. on device loss or teardown. */
    public purge(): void {
        for (const pool of this.texturePool.values()) {
            for (const entry of pool) {
                entry.resource.texture.destroy();
            }
        }
        this.texturePool.clear();

        for (const pool of this.bufferPool.values()) {
            for (const entry of pool) {
                entry.resource.buffer.destroy();
            }
        }
        this.bufferPool.clear();
    }
}
