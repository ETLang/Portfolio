/**
 * Handle to one entry in a PackedUniformArray. `index` is the entry's current byte-stride
 * position within the array - it moves whenever the array relocates it (markDynamic, another
 * entry's removal, or another insertStatic displacing it), so callers must always read it
 * fresh (e.g. when building an index-buffer entry) rather than caching the raw number.
 *
 * `index` is mutated only by the owning PackedUniformArray; treat it as read-only from
 * anywhere else.
 */
export class Entry {
    public index: number;

    constructor(index: number) {
        this.index = index;
    }
}

/**
 * A growable GPU storage buffer of fixed-stride entries, partitioned into a static region
 * ([0, staticCount)) and a dynamic region ([staticCount, count)). Static entries are written
 * once (insertStatic) and never repositioned; entries explicitly marked dynamic
 * (markDynamic) move to the back so the whole dynamic region stays one contiguous byte range,
 * separate from static memory. `T` is a phantom type only - it never appears in a method
 * signature - naming what shape of data a given instance holds (e.g.
 * `PackedUniformArray<mat4>`) for readability at the call site; the class itself works purely
 * in bytes via caller-supplied fill callbacks.
 *
 * Writes are staged into a CPU-side mirror and only reach the GPU on flush() - see flush()'s
 * doc comment for why.
 */
export class PackedUniformArray<T = unknown> {
    // Zero-runtime-cost phantom marker (erased - see erasableSyntaxOnly): exists only so T is
    // referenced somewhere, satisfying noUnusedLocals for a type parameter that's intentionally
    // never used in a method signature - see the class doc.
    declare private readonly _phantomType?: T;

    private readonly device: GPUDevice;
    private readonly strideBytes: number;

    private capacity: number;
    private buffer: GPUBuffer;
    private mirror: ArrayBuffer;
    private mirrorView: DataView;

    /** entries[i].index === i for every currently-occupied i < count - doubles as the position -> handle map. */
    private entries: Entry[] = [];
    private staticCount = 0;
    private count = 0;

    private dirtyMin = -1;
    private dirtyMax = -1;
    private bufferReplacedCallbacks: (() => void)[] = [];

    constructor(device: GPUDevice, strideBytes: number, initialCapacity = 16) {
        this.device = device;
        this.strideBytes = strideBytes;
        this.capacity = Math.max(1, initialCapacity);
        this.buffer = this.createBuffer(this.capacity);
        this.mirror = new ArrayBuffer(this.capacity * strideBytes);
        this.mirrorView = new DataView(this.mirror);
    }

    /** GPUBuffer to bind - its identity changes on growth; use onBufferReplaced to know when. */
    public getBuffer(): GPUBuffer {
        return this.buffer;
    }

    /** Registers a callback fired synchronously whenever growth swaps in a new GPUBuffer, so a cached bind group referencing the old one can be invalidated. */
    public onBufferReplaced(cb: () => void): void {
        this.bufferReplacedCallbacks.push(cb);
    }

    public getStaticCount(): number {
        return this.staticCount;
    }

    public getCount(): number {
        return this.count;
    }

    /**
     * Inserts a new STATIC entry at the end of the static region, staging its initial bytes,
     * and returns its stable handle. If a dynamic region exists, its first entry is displaced
     * to the very end of the array to make room - O(1) regardless of array size, since exactly
     * one existing entry (at most) is relocated.
     */
    public insertStatic(fill: (view: DataView, byteOffset: number) => void): Entry {
        this.ensureCapacity(this.count + 1);

        if (this.count === this.staticCount) {
            const entry = new Entry(this.count);
            this.entries[this.count] = entry;
            this.count++;
            this.staticCount++;
            this.writeBytes(entry.index, fill);
            return entry;
        }

        const boundaryIndex = this.staticCount;
        const displaced = this.entries[boundaryIndex];
        this.relocateTo(displaced, this.count);
        this.count++;

        const entry = new Entry(boundaryIndex);
        this.entries[boundaryIndex] = entry;
        this.staticCount++;
        this.writeBytes(entry.index, fill);
        return entry;
    }

    /**
     * Removes `entry` (static or dynamic), swap-compacting its own region so both regions stay
     * dense/contiguous. O(1): at most two entry-swaps, regardless of array size.
     */
    public remove(entry: Entry): void {
        if (entry.index < this.staticCount) {
            const lastStaticIndex = this.staticCount - 1;
            this.swap(entry, this.entries[lastStaticIndex]);
            const lastIndex = this.count - 1;
            this.swap(this.entries[lastStaticIndex], this.entries[lastIndex]);
            this.staticCount--;
            this.count--;
        } else {
            const lastIndex = this.count - 1;
            this.swap(entry, this.entries[lastIndex]);
            this.count--;
        }
    }

    /** Moves `entry` from the static region into the dynamic region. No-op if already dynamic (safe to call every frame). O(1). */
    public markDynamic(entry: Entry): void {
        if (entry.index >= this.staticCount) {
            return;
        }
        const lastStatic = this.entries[this.staticCount - 1];
        this.swap(entry, lastStatic);
        this.staticCount--;
    }

    /** Re-derives and stages exactly this entry's bytes, wherever it currently lives. Deferred - see flush(). */
    public writeEntry(entry: Entry, fill: (view: DataView, byteOffset: number) => void): void {
        this.writeBytes(entry.index, fill);
    }

    /**
     * Uploads every byte touched since the last flush (one queue.writeBuffer covering their
     * union), then clears the dirty range. No-op if nothing changed.
     *
     * Deliberately deferred rather than writing on every insertStatic/remove/markDynamic/
     * writeEntry call: the static/dynamic partition already guarantees the dynamic region is
     * one contiguous byte range, so a single flush per frame collapses "every dynamic entry
     * changed this frame" (the common case content is marked dynamic for) into exactly one
     * queue.writeBuffer call, instead of one per entry - a real cost in browsers where each
     * WebGPU call crosses a process boundary. Callers must call flush() once per frame, after
     * all mutations for that frame and before command-buffer recording.
     */
    public flush(): void {
        if (this.dirtyMin === -1) {
            return;
        }
        const data = this.mirror.slice(this.dirtyMin, this.dirtyMax);
        this.device.queue.writeBuffer(this.buffer, this.dirtyMin, data);
        this.dirtyMin = -1;
        this.dirtyMax = -1;
    }

    private writeBytes(index: number, fill: (view: DataView, byteOffset: number) => void): void {
        const byteOffset = index * this.strideBytes;
        fill(this.mirrorView, byteOffset);
        this.markDirtyRange(byteOffset, this.strideBytes);
    }

    /** Copies `entry`'s bytes to `newIndex` and updates its handle - a one-way move, not a swap. The vacated byte range is left as-is (harmless: callers of this only use it right before overwriting that range themselves). */
    private relocateTo(entry: Entry, newIndex: number): void {
        const oldOffset = entry.index * this.strideBytes;
        const newOffset = newIndex * this.strideBytes;
        new Uint8Array(this.mirror, newOffset, this.strideBytes).set(new Uint8Array(this.mirror, oldOffset, this.strideBytes));
        entry.index = newIndex;
        this.entries[newIndex] = entry;
        this.markDirtyRange(newOffset, this.strideBytes);
    }

    /** Exchanges the byte content (and handles) of two entries. No-op if they're the same entry. */
    private swap(a: Entry, b: Entry): void {
        if (a === b) {
            return;
        }
        const aIndex = a.index;
        const bIndex = b.index;
        const aOffset = aIndex * this.strideBytes;
        const bOffset = bIndex * this.strideBytes;

        const aBytes = this.mirror.slice(aOffset, aOffset + this.strideBytes);
        new Uint8Array(this.mirror, aOffset, this.strideBytes).set(new Uint8Array(this.mirror, bOffset, this.strideBytes));
        new Uint8Array(this.mirror, bOffset, this.strideBytes).set(new Uint8Array(aBytes));

        a.index = bIndex;
        b.index = aIndex;
        this.entries[aIndex] = b;
        this.entries[bIndex] = a;

        this.markDirtyRange(aOffset, this.strideBytes);
        this.markDirtyRange(bOffset, this.strideBytes);
    }

    private markDirtyRange(byteOffset: number, length: number): void {
        const end = byteOffset + length;
        this.dirtyMin = this.dirtyMin === -1 ? byteOffset : Math.min(this.dirtyMin, byteOffset);
        this.dirtyMax = this.dirtyMax === -1 ? end : Math.max(this.dirtyMax, end);
    }

    private ensureCapacity(requiredCount: number): void {
        if (requiredCount <= this.capacity) {
            return;
        }
        let newCapacity = this.capacity;
        while (newCapacity < requiredCount) {
            newCapacity *= 2;
        }

        const newMirror = new ArrayBuffer(newCapacity * this.strideBytes);
        new Uint8Array(newMirror).set(new Uint8Array(this.mirror, 0, this.count * this.strideBytes));
        this.mirror = newMirror;
        this.mirrorView = new DataView(this.mirror);
        this.capacity = newCapacity;

        this.buffer.destroy();
        this.buffer = this.createBuffer(newCapacity);
        for (const cb of this.bufferReplacedCallbacks) {
            cb();
        }

        // The new buffer starts uninitialized - everything currently in use needs re-uploading.
        this.markDirtyRange(0, this.count * this.strideBytes);
    }

    private createBuffer(capacity: number): GPUBuffer {
        return this.device.createBuffer({
            size: capacity * this.strideBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
    }
}
