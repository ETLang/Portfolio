import type { mat4 } from 'gl-matrix';
import { Entry, PackedUniformArray } from './packed_uniform_array.ts';
import type { SceneGraph } from './scene_graph.ts';

// Must match ObjectTransform in sprite.wgsl (and any future light/raytraced consumer).
const TRANSFORM_STRIDE_BYTES = 64; // mat4x4<f32>

interface OwnerRecord {
    entry: Entry;
    refCount: number;
}

/**
 * Shared "object transform" array: one packed mat4 entry per SceneObject that owns at least one
 * sprite/light/raytraced component, keyed by owner id. An object with e.g. both a sprite and a
 * light shares exactly one entry - and one refreshTransform/markDynamic call - between them, via
 * reference counting (ensureEntry/releaseEntry).
 */
export class TransformResources {
    private array: PackedUniformArray<mat4>;
    private owners = new Map<number, OwnerRecord>();
    private ownerRelocatedCallbacks: ((ownerId: number) => void)[] = [];

    constructor(device: GPUDevice) {
        this.array = new PackedUniformArray<mat4>(device, TRANSFORM_STRIDE_BYTES);
        // Translates the array's per-Entry relocation event (see PackedUniformArray.
        // onEntryRelocated) into the ownerId space consumers actually think in - light/raytraced/
        // sprite resources bake `ensureEntry(...).index` into their own buffers (a "which
        // transform slot" foreign key), which goes stale the moment that entry moves due to some
        // OTHER owner's markDynamic/insertStatic/remove call. Without this, that staleness is
        // silent: the transform buffer itself always stays internally consistent (see
        // PackedUniformArray's swap/relocateTo, which move the byte content along with the
        // index), so nothing errors - a draw just quietly samples a different object's transform.
        this.array.onEntryRelocated((entry) => {
            for (const [ownerId, record] of this.owners) {
                if (record.entry === entry) {
                    for (const cb of this.ownerRelocatedCallbacks) {
                        cb(ownerId);
                    }
                    return;
                }
            }
        });
    }

    public getBuffer(): GPUBuffer {
        return this.array.getBuffer();
    }

    public onBufferReplaced(cb: () => void): void {
        this.array.onBufferReplaced(cb);
    }

    /**
     * Registers a callback fired with `ownerId` whenever that owner's transform entry's `.index`
     * changes for a reason outside that owner's own control (see the constructor's comment) - a
     * consumer that baked `ensureEntry(ownerId, ...).index` into one of its own buffers must
     * re-bake it in response, or that reference silently goes stale.
     */
    public onOwnerRelocated(cb: (ownerId: number) => void): void {
        this.ownerRelocatedCallbacks.push(cb);
    }

    /** Gets (or creates) `ownerId`'s transform entry, incrementing its reference count. */
    public ensureEntry(ownerId: number, sceneGraph: SceneGraph): Entry {
        const existing = this.owners.get(ownerId);
        if (existing) {
            existing.refCount++;
            return existing.entry;
        }
        const worldTransform = sceneGraph.getWorldTransform(ownerId);
        const entry = this.array.insertStatic((view, byteOffset) => writeTransform(view, byteOffset, worldTransform));
        this.owners.set(ownerId, { entry, refCount: 1 });
        return entry;
    }

    /** Looks up `ownerId`'s current transform entry without creating one or affecting its reference count. Undefined for an unknown owner. */
    public getEntry(ownerId: number): Entry | undefined {
        return this.owners.get(ownerId)?.entry;
    }

    /** Decrements `ownerId`'s reference count, removing its entry once no component references it anymore. No-op for an unknown owner. */
    public releaseEntry(ownerId: number): void {
        const record = this.owners.get(ownerId);
        if (!record) {
            return;
        }
        record.refCount--;
        if (record.refCount <= 0) {
            this.array.remove(record.entry);
            this.owners.delete(ownerId);
        }
    }

    /** Targeted re-upload of `ownerId`'s current world transform. No-op for an unknown owner. */
    public refreshTransform(ownerId: number, sceneGraph: SceneGraph): void {
        const record = this.owners.get(ownerId);
        if (!record) {
            return;
        }
        const worldTransform = sceneGraph.getWorldTransform(ownerId);
        this.array.writeEntry(record.entry, (view, byteOffset) => writeTransform(view, byteOffset, worldTransform));
    }

    /** Moves `ownerId`'s entry into the dynamic region. No-op for an unknown owner, or if already dynamic. */
    public markDynamic(ownerId: number): void {
        const record = this.owners.get(ownerId);
        if (!record) {
            return;
        }
        this.array.markDynamic(record.entry);
    }

    public flush(): void {
        this.array.flush();
    }
}

function writeTransform(view: DataView, byteOffset: number, worldTransform: mat4): void {
    const floats = new Float32Array(view.buffer, view.byteOffset + byteOffset, 16);
    floats.set(worldTransform as Float32Array);
}
