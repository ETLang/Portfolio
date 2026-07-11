export type DynamicFlag = 'dynamic' | 'dirty';

/**
 * Tracks per-entry dynamic/dirty status, keyed by the live struct reference
 * itself (not by name/path/index - those are resolved once, at the call site
 * that marks an entry, not on every frame). 'dynamic' persists every frame;
 * 'dirty' behaves like 'dynamic' for exactly one frame, then reverts to the
 * implicit default of 'static' (i.e. absence from this map).
 */
export class DynamicSet<T> {
    private flags = new Map<T, DynamicFlag>();

    public markDynamic(item: T): void {
        this.flags.set(item, 'dynamic');
    }

    public markDirty(item: T): void {
        if (this.flags.get(item) !== 'dynamic') {
            this.flags.set(item, 'dirty');
        }
    }

    /** Entries needing an update this frame (dynamic ∪ dirty). Snapshot, not a live view. */
    public activeThisFrame(): T[] {
        return [...this.flags.keys()];
    }

    /** Entries with the persistent 'dynamic' flag only (excludes one-shot 'dirty'). Snapshot, not a live view. */
    public dynamicOnly(): T[] {
        const result: T[] = [];
        for (const [item, flag] of this.flags) {
            if (flag === 'dynamic') {
                result.push(item);
            }
        }
        return result;
    }

    /** Removes an entry entirely, regardless of its current flag. For entries that no longer exist (e.g. a destroyed object). */
    public delete(item: T): void {
        this.flags.delete(item);
    }

    /** Drops 'dirty' entries; leaves 'dynamic' entries untouched. Call once per frame, after processing. */
    public clearDirty(): void {
        for (const [item, flag] of this.flags) {
            if (flag === 'dirty') {
                this.flags.delete(item);
            }
        }
    }
}
