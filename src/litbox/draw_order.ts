/**
 * Within each maximal run of `items` sharing the same draw-order key (per `compare`, already
 * adjacent after an ascending sort with that same comparator), regroups that run by `key`
 * (first-seen key first, stable within each key's bucket), mutating `items` in place. A run of
 * length 1 (or already-uniform key) is left untouched. Safe by construction: within a tied group,
 * `compare` reports no preference between any two members, so their relative order is unobserved
 * by design - reordering them can never change draw-order correctness, only how many instanced/
 * ranged draw calls a run-length batching pass (e.g. SpriteResources.draw,
 * RaytracedResources.renderGBuffer) needs to express that run.
 *
 * `key` defaults to each item's `texture` (bucket equality follows Map's SameValueZero semantics -
 * reference equality for objects like GPUTexture). Pass a custom `key` to bucket by more than
 * texture alone - e.g. RaytracedResources combines texture with primitiveShapeId, since objects
 * with different shapes now draw from different mesh vertex ranges and can't share a batched draw
 * call even when their texture matches.
 */
export function clusterByTextureWithinTiedGroups<T extends { texture: unknown }>(
    items: T[],
    compare: (a: T, b: T) => number,
    key: (item: T) => unknown = item => item.texture,
): void {
    let start = 0;
    while (start < items.length) {
        let end = start + 1;
        while (end < items.length && compare(items[start], items[end]) === 0) {
            end++;
        }

        if (end - start > 1) {
            const byKey = new Map<unknown, T[]>();
            for (let i = start; i < end; i++) {
                const item = items[i];
                const k = key(item);
                const bucket = byKey.get(k);
                if (bucket) {
                    bucket.push(item);
                } else {
                    byKey.set(k, [item]);
                }
            }
            let i = start;
            for (const bucket of byKey.values()) {
                for (const item of bucket) {
                    items[i++] = item;
                }
            }
        }

        start = end;
    }
}
