/**
 * Within each maximal run of `items` sharing the same draw-order key (per `compare`, already
 * adjacent after an ascending sort with that same comparator), regroups that run by texture
 * (first-seen texture first, stable within each texture bucket), mutating `items` in place. A run
 * of length 1 (or already-uniform texture) is left untouched. Safe by construction: within a tied
 * group, `compare` reports no preference between any two members, so their relative order is
 * unobserved by design - reordering them can never change draw-order correctness, only how many
 * instanced draw calls a run-length batching pass (e.g. SpriteResources.draw,
 * RaytracedResources.renderGBuffer) needs to express that run.
 */
export function clusterByTextureWithinTiedGroups<T extends { texture: unknown }>(items: T[], compare: (a: T, b: T) => number): void {
    let start = 0;
    while (start < items.length) {
        let end = start + 1;
        while (end < items.length && compare(items[start], items[end]) === 0) {
            end++;
        }

        if (end - start > 1) {
            const byTexture = new Map<unknown, T[]>();
            for (let i = start; i < end; i++) {
                const item = items[i];
                const bucket = byTexture.get(item.texture);
                if (bucket) {
                    bucket.push(item);
                } else {
                    byTexture.set(item.texture, [item]);
                }
            }
            let i = start;
            for (const bucket of byTexture.values()) {
                for (const item of bucket) {
                    items[i++] = item;
                }
            }
        }

        start = end;
    }
}
