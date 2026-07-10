import { describe, expect, it } from 'vitest';
import { DynamicSet } from '../dynamic_set.ts';

describe('DynamicSet', () => {
    it('starts empty (implicit static default)', () => {
        const set = new DynamicSet<string>();
        expect(set.activeThisFrame()).toEqual([]);
    });

    it('markDynamic makes an entry active every frame', () => {
        const set = new DynamicSet<string>();
        set.markDynamic('a');
        expect(set.activeThisFrame()).toEqual(['a']);
        set.clearDirty();
        expect(set.activeThisFrame()).toEqual(['a']);
    });

    it('markDirty makes an entry active for exactly one frame, then reverts to static', () => {
        const set = new DynamicSet<string>();
        set.markDirty('a');
        expect(set.activeThisFrame()).toEqual(['a']);
        set.clearDirty();
        expect(set.activeThisFrame()).toEqual([]);
    });

    it('markDirty on an already-dynamic entry is a no-op (stays dynamic, not downgraded)', () => {
        const set = new DynamicSet<string>();
        set.markDynamic('a');
        set.markDirty('a');
        set.clearDirty();
        expect(set.activeThisFrame()).toEqual(['a']);
    });

    it('clearDirty only drops dirty entries, leaving dynamic entries untouched', () => {
        const set = new DynamicSet<string>();
        set.markDynamic('a');
        set.markDirty('b');
        set.clearDirty();
        expect(set.activeThisFrame()).toEqual(['a']);
    });

    it('uses object identity as the key, not structural equality', () => {
        const set = new DynamicSet<{ id: number }>();
        const a = { id: 1 };
        const b = { id: 1 };
        set.markDynamic(a);
        expect(set.activeThisFrame()).toEqual([a]);
        expect(set.activeThisFrame()).not.toContain(b);
    });
});
