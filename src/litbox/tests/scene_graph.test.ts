import { describe, expect, it, vi } from 'vitest';
import { SceneGraph } from '../scene_graph.ts';
import type { Scene, SceneObject } from '../scene.ts';

function makeObject(id: number, parentId: number, x = 0): SceneObject {
    return {
        active: true,
        id,
        name: `obj${id}`,
        parentId,
        position: { x, y: 0 },
        depth: 0,
        rotation: 0,
        scale: { x: 1, y: 1 },
    };
}

function makeScene(objects: SceneObject[]): Scene {
    return {
        simulations: [],
        objects,
        cameras: [],
        raytraced: [],
        sprites: [],
        pointLights: [],
        spotlights: [],
        laserLights: [],
        directionalLights: [],
        ambientLights: [],
        textureAtlasKeys: [],
    };
}

describe('SceneGraph.getDescendantIds', () => {
    it('returns strict descendants, depth-first, excluding the id itself', () => {
        // 1 (root)
        //  - 2
        //    - 3
        //  - 4
        const scene = makeScene([
            makeObject(1, -1),
            makeObject(2, 1),
            makeObject(3, 2),
            makeObject(4, 1),
        ]);
        const graph = new SceneGraph(scene);
        expect(graph.getDescendantIds(1)).toEqual([2, 3, 4]);
        expect(graph.getDescendantIds(2)).toEqual([3]);
        expect(graph.getDescendantIds(3)).toEqual([]);
    });

    it('terminates and warns on a parentId cycle instead of looping forever', () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        // 10 <-> 11 cycle
        const scene = makeScene([
            makeObject(10, 11),
            makeObject(11, 10),
        ]);
        const graph = new SceneGraph(scene);
        expect(() => graph.getDescendantIds(10)).not.toThrow();
        expect(warn).toHaveBeenCalled();
        warn.mockRestore();
    });
});

describe('SceneGraph.invalidateSubtree', () => {
    it('clears cached world transforms for the id and its whole subtree, picking up subsequent mutations', () => {
        const scene = makeScene([
            makeObject(1, -1),
            makeObject(2, 1, 5),
        ]);
        const graph = new SceneGraph(scene);

        expect(graph.getWorldTransform(2)[12]).toBe(5);

        scene.objects[1].position.x = 50;
        // Without invalidation, the cache still returns the stale value.
        expect(graph.getWorldTransform(2)[12]).toBe(5);

        graph.invalidateSubtree(1);
        expect(graph.getWorldTransform(2)[12]).toBe(50);
    });

    it('does not clear caches outside the invalidated subtree', () => {
        const scene = makeScene([
            makeObject(1, -1),
            makeObject(2, 1),
            makeObject(4, -1, 7),
        ]);
        const graph = new SceneGraph(scene);

        expect(graph.getWorldTransform(4)[12]).toBe(7);
        scene.objects[2].position.x = 70;

        graph.invalidateSubtree(1); // unrelated subtree
        expect(graph.getWorldTransform(4)[12]).toBe(7); // still stale/cached, untouched
    });
});

describe('SceneGraph.addObject', () => {
    it('indexes a new root object without a childrenByParentId entry', () => {
        const scene = makeScene([makeObject(1, -1)]);
        const graph = new SceneGraph(scene);

        const obj = makeObject(2, -1);
        graph.addObject(obj);

        expect(graph.getObject(2)).toBe(obj);
        expect(graph.getDescendantIds(2)).toEqual([]);
    });

    it('indexes a new non-root object as a child of its parent', () => {
        const scene = makeScene([makeObject(1, -1)]);
        const graph = new SceneGraph(scene);

        const child = makeObject(2, 1);
        graph.addObject(child);

        expect(graph.getDescendantIds(1)).toEqual([2]);
    });
});

describe('SceneGraph.removeObject', () => {
    it('removes a leaf and returns just its own id', () => {
        const scene = makeScene([makeObject(1, -1), makeObject(2, 1)]);
        const graph = new SceneGraph(scene);

        expect(graph.removeObject(2)).toEqual([2]);
        expect(graph.getObject(2)).toBeUndefined();
        expect(graph.getDescendantIds(1)).toEqual([]);
    });

    it('cascades to the whole subtree, root id first', () => {
        const scene = makeScene([
            makeObject(1, -1),
            makeObject(2, 1),
            makeObject(3, 2),
            makeObject(4, 1),
        ]);
        const graph = new SceneGraph(scene);

        expect(graph.removeObject(1)).toEqual([1, 2, 3, 4]);
        expect(graph.getObject(1)).toBeUndefined();
        expect(graph.getObject(2)).toBeUndefined();
        expect(graph.getObject(3)).toBeUndefined();
        expect(graph.getObject(4)).toBeUndefined();
    });

    it('drops cached world transform and active-in-hierarchy state for the removed subtree', () => {
        const scene = makeScene([makeObject(1, -1), makeObject(2, 1, 5)]);
        const graph = new SceneGraph(scene);
        graph.getWorldTransform(2);
        graph.isActiveInHierarchy(2);

        graph.removeObject(2);

        // Re-adding an object with the same id must not resurrect stale cached state.
        graph.addObject(makeObject(2, 1, 99));
        expect(graph.getWorldTransform(2)[12]).toBe(99);
    });

    it('updates the former parent\'s children list', () => {
        const scene = makeScene([makeObject(1, -1), makeObject(2, 1), makeObject(3, 1)]);
        const graph = new SceneGraph(scene);

        graph.removeObject(2);

        expect(graph.getDescendantIds(1)).toEqual([3]);
    });

    it('is a no-op returning [] for an unknown id', () => {
        const scene = makeScene([makeObject(1, -1)]);
        const graph = new SceneGraph(scene);

        expect(graph.removeObject(999)).toEqual([]);
    });
});

describe('SceneGraph.setParent', () => {
    it('moves an object to a new parent, updating both children lists', () => {
        const scene = makeScene([
            makeObject(1, -1),
            makeObject(2, -1),
            makeObject(3, 1),
        ]);
        const graph = new SceneGraph(scene);

        graph.setParent(3, 2);

        expect(graph.getDescendantIds(1)).toEqual([]);
        expect(graph.getDescendantIds(2)).toEqual([3]);
        expect(graph.getObject(3)!.parentId).toBe(2);
    });

    it('moves an object to the scene root via parentId -1', () => {
        const scene = makeScene([makeObject(1, -1), makeObject(2, 1)]);
        const graph = new SceneGraph(scene);

        graph.setParent(2, -1);

        expect(graph.getDescendantIds(1)).toEqual([]);
        expect(graph.getObject(2)!.parentId).toBe(-1);
    });

    it('invalidates the cached world transform of the reparented subtree', () => {
        const scene = makeScene([
            makeObject(1, -1, 100),
            makeObject(2, -1, 0),
            makeObject(3, 1, 5),
        ]);
        const graph = new SceneGraph(scene);
        expect(graph.getWorldTransform(3)[12]).toBe(105); // under object 1 (x=100)

        graph.setParent(3, 2); // object 2 sits at x=0

        expect(graph.getWorldTransform(3)[12]).toBe(5);
    });

    it('throws when reparenting an object to itself', () => {
        const scene = makeScene([makeObject(1, -1)]);
        const graph = new SceneGraph(scene);

        expect(() => graph.setParent(1, 1)).toThrow(/itself/);
    });

    it('throws when reparenting an object to its own descendant', () => {
        const scene = makeScene([makeObject(1, -1), makeObject(2, 1)]);
        const graph = new SceneGraph(scene);

        expect(() => graph.setParent(1, 2)).toThrow(/descendant/);
    });

    it('throws for an unknown id or unknown new parent', () => {
        const scene = makeScene([makeObject(1, -1)]);
        const graph = new SceneGraph(scene);

        expect(() => graph.setParent(999, 1)).toThrow();
        expect(() => graph.setParent(1, 999)).toThrow();
    });
});
