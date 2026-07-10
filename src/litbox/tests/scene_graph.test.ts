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
