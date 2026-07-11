import { describe, expect, it } from 'vitest';
import { TransformResources } from '../transform_resources.ts';
import { SceneGraph } from '../scene_graph.ts';
import { createFakeGpuDevice } from './test_gpu_stubs.ts';
import type { Scene, SceneObject } from '../scene.ts';

function makeObject(id: number, x: number): SceneObject {
    return {
        active: true,
        id,
        name: `obj${id}`,
        parentId: -1,
        position: { x, y: 0 },
        depth: 0,
        rotation: 0,
        scale: { x: 1, y: 1 },
    };
}

function makeScene(): Scene {
    return {
        simulations: [],
        objects: [makeObject(1, 3), makeObject(2, 7)],
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

describe('TransformResources', () => {
    it('ensureEntry creates one entry per distinct owner and stages its world transform', () => {
        const device = createFakeGpuDevice();
        const resources = new TransformResources(device as unknown as GPUDevice);
        const scene = makeScene();
        const sceneGraph = new SceneGraph(scene);

        const entry1 = resources.ensureEntry(1, sceneGraph);
        const entry2 = resources.ensureEntry(2, sceneGraph);
        resources.flush();

        expect(entry1.index).not.toBe(entry2.index);
        expect(device.writeCalls).toHaveLength(1); // one coalesced flush covering both new entries
    });

    it('ensureEntry called again for the same owner returns the same entry and increments refcount (does not insert a second entry)', () => {
        const device = createFakeGpuDevice();
        const resources = new TransformResources(device as unknown as GPUDevice);
        const scene = makeScene();
        const sceneGraph = new SceneGraph(scene);

        const first = resources.ensureEntry(1, sceneGraph);
        const second = resources.ensureEntry(1, sceneGraph); // e.g. a sprite and a light on the same owner

        expect(second).toBe(first);
    });

    it('releaseEntry only removes the entry once every reference has been released', () => {
        const device = createFakeGpuDevice();
        const resources = new TransformResources(device as unknown as GPUDevice);
        const scene = makeScene();
        const sceneGraph = new SceneGraph(scene);

        resources.ensureEntry(1, sceneGraph);
        resources.ensureEntry(1, sceneGraph); // refCount 2
        resources.releaseEntry(1); // refCount 1 - still referenced
        resources.flush();
        device.writeCalls = [];

        resources.refreshTransform(1, sceneGraph);
        resources.flush();
        expect(device.writeCalls).toHaveLength(1); // still tracked

        resources.releaseEntry(1); // refCount 0 - now removed
        device.writeCalls = [];
        resources.refreshTransform(1, sceneGraph);
        resources.flush();
        expect(device.writeCalls).toHaveLength(0); // no longer tracked
    });

    it('refreshTransform and markDynamic no-op for an unknown owner', () => {
        const device = createFakeGpuDevice();
        const resources = new TransformResources(device as unknown as GPUDevice);
        const sceneGraph = new SceneGraph(makeScene());

        expect(() => resources.refreshTransform(999, sceneGraph)).not.toThrow();
        expect(() => resources.markDynamic(999)).not.toThrow();
        resources.flush();
        expect(device.writeCalls).toHaveLength(0);
    });

    it('refreshTransform stages the current world transform, uploaded on the next flush', () => {
        const device = createFakeGpuDevice();
        const resources = new TransformResources(device as unknown as GPUDevice);
        const scene = makeScene();
        resources.ensureEntry(1, new SceneGraph(scene));
        resources.flush();
        device.writeCalls = [];

        scene.objects[0].position.x = 42;
        resources.refreshTransform(1, new SceneGraph(scene));
        resources.flush();

        expect(device.writeCalls).toHaveLength(1);
        expect(new Float32Array(device.writeCalls[0].data)[12]).toBe(42); // translation.x
    });
});
