import { describe, expect, it } from 'vitest';
import { LightResources } from '../light_resources.ts';
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
        pointLights: [{ ownerId: 1, color: { r: 1, g: 0, b: 0, a: 1 }, intensity: 2, bounces: 1 }],
        spotlights: [{ ownerId: 2, color: { r: 0, g: 1, b: 0, a: 1 }, intensity: 3, pinch: 0.25, bounces: 2 }],
        laserLights: [],
        directionalLights: [],
        ambientLights: [],
    };
}

describe('LightResources', () => {
    it('writes the full transform+properties buffers once on updateFromScene', () => {
        const device = createFakeGpuDevice();
        const resources = new LightResources(device as unknown as GPUDevice);
        const scene = makeScene();
        const sceneGraph = new SceneGraph(scene);

        resources.updateFromScene(scene, sceneGraph);

        expect(resources.getCount()).toBe(2);
        expect(device.writeCalls).toHaveLength(2);
        expect(device.writeCalls.map(c => c.buffer)).toEqual([
            resources.getTransformBuffer(),
            resources.getPropertiesBuffer(),
        ]);
    });

    it('refreshTransform rewrites only that owner\'s transform slice, not properties', () => {
        const device = createFakeGpuDevice();
        const resources = new LightResources(device as unknown as GPUDevice);
        const scene = makeScene();
        const sceneGraph = new SceneGraph(scene);
        resources.updateFromScene(scene, sceneGraph);
        device.writeCalls = [];

        resources.refreshTransform(2, sceneGraph); // owner 2 -> spotlight, flat index 1

        expect(device.writeCalls).toHaveLength(1);
        const call = device.writeCalls[0];
        expect(call.buffer).toBe(resources.getTransformBuffer());
        expect(call.bufferOffset).toBe(1 * 32); // index 1 * LIGHT_TRANSFORM_STRIDE_BYTES
        expect(new Float32Array(call.data)[0]).toBe(7); // owner 2's worldPosition.x
    });

    it('refreshProperties rewrites only that owner\'s properties slice, not transform', () => {
        const device = createFakeGpuDevice();
        const resources = new LightResources(device as unknown as GPUDevice);
        const scene = makeScene();
        const sceneGraph = new SceneGraph(scene);
        resources.updateFromScene(scene, sceneGraph);
        device.writeCalls = [];

        resources.refreshProperties(1); // owner 1 -> point light, flat index 0

        expect(device.writeCalls).toHaveLength(1);
        const call = device.writeCalls[0];
        expect(call.buffer).toBe(resources.getPropertiesBuffer());
        expect(call.bufferOffset).toBe(0);
        expect(new Float32Array(call.data)[0]).toBe(1); // color.r
    });

    it('no-ops for an owner with no lights', () => {
        const device = createFakeGpuDevice();
        const resources = new LightResources(device as unknown as GPUDevice);
        const scene = makeScene();
        const sceneGraph = new SceneGraph(scene);
        resources.updateFromScene(scene, sceneGraph);
        device.writeCalls = [];

        resources.refreshTransform(999, sceneGraph);
        resources.refreshProperties(999);

        expect(device.writeCalls).toHaveLength(0);
    });
});
