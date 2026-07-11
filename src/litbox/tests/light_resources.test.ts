import { describe, expect, it } from 'vitest';
import { LightResources } from '../light_resources.ts';
import { TransformResources } from '../transform_resources.ts';
import { SceneGraph } from '../scene_graph.ts';
import { createFakeGpuDevice, type FakeGpuDevice } from './test_gpu_stubs.ts';
import type { PointLight, Scene, SceneObject, Spotlight } from '../scene.ts';

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
        textureAtlasKeys: [],
    };
}

function setup(): { device: FakeGpuDevice; resources: LightResources; transformResources: TransformResources; scene: Scene; sceneGraph: SceneGraph } {
    const device = createFakeGpuDevice();
    const gpuDevice = device as unknown as GPUDevice;
    const resources = new LightResources(gpuDevice);
    const transformResources = new TransformResources(gpuDevice);
    const scene = makeScene();
    const sceneGraph = new SceneGraph(scene);
    return { device, resources, transformResources, scene, sceneGraph };
}

describe('LightResources', () => {
    it('updateFromScene stages one properties entry per light, uploaded on flush', () => {
        const { device, resources, transformResources, scene, sceneGraph } = setup();

        resources.updateFromScene(scene, sceneGraph, transformResources);
        expect(resources.getCount()).toBe(2);
        expect(device.writeCalls).toHaveLength(0); // nothing reaches the GPU before flush()

        resources.flush();
        transformResources.flush();
        expect(device.writeCalls).toHaveLength(2); // one for LightResources' array, one for the shared transform array
    });

    it('refreshProperties rewrites only that light\'s entry', () => {
        const { device, resources, transformResources, scene, sceneGraph } = setup();
        resources.updateFromScene(scene, sceneGraph, transformResources);
        resources.flush();
        transformResources.flush();
        device.writeCalls = [];

        const pointLight = scene.pointLights[0];
        resources.refreshProperties(pointLight, transformResources);
        resources.flush();

        expect(device.writeCalls).toHaveLength(1);
        const view = new DataView(device.writeCalls[0].data);
        expect(view.getFloat32(0, true)).toBe(1); // color.r
    });

    it('refreshProperties reads a spotlight\'s pinch fresh from the live reference', () => {
        const { device, resources, transformResources, scene, sceneGraph } = setup();
        resources.updateFromScene(scene, sceneGraph, transformResources);
        resources.flush();
        transformResources.flush();
        device.writeCalls = [];

        const spotlight = scene.spotlights[0] as Spotlight;
        spotlight.pinch = 0.9;
        resources.refreshProperties(spotlight, transformResources);
        resources.flush();

        const view = new DataView(device.writeCalls[0].data);
        expect(view.getFloat32(32, true)).toBeCloseTo(0.9); // pinch offset
    });

    it('addLight appends a new light without touching existing entries', () => {
        const { device, resources, transformResources, scene, sceneGraph } = setup();
        resources.updateFromScene(scene, sceneGraph, transformResources);
        resources.flush();
        transformResources.flush();
        device.writeCalls = [];

        const newLight: PointLight = { ownerId: 1, color: { r: 0, g: 0, b: 1, a: 1 }, intensity: 5, bounces: 0 };
        resources.addLight('point', newLight, sceneGraph, transformResources);
        expect(resources.getCount()).toBe(3);

        resources.flush();
        expect(device.writeCalls).toHaveLength(1); // just the new light's entry

        device.writeCalls = [];
        resources.refreshProperties(scene.pointLights[0], transformResources);
        resources.flush();
        expect(device.writeCalls).toHaveLength(1); // original light untouched by the insert
    });

    it('removeLight removes exactly one light and releases its transform reference', () => {
        const { device, resources, transformResources, scene, sceneGraph } = setup();
        resources.updateFromScene(scene, sceneGraph, transformResources);
        resources.flush();
        transformResources.flush();

        const pointLight = scene.pointLights[0];
        resources.removeLight(pointLight, transformResources);
        expect(resources.getCount()).toBe(1);
        expect(transformResources.getEntry(1)).toBeUndefined(); // owner 1's sole light was removed
        resources.flush(); // settle the compaction write left by removing a non-last entry

        device.writeCalls = [];
        resources.refreshProperties(pointLight, transformResources);
        resources.flush();
        expect(device.writeCalls).toHaveLength(0); // no longer tracked
    });

    it('removeLight is a no-op for an untracked light', () => {
        const { resources, transformResources, scene, sceneGraph } = setup();
        resources.updateFromScene(scene, sceneGraph, transformResources);

        const unknown: PointLight = { ownerId: 999, color: { r: 1, g: 1, b: 1, a: 1 }, intensity: 1, bounces: 1 };
        expect(() => resources.removeLight(unknown, transformResources)).not.toThrow();
        expect(resources.getCount()).toBe(2);
    });

    it('markDynamic relocates a light\'s entry without losing its data', () => {
        const { device, resources, transformResources, scene, sceneGraph } = setup();
        resources.updateFromScene(scene, sceneGraph, transformResources);

        const pointLight = scene.pointLights[0];
        resources.markDynamic(pointLight);
        resources.markDynamic(pointLight); // idempotent - should not throw or double-move

        // Inserting a new static light displaces the dynamic entry to the tail - if markDynamic
        // wired through correctly, pointLight's data must survive that relocation.
        const newLight: PointLight = { ownerId: 2, color: { r: 0.5, g: 0.5, b: 0.5, a: 1 }, intensity: 9, bounces: 3 };
        resources.addLight('point', newLight, sceneGraph, transformResources);
        resources.flush();

        device.writeCalls = [];
        resources.refreshProperties(pointLight, transformResources);
        resources.flush();
        const view = new DataView(device.writeCalls[0].data);
        expect(view.getFloat32(0, true)).toBe(1); // color.r survived the relocation
    });

    it('refreshProperties and markDynamic no-op for an untracked light', () => {
        const { device, resources, transformResources, scene, sceneGraph } = setup();
        resources.updateFromScene(scene, sceneGraph, transformResources);
        resources.flush();
        transformResources.flush();
        device.writeCalls = [];

        const unknown: PointLight = { ownerId: 999, color: { r: 1, g: 1, b: 1, a: 1 }, intensity: 1, bounces: 1 };
        expect(() => resources.refreshProperties(unknown, transformResources)).not.toThrow();
        expect(() => resources.markDynamic(unknown)).not.toThrow();
        resources.flush();

        expect(device.writeCalls).toHaveLength(0);
    });
});
