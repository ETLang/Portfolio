import { describe, expect, it } from 'vitest';
import { SpriteResources } from '../sprite_resources.ts';
import { SceneGraph } from '../scene_graph.ts';
import { TextureCache } from '../texture_cache.ts';
import { SimulationResources } from '../simulation.ts';
import { createFakeGpuDevice, type FakeGpuDevice } from './test_gpu_stubs.ts';
import type { Color, Scene, SceneObject, SceneSprite } from '../scene.ts';

const WHITE: Color = { r: 1, g: 1, b: 1, a: 1 };

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

function makeSprite(ownerId: number): SceneSprite {
    return {
        ownerId,
        layer: 0,
        opacity: 1,
        image: '',
        colorMod: WHITE,
        ambient: WHITE,
        emissive: WHITE,
        simContribution: WHITE,
        simBlur: 0,
        primitiveShape: 'rect',
    };
}

function makeScene(): Scene {
    return {
        simulations: [],
        objects: [makeObject(1, 3), makeObject(2, 7)],
        cameras: [],
        raytraced: [],
        sprites: [makeSprite(1), makeSprite(2)],
        pointLights: [],
        spotlights: [],
        laserLights: [],
        directionalLights: [],
        ambientLights: [],
    };
}

async function setup(): Promise<{ device: FakeGpuDevice; spriteResources: SpriteResources; scene: Scene; sceneGraph: SceneGraph; textureCache: TextureCache }> {
    const device = createFakeGpuDevice();
    const gpuDevice = device as unknown as GPUDevice;
    const textureCache = new TextureCache(gpuDevice);
    const simulationResources = new SimulationResources(gpuDevice);
    const cameraBindGroupLayout = gpuDevice.createBindGroupLayout({ entries: [] });
    simulationResources.initialize(cameraBindGroupLayout);

    const spriteResources = new SpriteResources(gpuDevice);
    spriteResources.initialize(cameraBindGroupLayout, 'rgba16float');

    const scene = makeScene();
    const sceneGraph = new SceneGraph(scene);
    await spriteResources.updateFromScene(scene, sceneGraph, textureCache, simulationResources);

    return { device, spriteResources, scene, sceneGraph, textureCache };
}

describe('SpriteResources', () => {
    it('writes both transform and properties buffers once per sprite on updateFromScene', async () => {
        const { device } = await setup();
        expect(device.writeCalls).toHaveLength(4); // 2 sprites x (transform + properties)
    });

    it('refreshTransform rewrites only that owner\'s transform buffer with a fresh world transform', async () => {
        const { device, spriteResources, scene } = await setup();
        device.writeCalls = [];

        scene.objects[0].position.x = 99; // owner 1's object
        const freshGraph = new SceneGraph(scene);
        spriteResources.refreshTransform(1, freshGraph);

        expect(device.writeCalls).toHaveLength(1);
        expect(new Float32Array(device.writeCalls[0].data)[12]).toBe(99);
    });

    it('refreshProperties rewrites only that owner\'s properties buffer with current sprite field values', async () => {
        const { device, spriteResources, scene } = await setup();
        device.writeCalls = [];

        scene.sprites[0].opacity = 0.25; // owner 1's sprite, live reference
        spriteResources.refreshProperties(1);

        expect(device.writeCalls).toHaveLength(1);
        const floats = new Float32Array(device.writeCalls[0].data);
        expect(floats[16]).toBeCloseTo(0.25); // opacity: byte 64 -> float index 16
    });

    it('no-ops for an owner with no sprites', async () => {
        const { device, spriteResources } = await setup();
        device.writeCalls = [];

        spriteResources.refreshTransform(999, new SceneGraph(makeScene()));
        spriteResources.refreshProperties(999);

        expect(device.writeCalls).toHaveLength(0);
    });

    it('removeByOwnerIds drops the matching sprite so its refresh calls become no-ops, leaving others untouched', async () => {
        const { device, spriteResources } = await setup();

        spriteResources.removeByOwnerIds(new Set([1]));
        device.writeCalls = [];

        spriteResources.refreshTransform(1, new SceneGraph(makeScene()));
        spriteResources.refreshProperties(1);
        expect(device.writeCalls).toHaveLength(0); // owner 1 sprite is gone

        spriteResources.refreshProperties(2);
        expect(device.writeCalls).toHaveLength(1); // owner 2 sprite still present
    });

    it('removeByOwnerIds is a no-op for an unknown owner', async () => {
        const { device, spriteResources } = await setup();

        spriteResources.removeByOwnerIds(new Set([999]));
        device.writeCalls = [];

        spriteResources.refreshProperties(1);
        expect(device.writeCalls).toHaveLength(1); // owner 1 sprite still present
    });

    it('addSprite uploads and appends a new sprite, leaving existing sprites untouched', async () => {
        const { device, spriteResources, sceneGraph, textureCache } = await setup();
        device.writeCalls = [];

        const newSprite = makeSprite(3);
        await spriteResources.addSprite(newSprite, sceneGraph, textureCache);
        expect(device.writeCalls).toHaveLength(2); // transform + properties for the new sprite only

        device.writeCalls = [];
        spriteResources.refreshProperties(1);
        expect(device.writeCalls).toHaveLength(1); // owner 1's original sprite is still present
    });

    it('removeSprite removes exactly the given sprite, leaving a sibling sprite owned by the same object intact', async () => {
        const device = createFakeGpuDevice();
        const gpuDevice = device as unknown as GPUDevice;
        const textureCache = new TextureCache(gpuDevice);
        const simulationResources = new SimulationResources(gpuDevice);
        const cameraBindGroupLayout = gpuDevice.createBindGroupLayout({ entries: [] });
        simulationResources.initialize(cameraBindGroupLayout);

        const spriteResources = new SpriteResources(gpuDevice);
        spriteResources.initialize(cameraBindGroupLayout, 'rgba16float');

        const spriteA = makeSprite(1);
        const spriteB = makeSprite(1); // same owner as spriteA
        const scene: Scene = { ...makeScene(), sprites: [spriteA, spriteB] };
        const sceneGraph = new SceneGraph(scene);
        await spriteResources.updateFromScene(scene, sceneGraph, textureCache, simulationResources);
        device.writeCalls = [];

        spriteResources.removeSprite(spriteA);
        spriteResources.refreshProperties(1); // combined-owner refresh should now only touch spriteB

        expect(device.writeCalls).toHaveLength(1);
    });

    it('removeSprite is a no-op for a sprite reference it does not track', async () => {
        const { device, spriteResources } = await setup();
        device.writeCalls = [];

        spriteResources.removeSprite(makeSprite(999));
        spriteResources.refreshProperties(1);

        expect(device.writeCalls).toHaveLength(1); // owner 1 sprite untouched
    });
});
