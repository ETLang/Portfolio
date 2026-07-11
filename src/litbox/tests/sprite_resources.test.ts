import { describe, expect, it } from 'vitest';
import { compareDrawOrder, SpriteResources } from '../sprite_resources.ts';
import { SceneGraph } from '../scene_graph.ts';
import { TextureCache } from '../texture_cache.ts';
import { SimulationResources } from '../simulation.ts';
import { TransformResources } from '../transform_resources.ts';
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

function makeSprite(ownerId: number, layer = 0, sortOrder = 0): SceneSprite {
    return {
        ownerId,
        layer,
        sortOrder,
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

function makeScene(sprites?: SceneSprite[]): Scene {
    return {
        simulations: [],
        objects: [makeObject(1, 3), makeObject(2, 7)],
        cameras: [],
        raytraced: [],
        sprites: sprites ?? [makeSprite(1), makeSprite(2)],
        pointLights: [],
        spotlights: [],
        laserLights: [],
        directionalLights: [],
        ambientLights: [],
        textureAtlasKeys: [],
    };
}

interface Fixture {
    device: FakeGpuDevice;
    spriteResources: SpriteResources;
    transformResources: TransformResources;
    scene: Scene;
    sceneGraph: SceneGraph;
    textureCache: TextureCache;
}

async function setup(sprites?: SceneSprite[]): Promise<Fixture> {
    const device = createFakeGpuDevice();
    const gpuDevice = device as unknown as GPUDevice;
    const textureCache = new TextureCache(gpuDevice);
    const simulationResources = new SimulationResources(gpuDevice);
    const cameraBindGroupLayout = gpuDevice.createBindGroupLayout({ entries: [] });
    simulationResources.initialize(cameraBindGroupLayout);

    const spriteResources = new SpriteResources(gpuDevice);
    spriteResources.initialize(cameraBindGroupLayout, 'rgba16float');
    const transformResources = new TransformResources(gpuDevice);

    const scene = makeScene(sprites);
    const sceneGraph = new SceneGraph(scene);
    textureCache.loadScene('', scene.textureAtlasKeys);
    await spriteResources.updateFromScene(scene, sceneGraph, textureCache, simulationResources, transformResources);

    return { device, spriteResources, transformResources, scene, sceneGraph, textureCache };
}

function flushAll(fixture: Fixture): void {
    fixture.spriteResources.flush();
    fixture.transformResources.flush();
}

describe('SpriteResources', () => {
    it('stages properties, atlas, transform, and index data for every sprite on updateFromScene, uploaded on flush', async () => {
        const fixture = await setup();
        expect(fixture.device.writeCalls).toHaveLength(0); // nothing reaches the GPU before flush()

        flushAll(fixture);
        // One coalesced write per array touched: sprite properties, sprite atlas, sprite index, shared transforms.
        expect(fixture.device.writeCalls).toHaveLength(4);
    });

    it('refreshProperties rewrites only that sprite\'s properties entry with current field values', async () => {
        const fixture = await setup();
        flushAll(fixture);
        fixture.device.writeCalls = [];

        fixture.scene.sprites[0].opacity = 0.25; // owner 1's sprite, live reference
        fixture.spriteResources.refreshProperties(fixture.scene.sprites[0]);
        fixture.spriteResources.flush();

        expect(fixture.device.writeCalls).toHaveLength(1);
        const view = new DataView(fixture.device.writeCalls[0].data);
        expect(view.getFloat32(64, true)).toBeCloseTo(0.25); // opacity offset within SpriteProperties
    });

    it('refreshProperties no-ops for an untracked sprite', async () => {
        const fixture = await setup();
        flushAll(fixture);
        fixture.device.writeCalls = [];

        fixture.spriteResources.refreshProperties(makeSprite(999));
        fixture.spriteResources.flush();

        expect(fixture.device.writeCalls).toHaveLength(0);
    });

    it('removeByOwnerIds drops the matching sprite so a later refresh becomes a no-op, leaving others untouched', async () => {
        const fixture = await setup();
        const [spriteA, spriteB] = fixture.scene.sprites;

        fixture.spriteResources.removeByOwnerIds(new Set([1]), fixture.transformResources);
        flushAll(fixture);
        fixture.device.writeCalls = [];

        fixture.spriteResources.refreshProperties(spriteA);
        fixture.spriteResources.flush();
        expect(fixture.device.writeCalls).toHaveLength(0); // owner 1's sprite is gone

        fixture.spriteResources.refreshProperties(spriteB);
        fixture.spriteResources.flush();
        expect(fixture.device.writeCalls).toHaveLength(1); // owner 2's sprite still present
    });

    it('removeByOwnerIds is a no-op for an unknown owner', async () => {
        const fixture = await setup();
        fixture.spriteResources.removeByOwnerIds(new Set([999]), fixture.transformResources);
        flushAll(fixture);
        fixture.device.writeCalls = [];

        fixture.spriteResources.refreshProperties(fixture.scene.sprites[0]);
        fixture.spriteResources.flush();
        expect(fixture.device.writeCalls).toHaveLength(1); // owner 1's sprite still present
    });

    it('addSprite uploads and appends a new sprite, leaving existing sprites untouched', async () => {
        const fixture = await setup();
        flushAll(fixture);
        fixture.device.writeCalls = [];

        const newSprite = makeSprite(3);
        await fixture.spriteResources.addSprite(newSprite, fixture.sceneGraph, fixture.textureCache, fixture.transformResources);
        flushAll(fixture);
        expect(fixture.device.writeCalls.length).toBeGreaterThan(0); // the new sprite's data (+ its new transform entry) went out

        fixture.device.writeCalls = [];
        fixture.spriteResources.refreshProperties(fixture.scene.sprites[0]);
        fixture.spriteResources.flush();
        expect(fixture.device.writeCalls).toHaveLength(1); // owner 1's original sprite is still tracked correctly
    });

    it('removeSprite removes exactly the given sprite, leaving a sibling sprite owned by the same object intact', async () => {
        const spriteA = makeSprite(1);
        const spriteB = makeSprite(1); // same owner as spriteA
        const fixture = await setup([spriteA, spriteB]);
        flushAll(fixture);
        fixture.device.writeCalls = [];

        fixture.spriteResources.removeSprite(spriteA, fixture.transformResources);
        flushAll(fixture);
        fixture.device.writeCalls = [];

        fixture.spriteResources.refreshProperties(spriteB);
        fixture.spriteResources.flush();
        expect(fixture.device.writeCalls).toHaveLength(1);

        fixture.device.writeCalls = [];
        fixture.spriteResources.refreshProperties(spriteA);
        fixture.spriteResources.flush();
        expect(fixture.device.writeCalls).toHaveLength(0); // spriteA is gone
    });

    it('removeSprite is a no-op for a sprite reference it does not track', async () => {
        const fixture = await setup();
        fixture.spriteResources.removeSprite(makeSprite(999), fixture.transformResources);
        flushAll(fixture);
        fixture.device.writeCalls = [];

        fixture.spriteResources.refreshProperties(fixture.scene.sprites[0]);
        fixture.spriteResources.flush();
        expect(fixture.device.writeCalls).toHaveLength(1); // owner 1's sprite untouched
    });

    it('markDynamic relocates a sprite\'s properties entry without losing its data', async () => {
        const fixture = await setup();
        const [spriteA] = fixture.scene.sprites;

        fixture.spriteResources.markDynamic(spriteA);
        fixture.spriteResources.markDynamic(spriteA); // idempotent

        // Inserting a new sprite displaces the dynamic entry to the tail of its array - if
        // markDynamic wired through correctly, spriteA's properties must survive that move.
        const newSprite = makeSprite(3);
        await fixture.spriteResources.addSprite(newSprite, fixture.sceneGraph, fixture.textureCache, fixture.transformResources);
        flushAll(fixture);

        fixture.device.writeCalls = [];
        fixture.spriteResources.refreshProperties(spriteA);
        fixture.spriteResources.flush();
        const view = new DataView(fixture.device.writeCalls[0].data);
        expect(view.getFloat32(64, true)).toBe(1); // opacity survived the relocation
    });

    it('draw() visits sprites in ascending layer order regardless of scene array order', async () => {
        // Deliberately out of order in the source array.
        const back = makeSprite(1, 1, 0);
        const front = makeSprite(2, -1, 0);
        const middleFirst = makeSprite(1, 0, 5);
        const middleSecond = makeSprite(2, 0, 1);
        const fixture = await setup([back, front, middleFirst, middleSecond]);

        const seenLayers: number[] = [];
        const firstInstances: number[] = [];
        const passEncoder = {
            setPipeline: () => {},
            setVertexBuffer: () => {},
            setBindGroup: () => {},
            draw: (_vertexCount: number, _instanceCount: number, _firstVertex: number, firstInstance: number) => {
                firstInstances.push(firstInstance);
            },
        } as unknown as GPURenderPassEncoder;

        fixture.spriteResources.draw(passEncoder, (layer) => {
            seenLayers.push(layer);
            return true;
        });

        expect(seenLayers).toEqual([-1, 0, 0, 1]); // front, then the two layer-0 sprites, then back
        // firstInstance values are draw-order positions - must come out strictly ascending
        // (0,1,2,3) since draw() walks the already-sorted list in order.
        expect(firstInstances).toEqual([0, 1, 2, 3]);
    });
});

describe('compareDrawOrder', () => {
    it('orders ascending by layer first', () => {
        expect(compareDrawOrder({ layer: -1, sortOrder: 100 }, { layer: 1, sortOrder: -100 })).toBeLessThan(0);
    });

    it('breaks ties within a layer by ascending sortOrder', () => {
        expect(compareDrawOrder({ layer: 0, sortOrder: 1 }, { layer: 0, sortOrder: 5 })).toBeLessThan(0);
        expect(compareDrawOrder({ layer: 0, sortOrder: 5 }, { layer: 0, sortOrder: 1 })).toBeGreaterThan(0);
    });

    it('returns 0 for equal (layer, sortOrder) - relative order is unobserved by design', () => {
        expect(compareDrawOrder({ layer: 2, sortOrder: 3 }, { layer: 2, sortOrder: 3 })).toBe(0);
    });
});
