import { describe, expect, it } from 'vitest';
import { compareRaytracedDrawOrder, RaytracedResources } from '../raytraced_resources.ts';
import { SceneGraph } from '../scene_graph.ts';
import { TextureCache } from '../texture_cache.ts';
import { SimulationResources } from '../simulation.ts';
import { TransformResources } from '../transform_resources.ts';
import { ComputedDataManager } from '../computed_data_manager.ts';
import { LutResources } from '../lut_resources.ts';
import { createFakeGpuDevice, type FakeGpuDevice } from './test_gpu_stubs.ts';
import type { Color, RaytracedObject, Scene, SceneObject } from '../scene.ts';

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

function makeRaytraced(ownerId: number, sortOrder = 0): RaytracedObject {
    return {
        ownerId,
        sortOrder,
        logDensity: 0,
        roughness: 0,
        heightScale: 1,
        albedo: WHITE,
        albedoMap: '',
        logDensityMap: '',
        sdfNormalMap: '',
        primitiveShape: 'rect',
    };
}

function makeScene(raytraced?: RaytracedObject[]): Scene {
    return {
        simulations: [{ ownerId: 100, width: 64, height: 64, raysPerFrame: 1, integrationInterval: 1, photonBounces: 1 }],
        objects: [makeObject(100, 0), makeObject(1, 3), makeObject(2, 7)],
        cameras: [],
        raytraced: raytraced ?? [makeRaytraced(1), makeRaytraced(2)],
        sprites: [],
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
    raytracedResources: RaytracedResources;
    transformResources: TransformResources;
    scene: Scene;
    sceneGraph: SceneGraph;
    textureCache: TextureCache;
}

async function setup(raytraced?: RaytracedObject[]): Promise<Fixture> {
    const device = createFakeGpuDevice();
    const gpuDevice = device as unknown as GPUDevice;
    const textureCache = new TextureCache(gpuDevice);
    const lutResources = new LutResources(gpuDevice, textureCache);
    const computedDataManager = new ComputedDataManager(gpuDevice);
    const simulationResources = new SimulationResources(gpuDevice, computedDataManager);
    const cameraBindGroupLayout = gpuDevice.createBindGroupLayout({ entries: [] });
    simulationResources.initialize(cameraBindGroupLayout, lutResources);

    const raytracedResources = new RaytracedResources(gpuDevice, computedDataManager);
    raytracedResources.initialize();
    const transformResources = new TransformResources(gpuDevice);

    const scene = makeScene(raytraced);
    const sceneGraph = new SceneGraph(scene);
    textureCache.loadScene('', scene.textureAtlasKeys);
    simulationResources.loadFromScene(scene, sceneGraph);
    await raytracedResources.loadFromScene(scene, sceneGraph, textureCache, simulationResources, transformResources);

    return { device, raytracedResources, transformResources, scene, sceneGraph, textureCache };
}

interface RecordedDraw {
    instanceCount: number;
    firstInstance: number;
}

/** A minimal GPURenderPassEncoder/GPUCommandEncoder stand-in that just records each draw() call's (instanceCount, firstInstance), for asserting on RaytracedResources' run-batching behavior. */
function makeRecordingEncoder(): { encoder: GPUCommandEncoder; draws: RecordedDraw[] } {
    const draws: RecordedDraw[] = [];
    const passEncoder = {
        setPipeline: () => {},
        setVertexBuffer: () => {},
        setBindGroup: () => {},
        draw: (_vertexCount: number, instanceCount: number, _firstVertex: number, firstInstance: number) => {
            draws.push({ instanceCount, firstInstance });
        },
        end: () => {},
    };
    const encoder = {
        beginRenderPass: () => passEncoder,
    } as unknown as GPUCommandEncoder;
    return { encoder, draws };
}

describe('RaytracedResources.renderGBuffer', () => {
    it('batches a run of consecutive, active, same-texture objects into a single instanced draw call', async () => {
        // All resolve to the same fallback texture in this test harness (no fetch stub) -
        // exactly the case renderGBuffer()'s run-length batching should collapse into one call.
        const fixture = await setup([makeRaytraced(1, 0), makeRaytraced(2, 1), makeRaytraced(1, 2), makeRaytraced(2, 3)]);
        const recorded = makeRecordingEncoder();

        fixture.raytracedResources.renderGBuffer(recorded.encoder);

        expect(recorded.draws).toEqual([{ instanceCount: 4, firstInstance: 0 }]);
    });

    it('splits an otherwise-same-texture run around an inactive object (an instanced draw cannot skip a middle instance)', async () => {
        const objects = [makeObject(100, 0), makeObject(1, 0), makeObject(2, 0), makeObject(3, 0), makeObject(4, 0)];
        objects[3].active = false; // the 3rd raytraced object in draw order (ownerId 3) is inactive
        const raytraced = [makeRaytraced(1, 0), makeRaytraced(2, 1), makeRaytraced(3, 2), makeRaytraced(4, 3)];
        const scene: Scene = { ...makeScene(raytraced), objects };

        const device = createFakeGpuDevice();
        const gpuDevice = device as unknown as GPUDevice;
        const textureCache = new TextureCache(gpuDevice);
        const lutResources = new LutResources(gpuDevice, textureCache);
        const computedDataManager = new ComputedDataManager(gpuDevice);
        const simulationResources = new SimulationResources(gpuDevice, computedDataManager);
        const cameraBindGroupLayout = gpuDevice.createBindGroupLayout({ entries: [] });
        simulationResources.initialize(cameraBindGroupLayout, lutResources);
        const raytracedResources = new RaytracedResources(gpuDevice, computedDataManager);
        raytracedResources.initialize();
        const transformResources = new TransformResources(gpuDevice);
        const sceneGraph = new SceneGraph(scene);
        textureCache.loadScene('', scene.textureAtlasKeys);
        simulationResources.loadFromScene(scene, sceneGraph);
        await raytracedResources.loadFromScene(scene, sceneGraph, textureCache, simulationResources, transformResources);

        const recorded = makeRecordingEncoder();
        raytracedResources.renderGBuffer(recorded.encoder);

        // ownerId 3's object (position 2) is excluded entirely; positions [0,2) and [3,4) are two
        // separate runs, since a single instanced draw can't skip the gap between them.
        expect(recorded.draws).toEqual([
            { instanceCount: 2, firstInstance: 0 },
            { instanceCount: 1, firstInstance: 3 },
        ]);
    });

    it('is a no-op with zero draw calls when there is no simulation to size the G-Buffer against', async () => {
        const scene = makeScene([makeRaytraced(1)]);
        scene.simulations = [];
        const device = createFakeGpuDevice();
        const gpuDevice = device as unknown as GPUDevice;
        const textureCache = new TextureCache(gpuDevice);
        const lutResources = new LutResources(gpuDevice, textureCache);
        const computedDataManager = new ComputedDataManager(gpuDevice);
        const simulationResources = new SimulationResources(gpuDevice, computedDataManager);
        const cameraBindGroupLayout = gpuDevice.createBindGroupLayout({ entries: [] });
        simulationResources.initialize(cameraBindGroupLayout, lutResources);
        const raytracedResources = new RaytracedResources(gpuDevice, computedDataManager);
        raytracedResources.initialize();
        const transformResources = new TransformResources(gpuDevice);
        const sceneGraph = new SceneGraph(scene);
        textureCache.loadScene('', scene.textureAtlasKeys);
        simulationResources.loadFromScene(scene, sceneGraph);
        await raytracedResources.loadFromScene(scene, sceneGraph, textureCache, simulationResources, transformResources);

        const recorded = makeRecordingEncoder();
        raytracedResources.renderGBuffer(recorded.encoder);

        expect(recorded.draws).toEqual([]);
    });
});

describe('compareRaytracedDrawOrder', () => {
    it('orders ascending by sortOrder', () => {
        expect(compareRaytracedDrawOrder({ sortOrder: -100 }, { sortOrder: 100 })).toBeLessThan(0);
        expect(compareRaytracedDrawOrder({ sortOrder: 5 }, { sortOrder: 1 })).toBeGreaterThan(0);
    });

    it('returns 0 for equal sortOrder - relative order is unobserved by design', () => {
        expect(compareRaytracedDrawOrder({ sortOrder: 3 }, { sortOrder: 3 })).toBe(0);
    });
});
