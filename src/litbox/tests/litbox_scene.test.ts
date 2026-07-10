import { describe, expect, it } from 'vitest';
import { LitboxScene } from '../litbox_scene.ts';
import type { Color, PointLight, Scene, SceneCamera, SceneObject, SceneSimulation, SceneSprite, Spotlight } from '../scene.ts';

class TestScene extends LitboxScene {
    public static readonly jsonPath = 'test.json';
}

const WHITE: Color = { r: 1, g: 1, b: 1, a: 1 };

function makeObject(id: number, name: string, parentId: number): SceneObject {
    return {
        active: true,
        id,
        name,
        parentId,
        position: { x: 0, y: 0 },
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

function makePointLight(ownerId: number): PointLight {
    return { ownerId, color: WHITE, intensity: 1, bounces: 1 };
}

function makeSpotlight(ownerId: number): Spotlight {
    return { ownerId, color: WHITE, intensity: 1, pinch: 0.5, bounces: 1 };
}

function makeCamera(ownerId: number): SceneCamera {
    return { ownerId, verticalSize: 5, exposure: 1 };
}

function makeSimulation(ownerId: number): SceneSimulation {
    return { ownerId, width: 64, height: 64, raysPerFrame: 1, integrationInterval: 1, photonBounces: 1 };
}

// Root
//  - Left Wall
//    - Sprite (id 4)
//  - Right Wall
//    - Sprite (id 5)
//  - Light Owner (id 6): owns a PointLight and a Spotlight
function makeFixtureScene(): Scene {
    const objects = [
        makeObject(1, 'Root', -1),
        makeObject(2, 'Left Wall', 1),
        makeObject(3, 'Right Wall', 1),
        makeObject(4, 'Sprite', 2),
        makeObject(5, 'Sprite', 3),
        makeObject(6, 'Light Owner', 1),
    ];
    return {
        simulations: [],
        objects,
        cameras: [],
        raytraced: [],
        sprites: [makeSprite(4), makeSprite(5)],
        pointLights: [makePointLight(6)],
        spotlights: [makeSpotlight(6)],
        laserLights: [],
        directionalLights: [],
        ambientLights: [],
    };
}

describe('LitboxScene name/path resolution', () => {
    it('throws when a bare name is ambiguous', () => {
        const scene = new TestScene(makeFixtureScene());
        expect(() => scene.makeTransformDynamic('Sprite')).toThrow(/2 SceneObjects named "Sprite"/);
    });

    it('throws when a bare name matches nothing', () => {
        const scene = new TestScene(makeFixtureScene());
        expect(() => scene.makeTransformDynamic('Nonexistent')).toThrow(/no SceneObject named "Nonexistent"/);
    });

    it('resolves a "/"-separated path through a specific parent, disambiguating a duplicated leaf name', () => {
        const scene = new TestScene(makeFixtureScene());
        const leftSprite = scene.makeSpriteDynamic('Left Wall/Sprite');
        const rightSprite = scene.makeSpriteDynamic('Right Wall/Sprite');
        expect(leftSprite.ownerId).toBe(4);
        expect(rightSprite.ownerId).toBe(5);
    });

    it('throws at construction if any SceneObject name contains "/"', () => {
        const scene = makeFixtureScene();
        scene.objects.push(makeObject(99, 'Bad/Name', 1));
        expect(() => new TestScene(scene)).toThrow(/contains a "\/" character/);
    });

    it('finds lights combined across kind arrays in kind order (point before spot)', () => {
        const scene = new TestScene(makeFixtureScene());
        const first = scene.makeLightDynamic('Light Owner', 0);
        const second = scene.makeLightDynamic('Light Owner', 1);
        expect('bounces' in first && !('pinch' in first)).toBe(true);
        expect('pinch' in second).toBe(true);
    });
});

describe('LitboxScene dynamic/dirty marking', () => {
    it('make*Dynamic returns the live struct and marks it dynamic every frame', () => {
        const scene = new TestScene(makeFixtureScene());
        const obj = scene.makeTransformDynamic('Root');
        expect(scene.getDynamicFrameState().transforms).toContain(obj);
        scene.clearFrameDirtyFlags();
        expect(scene.getDynamicFrameState().transforms).toContain(obj);
    });

    it('markTransformDirty on an already-dynamic object is a no-op (stays dynamic across clearFrameDirtyFlags)', () => {
        const scene = new TestScene(makeFixtureScene());
        scene.makeTransformDynamic('Root');
        scene.markTransformDirty('Root');
        scene.clearFrameDirtyFlags();
        const rootMatches = scene.getDynamicFrameState().transforms.filter(o => o.name === 'Root');
        expect(rootMatches).toHaveLength(1);
    });

    it('clearFrameDirtyFlags drops dirty-only entries', () => {
        const scene = new TestScene(makeFixtureScene());
        scene.markTransformDirty('Root');
        expect(scene.getDynamicFrameState().transforms).toHaveLength(1);
        scene.clearFrameDirtyFlags();
        expect(scene.getDynamicFrameState().transforms).toHaveLength(0);
    });
});

describe('LitboxScene.createObject', () => {
    it('allocates a fresh id above the highest existing one and returns the live struct', () => {
        const scene = new TestScene(makeFixtureScene()); // highest existing id is 6
        const obj = scene.createObject({ name: 'New Object' });
        expect(obj.id).toBe(7);
        expect(obj.parentId).toBe(-1);
        expect(scene.data.objects).toContain(obj);
    });

    it('resolves options.parent via the existing path resolution', () => {
        const scene = new TestScene(makeFixtureScene());
        const obj = scene.createObject({ name: 'Nested', parent: 'Left Wall' });
        expect(obj.parentId).toBe(2); // 'Left Wall' id
    });

    it('is immediately resolvable by name for subsequent calls in the same onFrame', () => {
        const scene = new TestScene(makeFixtureScene());
        scene.createObject({ name: 'New Object' });
        expect(scene.makeTransformDynamic('New Object').name).toBe('New Object');
    });

    it('throws on a "/" in the name, without allocating an id', () => {
        const scene = new TestScene(makeFixtureScene());
        expect(() => scene.createObject({ name: 'Bad/Name' })).toThrow(/contains a "\/" character/);
        const obj = scene.createObject({ name: 'Next' });
        expect(obj.id).toBe(7); // id wasn't consumed by the failed call
    });

    it('records a pending create op', () => {
        const scene = new TestScene(makeFixtureScene());
        const obj = scene.createObject({ name: 'New Object' });
        const ops = scene.getPendingStructuralOps();
        expect(ops).toEqual([{ type: 'create', object: obj }]);
    });
});

describe('LitboxScene.destroyObject', () => {
    it('removes the object and its descendants from data.objects', () => {
        const scene = new TestScene(makeFixtureScene());
        scene.destroyObject('Left Wall'); // owns Sprite (id 4)
        expect(scene.data.objects.map(o => o.id)).toEqual([1, 3, 5, 6]);
    });

    it('removes owned sprites/lights/raytraced entries', () => {
        const scene = new TestScene(makeFixtureScene());
        scene.destroyObject('Light Owner');
        expect(scene.data.pointLights).toHaveLength(0);
        expect(scene.data.spotlights).toHaveLength(0);
    });

    it('cleans nameIndex so resolvePath throws afterward', () => {
        const scene = new TestScene(makeFixtureScene());
        scene.destroyObject('Left Wall');
        expect(() => scene.makeTransformDynamic('Left Wall')).toThrow(/no SceneObject named/);
    });

    it('drops dynamic/dirty flags for anything destroyed', () => {
        const scene = new TestScene(makeFixtureScene());
        scene.makeLightDynamic('Light Owner', 0);
        expect(scene.getDynamicFrameState().lights).toHaveLength(1);

        scene.destroyObject('Light Owner');

        expect(scene.getDynamicFrameState().lights).toHaveLength(0);
    });

    it('records a pending destroy op with just the root id', () => {
        const scene = new TestScene(makeFixtureScene());
        scene.destroyObject('Left Wall');
        expect(scene.getPendingStructuralOps()).toEqual([{ type: 'destroy', rootId: 2 }]);
    });

    it('throws without mutating data when the cascade owns a camera', () => {
        const fixture = makeFixtureScene();
        fixture.cameras.push(makeCamera(6)); // 'Light Owner'
        const scene = new TestScene(fixture);

        expect(() => scene.destroyObject('Light Owner')).toThrow(/camera or simulation/);
        expect(scene.data.objects.map(o => o.id)).toContain(6);
        expect(scene.getPendingStructuralOps()).toHaveLength(0);
    });

    it('throws without mutating data when the cascade owns a simulation', () => {
        const fixture = makeFixtureScene();
        fixture.simulations.push(makeSimulation(1)); // 'Root'
        const scene = new TestScene(fixture);

        expect(() => scene.destroyObject('Root')).toThrow(/camera or simulation/);
        expect(scene.data.objects.map(o => o.id)).toContain(1);
        expect(scene.getPendingStructuralOps()).toHaveLength(0);
    });
});

describe('LitboxScene.reparentObject', () => {
    it('updates parentId directly on the live struct', () => {
        const scene = new TestScene(makeFixtureScene());
        scene.reparentObject('Left Wall/Sprite', 'Right Wall');
        expect(scene.data.objects.find(o => o.id === 4)!.parentId).toBe(3);
    });

    it('moves an object to scene root when newParent is null', () => {
        const scene = new TestScene(makeFixtureScene());
        scene.reparentObject('Left Wall', null);
        expect(scene.data.objects.find(o => o.id === 2)!.parentId).toBe(-1);
    });

    it('throws on reparenting to itself, without mutating parentId', () => {
        const scene = new TestScene(makeFixtureScene());
        expect(() => scene.reparentObject('Left Wall', 'Left Wall')).toThrow(/itself/);
        expect(scene.data.objects.find(o => o.id === 2)!.parentId).toBe(1);
    });

    it('throws on reparenting to its own descendant', () => {
        const scene = new TestScene(makeFixtureScene());
        expect(() => scene.reparentObject('Left Wall', 'Left Wall/Sprite')).toThrow(/descendant/);
    });

    it('records a pending reparent op', () => {
        const scene = new TestScene(makeFixtureScene());
        scene.reparentObject('Left Wall/Sprite', 'Right Wall');
        expect(scene.getPendingStructuralOps()).toEqual([{ type: 'reparent', id: 4, newParentId: 3 }]);
    });
});

describe('LitboxScene pending structural ops', () => {
    it('clearPendingStructuralOps empties the queue', () => {
        const scene = new TestScene(makeFixtureScene());
        scene.createObject({ name: 'New Object' });
        scene.clearPendingStructuralOps();
        expect(scene.getPendingStructuralOps()).toEqual([]);
    });
});
