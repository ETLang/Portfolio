import { describe, expect, it } from 'vitest';
import { LitboxScene } from '../litbox_scene.ts';
import type { Color, PointLight, Scene, SceneObject, SceneSprite, Spotlight } from '../scene.ts';

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
