import { mat4 } from 'gl-matrix';
import { describe, expect, it } from 'vitest';
import {
    segmentIntersectsUnitEllipse,
    segmentIntersectsUnitRect,
    testSegmentAgainstShape,
    type LocalHit,
} from '../collision.ts';
import type { Vector2 } from '../scene.ts';

function expectVectorClose(actual: Vector2, expected: Vector2, precision = 6): void {
    expect(actual.x).toBeCloseTo(expected.x, precision);
    expect(actual.y).toBeCloseTo(expected.y, precision);
}

function expectHitClose(actual: LocalHit | null, expected: LocalHit): void {
    expect(actual).not.toBeNull();
    expect(actual!.t).toBeCloseTo(expected.t, 6);
    expectVectorClose(actual!.point, expected.point);
    expectVectorClose(actual!.normal, expected.normal);
}

describe('segmentIntersectsUnitEllipse', () => {
    it('straight line through the center: enters at the near edge', () => {
        const hit = segmentIntersectsUnitEllipse({ x: -1, y: 0 }, { x: 1, y: 0 });
        expectHitClose(hit, { t: 0.25, point: { x: -0.5, y: 0 }, normal: { x: -1, y: 0 } });
    });

    it('exact tangent: still registers a single-point hit (discriminant === 0)', () => {
        const hit = segmentIntersectsUnitEllipse({ x: -1, y: 0.5 }, { x: 1, y: 0.5 });
        expectHitClose(hit, { t: 0.5, point: { x: 0, y: 0.5 }, normal: { x: 0, y: 1 } });
    });

    it('clean miss: line passes outside the circle entirely', () => {
        expect(segmentIntersectsUnitEllipse({ x: -1, y: 1 }, { x: 1, y: 1 })).toBeNull();
    });

    it('zero-length segment outside the circle: no hit', () => {
        expect(segmentIntersectsUnitEllipse({ x: 1, y: 1 }, { x: 1, y: 1 })).toBeNull();
    });

    it('already inside: immediate hit at t=0, normal points away from center', () => {
        const hit = segmentIntersectsUnitEllipse({ x: 0.1, y: 0.1 }, { x: 1, y: 1 });
        const norm = Math.SQRT1_2;
        expectHitClose(hit, { t: 0, point: { x: 0.1, y: 0.1 }, normal: { x: norm, y: norm } });
    });
});

describe('segmentIntersectsUnitRect', () => {
    it('edge hit, no ambiguity: straight into the right face', () => {
        const hit = segmentIntersectsUnitRect({ x: 1, y: 0 }, { x: -1, y: 0 });
        expectHitClose(hit, { t: 0.25, point: { x: 0.5, y: 0 }, normal: { x: 1, y: 0 } });
    });

    it('corner hit: diagonal grazing exactly through (0.5, 0.5) - x-axis slab processed first wins the tie', () => {
        const hit = segmentIntersectsUnitRect({ x: 1, y: 1 }, { x: -1, y: -1 });
        expectHitClose(hit, { t: 0.25, point: { x: 0.5, y: 0.5 }, normal: { x: 1, y: 0 } });
    });

    it('clean miss: parallel to and outside the x slab', () => {
        expect(segmentIntersectsUnitRect({ x: 1, y: 1 }, { x: 1, y: 2 })).toBeNull();
    });

    it('already inside: immediate hit at t=0, normal picks the nearest face', () => {
        const hit = segmentIntersectsUnitRect({ x: 0.1, y: 0.1 }, { x: 2, y: 2 });
        expectHitClose(hit, { t: 0, point: { x: 0.1, y: 0.1 }, normal: { x: 1, y: 0 } });
    });
});

describe('testSegmentAgainstShape', () => {
    it('ellipse: translation + uniform scale carry the local hit into world space', () => {
        const transform = mat4.create();
        mat4.translate(transform, transform, [10, 5, 0]);
        mat4.scale(transform, transform, [2, 2, 1]);
        const inverse = mat4.create();
        mat4.invert(inverse, transform);

        const hit = testSegmentAgainstShape({ x: 8, y: 5 }, { x: 12, y: 5 }, transform, inverse, 'ellipse');
        expect(hit).not.toBeNull();
        expectVectorClose(hit!.point, { x: 9, y: 5 });
        expectVectorClose(hit!.normal, { x: -1, y: 0 });
    });

    it('rect: a pure rotation carries both the segment and the resulting normal correctly', () => {
        const transform = mat4.create();
        mat4.rotateZ(transform, transform, Math.PI / 2);
        const inverse = mat4.create();
        mat4.invert(inverse, transform);

        // Local segment (1,0)->(-1,0), a clean edge hit at local (0.5, 0) - rotated 90 degrees
        // into world space as (0,1)->(0,-1), expected to hit world (0, 0.5) with normal (0, 1).
        const hit = testSegmentAgainstShape({ x: 0, y: 1 }, { x: 0, y: -1 }, transform, inverse, 'rect');
        expect(hit).not.toBeNull();
        expectVectorClose(hit!.point, { x: 0, y: 0.5 });
        expectVectorClose(hit!.normal, { x: 0, y: 1 });
    });

    it('returns null when the underlying local test misses', () => {
        const transform = mat4.create();
        const hit = testSegmentAgainstShape({ x: -1, y: 1 }, { x: 1, y: 1 }, transform, transform, 'ellipse');
        expect(hit).toBeNull();
    });
});
