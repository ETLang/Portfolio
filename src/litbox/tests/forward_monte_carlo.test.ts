import { describe, expect, it } from 'vitest';
import { mat4, vec4 } from 'gl-matrix';
import {
    luminance,
    computeRayCount,
    resolveBounces,
    computeWorldToTargetPixels,
    computeLightToTarget,
    computeDirectionalLightSegment,
    combineWriteCount,
} from '../forward_monte_carlo.ts';

describe('forward_monte_carlo pure per-light math', () => {
    describe('luminance', () => {
        it('weights R/G/B by Rec.709 coefficients summing to 1', () => {
            expect(luminance([1, 0, 0])).toBeCloseTo(0.2126);
            expect(luminance([0, 1, 0])).toBeCloseTo(0.7152);
            expect(luminance([0, 0, 1])).toBeCloseTo(0.0722);
            expect(luminance([1, 1, 1])).toBeCloseTo(1.0);
        });
    });

    describe('computeRayCount', () => {
        it('rounds a light\'s luminance-weighted share up to the next multiple of 64', () => {
            // Dominant light (100% of total luma): 1000 -> rounds up to 1024.
            expect(computeRayCount(1, 1, 1000)).toBe(1024);
        });

        it('leaves an already-exact multiple of 64 unchanged', () => {
            expect(computeRayCount(0.5, 1, 128)).toBe(64);
            expect(computeRayCount(1, 1, 128)).toBe(128);
        });

        it('gives a zero-luminance light a minimum of 64 rays (matches Unity\'s rounding trick exactly, not a bug)', () => {
            expect(computeRayCount(0, 1, 100000)).toBe(64);
        });
    });

    describe('resolveBounces', () => {
        it('uses the light\'s own bounce count when photonBounces is -1 (Unity\'s OverrideBounceCount == null sentinel)', () => {
            expect(resolveBounces(-1, 5)).toBe(5);
        });

        it('overrides every light\'s bounce count when photonBounces is not -1', () => {
            expect(resolveBounces(3, 5)).toBe(3);
            expect(resolveBounces(0, 5)).toBe(0);
        });
    });

    describe('computeWorldToTargetPixels', () => {
        it('maps the simulation\'s local [-0.5,0.5]^2 rect to [0,width] x [0,height] pixel space under an identity world transform, flipping Y so world "up" lands at pixel row 0', () => {
            const worldToTargetPixels = computeWorldToTargetPixels(mat4.create(), 100, 200);

            const bottomLeft = transformPoint(worldToTargetPixels, [-0.5, -0.5, 0]);
            const topRight = transformPoint(worldToTargetPixels, [0.5, 0.5, 0]);

            expect(bottomLeft[0]).toBeCloseTo(0);
            expect(bottomLeft[1]).toBeCloseTo(200);
            expect(topRight[0]).toBeCloseTo(100);
            expect(topRight[1]).toBeCloseTo(0);
        });
    });

    describe('computeLightToTarget', () => {
        it('combines worldToTargetPixels with the light\'s own world transform', () => {
            const worldToTargetPixels = computeWorldToTargetPixels(mat4.create(), 100, 200);
            const lightWorldTransform = mat4.create();
            mat4.fromTranslation(lightWorldTransform, [0.1, 0.2, 0]);

            const lightToTarget = computeLightToTarget(worldToTargetPixels, lightWorldTransform);
            const origin = transformPoint(lightToTarget, [0, 0, 0]);

            expect(origin[0]).toBeCloseTo(60);
            expect(origin[1]).toBeCloseTo(60);
        });
    });

    describe('computeDirectionalLightSegment', () => {
        it('passes local "down" through unchanged under an identity transform', () => {
            expect(computeDirectionalLightSegment(mat4.create(), 100, 200).direction).toEqual([0, -1]);
        });

        it('rotates local "down" by the light\'s own rotation', () => {
            const rotated = mat4.create();
            mat4.fromZRotation(rotated, Math.PI / 2);

            // Rotating (0,-1) by +90 degrees around Z gives (1,0): x' = x*cos-y*sin = 1, y' = x*sin+y*cos = 0.
            const [x, y] = computeDirectionalLightSegment(rotated, 100, 200).direction;
            expect(x).toBeCloseTo(1);
            expect(y).toBeCloseTo(0);
        });

        it('builds a segment perpendicular to the direction, with length matching its own segmentVector', () => {
            const { direction, segmentVector, length } = computeDirectionalLightSegment(mat4.create(), 100, 200);

            expect(direction[0] * segmentVector[0] + direction[1] * segmentVector[1]).toBeCloseTo(0);
            expect(length).toBeCloseTo(Math.hypot(segmentVector[0], segmentVector[1]));
        });

        it('places the whole segment outside the [0,width]x[0,height] target rect, for every direction', () => {
            for (const angle of [0, Math.PI / 6, Math.PI / 2, Math.PI, 3.4]) {
                const rotated = mat4.create();
                mat4.fromZRotation(rotated, angle);
                const { segmentStart, segmentVector } = computeDirectionalLightSegment(rotated, 100, 200);

                for (const t of [0, 0.25, 0.5, 0.75, 1]) {
                    const x = segmentStart[0] + segmentVector[0] * t;
                    const y = segmentStart[1] + segmentVector[1] * t;
                    const outside = x < 0 || x > 100 || y < 0 || y > 200;
                    expect(outside).toBe(true);
                }
            }
        });
    });

    describe('combineWriteCount', () => {
        it('combines low/high u32 halves into the manual uint64 they represent', () => {
            expect(combineWriteCount(5, 0)).toBe(5n);
            expect(combineWriteCount(0xFFFFFFFF, 0)).toBe(4294967295n);
        });

        it('accounts for the overflow/carry count in the high half', () => {
            expect(combineWriteCount(1, 1)).toBe(4294967297n);
        });
    });
});

function transformPoint(m: mat4, point: readonly [number, number, number]): [number, number, number] {
    const out = vec4.create();
    vec4.transformMat4(out, [point[0], point[1], point[2], 1], m);
    return [out[0], out[1], out[2]];
}
