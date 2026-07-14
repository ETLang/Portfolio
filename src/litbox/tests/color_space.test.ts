import { describe, expect, it } from 'vitest';
import { linearColorToSrgb, linearToSrgb, srgbColorToLinear, srgbToLinear } from '../color_space.ts';

describe('srgbToLinear', () => {
    it('maps the endpoints identically', () => {
        expect(srgbToLinear(0)).toBe(0);
        expect(srgbToLinear(1)).toBe(1);
    });

    it('matches the standard sRGB EOTF at a known midtone value', () => {
        // 0.5 sRGB -> ~0.214041 linear (standard reference value for the IEC 61966-2-1 curve).
        expect(srgbToLinear(0.5)).toBeCloseTo(0.214041, 5);
    });

    it('uses the linear segment below the 0.04045 threshold', () => {
        expect(srgbToLinear(0.04045)).toBeCloseTo(0.04045 / 12.92, 6);
        expect(srgbToLinear(0.02)).toBeCloseTo(0.02 / 12.92, 6);
    });
});

describe('srgbColorToLinear', () => {
    it('converts r/g/b but passes alpha through unchanged', () => {
        const result = srgbColorToLinear({ r: 1, g: 0.5, b: 0, a: 0.5 });
        expect(result.r).toBe(1);
        expect(result.g).toBeCloseTo(0.214041, 5);
        expect(result.b).toBe(0);
        expect(result.a).toBe(0.5); // unconverted
    });
});

describe('linearToSrgb', () => {
    it('maps the endpoints identically', () => {
        expect(linearToSrgb(0)).toBe(0);
        // Not exactly 1 due to floating-point rounding in Math.pow(1, 1/2.4) - toBeCloseTo, not toBe.
        expect(linearToSrgb(1)).toBeCloseTo(1, 10);
    });

    it('is the inverse of srgbToLinear', () => {
        for (const c of [0, 0.02, 0.04045, 0.1, 0.214041, 0.5, 0.75, 1]) {
            expect(linearToSrgb(srgbToLinear(c))).toBeCloseTo(c, 6);
            expect(srgbToLinear(linearToSrgb(c))).toBeCloseTo(c, 6);
        }
    });

    it('uses the linear segment below the 0.0031308 threshold', () => {
        expect(linearToSrgb(0.0031308)).toBeCloseTo(0.0031308 * 12.92, 6);
        expect(linearToSrgb(0.001)).toBeCloseTo(0.001 * 12.92, 6);
    });
});

describe('linearColorToSrgb', () => {
    it('converts r/g/b but passes alpha through unchanged', () => {
        const result = linearColorToSrgb({ r: 1, g: 0.214041, b: 0, a: 0.5 });
        expect(result.r).toBeCloseTo(1, 10);
        expect(result.g).toBeCloseTo(0.5, 5);
        expect(result.b).toBe(0);
        expect(result.a).toBe(0.5); // unconverted
    });
});
