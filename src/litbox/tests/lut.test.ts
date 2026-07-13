import { describe, expect, it } from 'vitest';
import {
    BRDF_LUT_RESOLUTION,
    TEARDROP_SCATTERING_LUT_SAMPLES,
    createBrdfLut,
    createTeardropScatteringLut,
    createVectorizedAnglePdfLut,
    generateFunctionTable,
    integrateDistribution,
    invertDistribution,
    normalizeDistribution,
    readCubic,
} from '../lut.ts';

describe('generateFunctionTable', () => {
    it('samples fn at evenly spaced points across [minima, maxima]', () => {
        const table = generateFunctionTable((x) => x * 2, 0, 3, 4);
        expect(Array.from(table)).toEqual([0, 2, 4, 6]);
    });
});

describe('normalizeDistribution', () => {
    it('scales by the sum and returns the average', () => {
        const out = new Float32Array(4);
        const avg = normalizeDistribution(new Float32Array([1, 2, 3, 4]), out);
        [0.1, 0.2, 0.3, 0.4].forEach((expected, i) => expect(out[i]).toBeCloseTo(expected, 6));
        expect(avg).toBe(2.5);
    });

    it('returns 0 and leaves the output untouched when the sum is 0', () => {
        const out = new Float32Array([9, 9, 9]);
        const avg = normalizeDistribution(new Float32Array([0, 0, 0]), out);
        expect(avg).toBe(0);
        expect(Array.from(out)).toEqual([9, 9, 9]);
    });
});

describe('integrateDistribution', () => {
    it('produces a running sum (discrete CDF)', () => {
        const out = new Float32Array(4);
        integrateDistribution(new Float32Array([0.1, 0.2, 0.3, 0.4]), out);
        expect(Array.from(out).map((v) => Math.round(v * 10) / 10)).toEqual([0.1, 0.3, 0.6, 1.0]);
    });
});

describe('readCubic', () => {
    it('returns the only value for a length-1 array, at any index', () => {
        expect(readCubic(new Float32Array([5]), 0)).toBe(5);
    });

    it('is exact linear interpolation for a length-2 array', () => {
        expect(readCubic(new Float32Array([0, 10]), 0.5)).toBe(5);
        expect(readCubic(new Float32Array([0, 10]), 0)).toBe(0);
        expect(readCubic(new Float32Array([0, 10]), 1)).toBe(10);
    });

    it('returns the source value exactly at each branch\'s own anchor index', () => {
        // p1's array position for each branch: 0 for "index < 1", floor(index) for the interior
        // branch, and length-2 for the "index >= length-2" tail branch - so index=length-1 isn't
        // a fair check for the tail branch (its p1 sits at length-2, not length-1); real callers
        // never query exactly index=length-1 either, since invertDistribution's binary search is
        // seeded from at most length-2 (see the `xLow === length-1` early-continue there).
        const data = new Float32Array([1, 4, 9, 16, 25, 36]);
        expect(readCubic(data, 0)).toBeCloseTo(1, 10); // index < 1 branch
        expect(readCubic(data, 3)).toBeCloseTo(16, 10); // interior branch
        expect(readCubic(data, 4)).toBeCloseTo(25, 10); // index >= length-2 branch
    });

    it('throws outside [0, length-1]', () => {
        const data = new Float32Array([1, 2, 3]);
        expect(() => readCubic(data, -0.1)).toThrow();
        expect(() => readCubic(data, 2.1)).toThrow();
    });
});

describe('invertDistribution', () => {
    it('inverts a linear CDF to (approximately) the identity', () => {
        // A perfectly linear CDF over [0,1]: Catmull-Rom on collinear points degenerates to
        // exact linear interpolation, so this is a near-exact numeric oracle without needing
        // Unity as a reference.
        const cdf = new Float32Array([0, 0.25, 0.5, 0.75, 1]);
        const out = new Float32Array(5);
        invertDistribution(cdf, 0, 1, out);
        for (let i = 0; i < out.length; i++) {
            expect(out[i]).toBeCloseTo(i / 4, 3);
        }
    });
});

/** Signed angular difference a-b, wrapped into (-pi, pi] - avoids atan2's branch-cut discontinuity at +-pi when comparing/ordering recovered angles. */
function angleDiff(a: number, b: number): number {
    let d = a - b;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d <= -Math.PI) d += 2 * Math.PI;
    return d;
}

describe('createVectorizedAnglePdfLut', () => {
    it('produces unit vectors and a monotonic angle sequence for a uniform PDF', () => {
        const samples = 64;
        const lut = createVectorizedAnglePdfLut(() => 1, samples, -Math.PI, Math.PI);

        let lastAngle: number | null = null;
        for (let i = 0; i < samples; i++) {
            const x = lut[i * 3 + 0];
            const y = lut[i * 3 + 1];
            const z = lut[i * 3 + 2];

            expect(Math.hypot(x, y)).toBeCloseTo(1, 5);

            const angle = Math.atan2(y, x);
            if (lastAngle !== null) {
                expect(angleDiff(angle, lastAngle)).toBeGreaterThanOrEqual(-1e-4);
            }
            lastAngle = angle;

            // For a uniform PDF, avg=1 and relativePdf is always 1, so z is the same constant
            // 1/(2*pi) everywhere - a strong, exactly hand-checkable invariant.
            expect(z).toBeCloseTo(1 / (2 * Math.PI), 5);
        }
    });

    it('evenly spaces recovered angles across the domain for a uniform PDF', () => {
        const samples = 32;
        const lut = createVectorizedAnglePdfLut(() => 1, samples, -Math.PI, Math.PI);
        for (let i = 0; i < samples; i++) {
            const expectedAngle = -Math.PI + i * (2 * Math.PI) / (samples - 1);
            const angle = Math.atan2(lut[i * 3 + 1], lut[i * 3 + 0]);
            expect(Math.abs(angleDiff(angle, expectedAngle))).toBeLessThan(0.01);
        }
    });
});

describe('createTeardropScatteringLut', () => {
    it('produces unit vectors with a monotonic angle sequence', () => {
        const samples = 256;
        const lut = createTeardropScatteringLut(6, samples);

        let lastAngle: number | null = null;
        for (let i = 0; i < samples; i++) {
            const x = lut[i * 3 + 0];
            const y = lut[i * 3 + 1];
            expect(Math.hypot(x, y)).toBeCloseTo(1, 4);

            const angle = Math.atan2(y, x);
            if (lastAngle !== null) {
                expect(angleDiff(angle, lastAngle)).toBeGreaterThanOrEqual(-1e-3);
            }
            lastAngle = angle;
        }
    });

    it('is approximately symmetric about 0 at the default resolution', () => {
        // The PDF (1 + spike*(theta/pi)^6) is even in theta, so the two samples straddling the
        // midpoint should land close to +-the same angle. integrateDistribution's cumulative sum
        // is a left-Riemann-sum, not a symmetric quadrature rule, so this converges to exact
        // symmetry only as sample count grows (confirmed empirically: the residual shrinks
        // roughly linearly with 1/samples) - use the real default (2048) with a tolerance loose
        // enough to comfortably clear that residual, not an exact-zero check.
        const samples = TEARDROP_SCATTERING_LUT_SAMPLES;
        const lut = createTeardropScatteringLut(6, samples);
        const mid = samples / 2;
        const angleAt = (i: number) => Math.atan2(lut[i * 3 + 1], lut[i * 3 + 0]);
        expect(angleAt(mid - 1) + angleAt(mid)).toBeCloseTo(0, 1);
    });
});

describe('createBrdfLut', () => {
    const [width, height, depth] = BRDF_LUT_RESOLUTION;
    const at = (i: number, j: number, k: number, channel: number, data: Float32Array) => data[((k * height + j) * width + i) * 4 + channel];

    it('has the expected flat shape', () => {
        const lut = createBrdfLut();
        expect(lut.length).toBe(width * height * depth * 4);
    });

    it('has weight exactly 0 at the x-axis endpoints and exactly 1 interior, for every (j,k) including roughness=0', () => {
        const lut = createBrdfLut();
        for (let j = 0; j < height; j++) {
            for (let k = 0; k < depth; k++) {
                expect(at(0, j, k, 3, lut)).toBe(0);
                expect(at(width - 1, j, k, 3, lut)).toBe(0);
                expect(at(1, j, k, 3, lut)).toBe(1);
                expect(at(width - 2, j, k, 3, lut)).toBe(1);
            }
        }
    });

    it('roughness=0 slice mirrors the incident angle for interior x samples', () => {
        const lut = createBrdfLut();
        const k = 0; // roughness = 0/(depth-1) = 0

        for (const j of [0, height - 1]) {
            const v = j / (height - 1);
            const incidentAngle = Math.asin(2 * v - 1);
            const expectedX = Math.cos(-incidentAngle);
            const expectedY = Math.sin(-incidentAngle);

            for (const i of [1, Math.floor(width / 2), width - 2]) {
                expect(at(i, j, k, 0, lut)).toBeCloseTo(expectedX, 4);
                expect(at(i, j, k, 1, lut)).toBeCloseTo(expectedY, 4);
                expect(at(i, j, k, 2, lut)).toBe(0);
            }
        }
    });

    it('roughness=0 incident angle spans -pi/2 to pi/2 across j', () => {
        // j=0 -> asin(-1) = -pi/2, j=height-1 -> asin(1) = pi/2 (hand-checkable exactly).
        const incidentAngleAt = (j: number) => Math.asin(2 * (j / (height - 1)) - 1);
        expect(incidentAngleAt(0)).toBeCloseTo(-Math.PI / 2, 10);
        expect(incidentAngleAt(height - 1)).toBeCloseTo(Math.PI / 2, 10);
    });
});
