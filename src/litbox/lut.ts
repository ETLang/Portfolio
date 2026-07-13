// Ported from litbox/Assets/Scripts/Util/LUT.cs. Pure CPU-side math, no GPUDevice dependency -
// see lut_resources.ts for the GPU-resource-owning class that uploads what this module generates.
//
// A LUT here is a table approximating the inverse-CDF of some (not necessarily normalized)
// probability distribution: generate the PDF on a uniform grid, normalize it, integrate it into a
// CDF, then invert the CDF so a uniform random number in [0,1] can be turned into a
// distribution-weighted sample via a single texture lookup on the GPU.
//
// Samplers must remap [0,1] so 0 lands on the first texel's center and 1 on the last texel's
// center, not the texture edge - see sampleLut1D/sampleLut2D/sampleLut3D in
// shaders/LitboxCommon.wgsl.

/** Samples `fn` at `samples` evenly-spaced points across [minima, maxima]. */
export function generateFunctionTable(fn: (x: number) => number, minima: number, maxima: number, samples: number): Float32Array {
    const table = new Float32Array(samples);
    for (let i = 0; i < samples; i++) {
        const x = minima + (maxima - minima) * i / (samples - 1);
        table[i] = fn(x);
    }
    return table;
}

/** Writes `distribution` scaled by its own sum into `outNormalized`; returns the pre-normalization average (0 if the sum is 0, in which case `outNormalized` is left untouched). */
export function normalizeDistribution(distribution: Float32Array, outNormalized: Float32Array): number {
    let sum = 0;
    for (const v of distribution) {
        sum += v;
    }
    if (sum === 0) {
        return 0;
    }
    for (let i = 0; i < distribution.length; i++) {
        outNormalized[i] = distribution[i] / sum;
    }
    return sum / distribution.length;
}

/** Writes the running sum (discrete CDF) of `distribution` into `outIntegral`. */
export function integrateDistribution(distribution: Float32Array, outIntegral: Float32Array): void {
    let accum = 0;
    for (let i = 0; i < distribution.length; i++) {
        accum += distribution[i];
        outIntegral[i] = accum;
    }
}

/** Catmull-Rom cubic read of `data` at fractional index `index` (edge points extrapolated, not clamped). Ported from TextureExtensions.cs's ReadCubic. */
export function readCubic(data: Float32Array, index: number): number {
    if (index < 0 || index > data.length - 1) {
        throw new RangeError(`readCubic: index ${index} out of range [0, ${data.length - 1}]`);
    }
    if (data.length === 1) {
        return data[0];
    }
    if (data.length === 2) {
        return data[0] + index * (data[1] - data[0]);
    }

    let p0: number, p1: number, p2: number, p3: number;
    if (index < 1) {
        p1 = data[0];
        p2 = data[1];
        p3 = data[2];
        p0 = 3 * p1 - 3 * p2 + p3;
    } else if (index >= data.length - 2) {
        p0 = data[data.length - 3];
        p1 = data[data.length - 2];
        p2 = data[data.length - 1];
        p3 = p0 - 3 * p1 + 3 * p2;
    } else {
        const i = Math.floor(index);
        p0 = data[i - 1];
        p1 = data[i];
        p2 = data[i + 1];
        p3 = data[i + 2];
    }

    const x = index - Math.floor(index);
    const xx = x * x;
    const xxx = xx * x;
    return (-0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3) * xxx
        + (p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3) * xx
        + (-0.5 * p0 + 0.5 * p2) * x
        + p1;
}

/**
 * Inverts a monotonically non-decreasing `fn` (e.g. a CDF from integrateDistribution) sampled
 * across [domainStart, domainEnd] into `outInverse`, an evenly-spaced table across `fn`'s own
 * [min, max] range. Uses binary search (via readCubic) to drill down to a precise domain value
 * for each output sample. Ported from LUT.cs's Invert - like the original, does not verify
 * monotonicity (callers here only ever pass contractually-nonnegative PDFs' integrals).
 */
export function invertDistribution(fn: Float32Array, domainStart: number, domainEnd: number, outInverse: Float32Array): void {
    let inverseStart = fn[0];
    let inverseEnd = fn[0];
    for (const v of fn) {
        if (v < inverseStart) inverseStart = v;
        if (v > inverseEnd) inverseEnd = v;
    }

    for (let i = 0; i < outInverse.length; i++) {
        const y = inverseStart + i * (inverseEnd - inverseStart) / (outInverse.length - 1);

        let xLow = fn.length - 1;
        for (let k = 0; k < fn.length; k++) {
            if (fn[k] > y) {
                xLow = k - 1;
                break;
            }
        }

        if (xLow === fn.length - 1) {
            outInverse[i] = domainStart + xLow * (domainEnd - domainStart) / (fn.length - 1);
            continue;
        }

        // Binary search to drill down to a precise domain value for this output sample.
        let low = xLow;
        let high = xLow + 1;
        while (high - low > 1e-5) {
            const mid = (low + high) / 2;
            const yMid = readCubic(fn, mid);
            if (yMid < y) {
                if (low === mid) break;
                low = mid;
            } else {
                if (high === mid) break;
                high = mid;
            }
        }

        outInverse[i] = domainStart + low * (domainEnd - domainStart) / (fn.length - 1);
    }
}

/**
 * Creates a random-angle generator table fitting a probability distribution function.
 *
 * `relativePdf` is sampled from `lower` to `upper` for angle probabilities; it need not be
 * normalized, but must be nonnegative across that domain.
 *
 * To use the generator, feed a uniform random number in [0,1] into GPU-side sampleLut1D as `u`.
 *
 * Returns a packed `samples*3` Float32Array (xyz interleaved): xy is the unit-length vector
 * pointing in the direction of the output angle, z is the inverse of the density at that angle
 * (the importance-sampling weight). Ported from LUT.cs's CreateVectorizedAnglePDFLUT.
 */
export function createVectorizedAnglePdfLut(relativePdf: (theta: number) => number, samples = 2048, lower = -Math.PI, upper = Math.PI): Float32Array {
    const table = generateFunctionTable(relativePdf, lower, upper, samples);
    const normalizedTable = new Float32Array(table.length);
    const avg = normalizeDistribution(table, normalizedTable);
    const integral = new Float32Array(table.length);
    integrateDistribution(normalizedTable, integral);
    const invertedAngles = new Float32Array(table.length);
    invertDistribution(integral, lower, upper, invertedAngles);

    const result = new Float32Array(table.length * 3);
    for (let i = 0; i < invertedAngles.length; i++) {
        const angle = invertedAngles[i];
        result[i * 3 + 0] = Math.cos(angle);
        result[i * 3 + 1] = Math.sin(angle);
        // Second, distinct evaluation of relativePdf - on the *inverted* angle just computed
        // above, not the original sample grid - matches LUT.cs:163-165 exactly. Easy to get
        // wrong by reusing `table`/`normalizedTable` here instead.
        result[i * 3 + 2] = avg / (2 * Math.PI * relativePdf(angle));
    }
    return result;
}

/** Must match TEARDROP_SCATTERING_LUT_TEXEL_COUNT define expected by sampleLut1D call sites sampling this LUT - see LitboxCommon.wgsl. */
export const TEARDROP_SCATTERING_LUT_SAMPLES = 2048;

/** Packed samples*3 (xyz interleaved) - see createVectorizedAnglePdfLut's return shape. Ported from LUT.cs's CreateTeardropScatteringLUT. */
export function createTeardropScatteringLut(spikeStrength = 6, samples = TEARDROP_SCATTERING_LUT_SAMPLES): Float32Array {
    return createVectorizedAnglePdfLut((theta) => {
        const x = theta / Math.PI;
        return 1 + spikeStrength * Math.pow(x, 6);
    }, samples);
}

/** [x, y, z] texel counts. Must match BRDF_LUT_TEXEL_COUNT_X/Y/Z defines expected by sampleLut3D call sites sampling this LUT - see LitboxCommon.wgsl. */
export const BRDF_LUT_RESOLUTION = [128, 64, 16] as const;

/**
 * BRDF LUT: 3 dimensional, axes are x=random-scatter PDF (128 samples, same
 * createVectorizedAnglePdfLut machinery as the Teardrop LUT, using a Trowbridge-Reitz/GGX normal
 * distribution as the PDF basis), y=incident angle (64), z=roughness (16).
 *
 * Returns a packed `128*64*16*4` Float32Array (xyzw interleaved, x-fastest / then y / then z -
 * standard WebGPU 3D texture writeTexture layout). Per output texel: xy is the unit-length
 * scattered-direction vector, z is that sample's slope *magnitude* (not a vector - used by a
 * future consumer for Hermite-spline reconstruction across the x axis), w is a blend weight (0 at
 * the two x-axis endpoints, 1 interior). Ported from LUT.cs's CreateBDRFLUT.
 */
export function createBrdfLut(): Float32Array {
    const [width, height, depth] = BRDF_LUT_RESOLUTION;
    const output = new Float32Array(width * height * depth * 4);
    const at = (i: number, j: number, k: number, channel: number): number => ((k * height + j) * width + i) * 4 + channel;

    for (let j = 0; j < height; j++) {
        const v = j / (height - 1);
        const normalCrossIncident = 2 * v - 1;
        const incidentAngle = Math.asin(normalCrossIncident);

        for (let k = 0; k < depth; k++) {
            const roughness = k / (depth - 1);

            const linePdf = createVectorizedAnglePdfLut((theta) => {
                // Trowbridge-Reitz (GGX) normal distribution function as the BRDF basis.
                const halfAngle = (theta + incidentAngle) / 2;
                const r = roughness * roughness;
                const cosHalf = Math.cos(halfAngle);
                return 1.0 / Math.pow(cosHalf * cosHalf * (r * r - 1) + 1, 2);
            }, width, -Math.PI / 2 + 0.0001, Math.PI / 2 - 0.0001);

            for (let i = 0; i < width; i++) {
                const x = linePdf[i * 3 + 0];
                const y = linePdf[i * 3 + 1];

                let slopeDx: number, slopeDy: number, weight: number, maxMag: number;
                if (i === 0) {
                    slopeDx = linePdf[(i + 1) * 3 + 0] - x;
                    slopeDy = linePdf[(i + 1) * 3 + 1] - y;
                    weight = 0;
                    maxMag = Number.MAX_VALUE;
                } else if (i === width - 1) {
                    slopeDx = x - linePdf[(i - 1) * 3 + 0];
                    slopeDy = y - linePdf[(i - 1) * 3 + 1];
                    weight = 0;
                    maxMag = Number.MAX_VALUE;
                } else {
                    const nx = linePdf[(i + 1) * 3 + 0], ny = linePdf[(i + 1) * 3 + 1];
                    const px = linePdf[(i - 1) * 3 + 0], py = linePdf[(i - 1) * 3 + 1];
                    // Clamp before acos: unit-vector dot products can drift slightly past +-1 in
                    // floating point, which would otherwise produce NaN here.
                    const angle1 = Math.acos(Math.min(1, Math.max(-1, nx * x + ny * y)));
                    const angle2 = Math.acos(Math.min(1, Math.max(-1, x * px + y * py)));
                    slopeDx = (nx - px) / 2;
                    slopeDy = (ny - py) / 2;
                    weight = 1;
                    maxMag = Math.min(angle1, angle2) * 1.5;
                }

                const slopeMag = Math.min(maxMag, Math.hypot(slopeDx, slopeDy));
                output[at(i, j, k, 0)] = x;
                output[at(i, j, k, 1)] = y;
                output[at(i, j, k, 2)] = slopeMag;
                output[at(i, j, k, 3)] = weight;
            }

            if (roughness === 0) {
                const rx = Math.cos(-incidentAngle);
                const ry = Math.sin(-incidentAngle);
                for (let i = 1; i < width - 1; i++) {
                    output[at(i, j, k, 0)] = rx;
                    output[at(i, j, k, 1)] = ry;
                    output[at(i, j, k, 2)] = 0;
                    output[at(i, j, k, 3)] = 1;
                }
            }
        }
    }

    return output;
}
