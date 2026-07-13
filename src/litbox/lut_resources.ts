import type { TextureCache } from './texture_cache.ts';
import {
    BRDF_LUT_RESOLUTION,
    TEARDROP_SCATTERING_LUT_SAMPLES,
    createBrdfLut,
    createTeardropScatteringLut,
} from './lut.ts';

const LUT_FORMAT: GPUTextureFormat = 'rgba16float';

/**
 * Converts an IEEE-754 single-precision float to a half-precision (float16) bit pattern
 * (round-to-nearest-even, matching the DataView.setFloat16 behavior this project's TS lib target
 * doesn't yet expose type declarations for - ES2023, not ES2025+). Exported for direct unit
 * testing since round-trip correctness matters here (this is the only place in the codebase that
 * writes raw float16 texture data from JS - every other rgba16float texture is a render/compute
 * target, never CPU-uploaded).
 */
export function float32ToFloat16(value: number): number {
    f32Scratch[0] = value;
    const bits = u32Scratch[0];
    const sign = (bits >>> 16) & 0x8000;
    const exp = (bits >>> 23) & 0xff;
    const mant = bits & 0x7fffff;

    if (exp === 0xff) {
        return sign | 0x7c00 | (mant !== 0 ? 0x0200 : 0); // Inf / NaN
    }

    let e = exp - 127 + 15; // candidate half-precision exponent field
    if (e >= 0x1f) {
        return sign | 0x7c00; // overflow -> Infinity
    }
    if (e <= 0) {
        if (e < -10) {
            return sign; // underflow -> zero
        }
        const mantWithLeadingBit = mant | 0x800000;
        const shift = 14 - e;
        let half = mantWithLeadingBit >>> shift;
        const remainder = mantWithLeadingBit & ((1 << shift) - 1);
        const halfway = 1 << (shift - 1);
        if (remainder > halfway || (remainder === halfway && (half & 1) === 1)) {
            half += 1; // round to nearest even
        }
        return sign | half; // a carry out of the mantissa here lands correctly in the exponent field
    }

    let half = mant >>> 13;
    const remainder = mant & 0x1fff;
    if (remainder > 0x1000 || (remainder === 0x1000 && (half & 1) === 1)) {
        half += 1;
        if (half === 0x400) {
            half = 0;
            e += 1;
            if (e >= 0x1f) {
                return sign | 0x7c00;
            }
        }
    }
    return sign | (e << 10) | half;
}

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

/**
 * Packs `data` (tightly packed, `componentsPerTexel` floats per texel) into a float16 RGBA
 * payload ready for GPUQueue.writeTexture - widening 3-component data to 4 with a zero alpha,
 * passing already-4-component data through unchanged.
 */
export function packFloat16Rgba(data: Float32Array, componentsPerTexel: 3 | 4): Uint16Array {
    const texelCount = data.length / componentsPerTexel;
    const out = new Uint16Array(texelCount * 4);
    for (let texel = 0; texel < texelCount; texel++) {
        for (let channel = 0; channel < 4; channel++) {
            out[texel * 4 + channel] = channel < componentsPerTexel
                ? float32ToFloat16(data[texel * componentsPerTexel + channel])
                : 0;
        }
    }
    return out;
}

/**
 * Owns the procedural, static lookup tables used for importance-sampled scattering - see
 * litbox/Assets/Scripts/Util/LUT.cs (generation math, ported in lut.ts) and BufferManager.cs
 * (TeardropScatteringLUT/BRDFLUT registration) for the Unity reference. Generated once at
 * construction and never rewritten, so - unlike ComputedDataManager's pooled scratch textures -
 * these are owned directly, the same way TextureCache owns its white/black default textures.
 *
 * Sampling these on the GPU requires a texel-center remap (LUT-space [0,1] must land on the
 * first/last texel's *center*, not the texture edge) - see sampleLut1D/sampleLut3D in
 * shaders/LitboxCommon.wgsl. Each LUT's texel count must reach those functions as a #define (see
 * getTeardropScatteringSampleCount/getBrdfResolution below and the doc comments in lut.ts) -
 * never hardcoded into a .wgsl file.
 */
export class LutResources {
    private teardropScatteringTexture: GPUTexture;
    private teardropScatteringView: GPUTextureView;
    private brdfTexture: GPUTexture;
    private brdfView: GPUTextureView;
    private sampler: GPUSampler;

    constructor(device: GPUDevice, textureCache: TextureCache) {
        // Unity's AsTexture() never overrides filterMode on the LUT textures it creates, so it
        // uses Unity's own default (bilinear) - reuse TextureCache's existing bilinear/clamped
        // sampler rather than creating a redundant one.
        this.sampler = textureCache.bilinearClamped;

        const teardropData = createTeardropScatteringLut();
        this.teardropScatteringTexture = device.createTexture({
            size: [TEARDROP_SCATTERING_LUT_SAMPLES, 1],
            format: LUT_FORMAT,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture(
            { texture: this.teardropScatteringTexture },
            packFloat16Rgba(teardropData, 3),
            { bytesPerRow: TEARDROP_SCATTERING_LUT_SAMPLES * 4 * 2, rowsPerImage: 1 },
            [TEARDROP_SCATTERING_LUT_SAMPLES, 1],
        );
        this.teardropScatteringView = this.teardropScatteringTexture.createView();

        const [brdfWidth, brdfHeight, brdfDepth] = BRDF_LUT_RESOLUTION;
        const brdfData = createBrdfLut();
        this.brdfTexture = device.createTexture({
            size: [brdfWidth, brdfHeight, brdfDepth],
            dimension: '3d',
            format: LUT_FORMAT,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture(
            { texture: this.brdfTexture },
            packFloat16Rgba(brdfData, 4),
            { bytesPerRow: brdfWidth * 4 * 2, rowsPerImage: brdfHeight },
            [brdfWidth, brdfHeight, brdfDepth],
        );
        this.brdfView = this.brdfTexture.createView();
    }

    public getTeardropScatteringView(): GPUTextureView {
        return this.teardropScatteringView;
    }

    /** Must reach GPU-side sampleLut1D call sites as the TEARDROP_SCATTERING_LUT_TEXEL_COUNT define - see lut.ts. */
    public getTeardropScatteringSampleCount(): number {
        return TEARDROP_SCATTERING_LUT_SAMPLES;
    }

    public getBrdfView(): GPUTextureView {
        return this.brdfView;
    }

    /** [x, y, z] texel counts - must reach GPU-side sampleLut3D call sites as the BRDF_LUT_TEXEL_COUNT_X/Y/Z defines - see lut.ts. */
    public getBrdfResolution(): typeof BRDF_LUT_RESOLUTION {
        return BRDF_LUT_RESOLUTION;
    }

    public getSampler(): GPUSampler {
        return this.sampler;
    }
}
