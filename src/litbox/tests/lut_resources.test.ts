import { describe, expect, it } from 'vitest';
import { LutResources, float32ToFloat16, packFloat16Rgba } from '../lut_resources.ts';
import { TextureCache } from '../texture_cache.ts';
import { BRDF_LUT_RESOLUTION, TEARDROP_SCATTERING_LUT_SAMPLES } from '../lut.ts';
import { createFakeGpuDevice } from './test_gpu_stubs.ts';

describe('float32ToFloat16', () => {
    it('matches known IEEE-754 half-precision bit patterns', () => {
        expect(float32ToFloat16(0)).toBe(0x0000);
        expect(float32ToFloat16(1.0)).toBe(0x3c00);
        expect(float32ToFloat16(-1.0)).toBe(0xbc00);
        expect(float32ToFloat16(0.5)).toBe(0x3800);
        expect(float32ToFloat16(-2.0)).toBe(0xc000);
    });

    it('rounds to nearest-even at the halfway point between two representable halfs', () => {
        // 1.0 has half-precision ULP 2^-10; a value exactly halfway between two representable
        // halfs should round to whichever has an even mantissa bit.
        const justAbove1 = 1 + Math.pow(2, -11); // exactly halfway between 1.0 and the next half
        const bits = float32ToFloat16(justAbove1);
        // 1.0 (0x3c00, even mantissa) wins the round-to-even tie over 0x3c01 (odd).
        expect(bits).toBe(0x3c00);
    });

    it('flushes very small magnitudes to zero (underflow) and preserves sign', () => {
        expect(float32ToFloat16(1e-30)).toBe(0x0000);
        expect(float32ToFloat16(-1e-30)).toBe(0x8000);
    });

    it('saturates to signed infinity on overflow', () => {
        expect(float32ToFloat16(1e30)).toBe(0x7c00);
        expect(float32ToFloat16(-1e30)).toBe(0xfc00);
    });
});

describe('packFloat16Rgba', () => {
    it('widens 3-component data to 4 with a zero alpha every 4th slot', () => {
        const data = new Float32Array([1, 0, 0, 0, 1, 0]); // 2 texels, xyz each
        const packed = packFloat16Rgba(data, 3);
        expect(packed.length).toBe(8);
        expect(packed[3]).toBe(0);
        expect(packed[7]).toBe(0);
        expect(packed[0]).toBe(float32ToFloat16(1));
        expect(packed[5]).toBe(float32ToFloat16(1));
    });

    it('passes already-4-component data through unchanged', () => {
        const data = new Float32Array([1, 2, 3, 4]);
        const packed = packFloat16Rgba(data, 4);
        expect(Array.from(packed)).toEqual([1, 2, 3, 4].map(float32ToFloat16));
    });
});

describe('LutResources', () => {
    function makeFixture() {
        const device = createFakeGpuDevice();
        const textureCache = new TextureCache(device as unknown as GPUDevice);
        device.createTextureCalls = []; // TextureCache's own white/black textures aren't under test
        device.writeTextureCalls = [];
        const lutResources = new LutResources(device as unknown as GPUDevice, textureCache);
        return { device, textureCache, lutResources };
    }

    it('creates the Teardrop scattering LUT as a 2048x1 rgba16float 2D texture', () => {
        const { device } = makeFixture();
        const teardropCall = device.createTextureCalls.find((c) => c.size[0] === TEARDROP_SCATTERING_LUT_SAMPLES);
        expect(teardropCall).toEqual({
            size: [TEARDROP_SCATTERING_LUT_SAMPLES, 1],
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
    });

    it('creates the BRDF LUT as a 128x64x16 rgba16float 3D texture', () => {
        const { device } = makeFixture();
        const [width, height, depth] = BRDF_LUT_RESOLUTION;
        const brdfCall = device.createTextureCalls.find((c) => c.dimension === '3d');
        expect(brdfCall).toEqual({
            size: [width, height, depth],
            dimension: '3d',
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
    });

    it('uploads correctly-sized float16 payloads for both LUTs', () => {
        const { device } = makeFixture();
        const [width, height, depth] = BRDF_LUT_RESOLUTION;

        expect(device.writeTextureCalls).toHaveLength(2);
        const teardropWrite = device.writeTextureCalls.find((c) => c.size[0] === TEARDROP_SCATTERING_LUT_SAMPLES);
        const brdfWrite = device.writeTextureCalls.find((c) => c.size[0] === width);

        expect(teardropWrite?.data.byteLength).toBe(TEARDROP_SCATTERING_LUT_SAMPLES * 4 * 2);
        expect(brdfWrite?.data.byteLength).toBe(width * height * depth * 4 * 2);
    });

    it('reuses TextureCache.bilinearClamped instead of creating its own sampler', () => {
        const { textureCache, lutResources } = makeFixture();
        expect(lutResources.getSampler()).toBe(textureCache.bilinearClamped);
    });

    it('exposes matching resolutions alongside the texture views', () => {
        const { lutResources } = makeFixture();
        expect(lutResources.getTeardropScatteringSampleCount()).toBe(TEARDROP_SCATTERING_LUT_SAMPLES);
        expect(lutResources.getBrdfResolution()).toEqual(BRDF_LUT_RESOLUTION);
        expect(lutResources.getTeardropScatteringView()).toBeDefined();
        expect(lutResources.getBrdfView()).toBeDefined();
    });
});
