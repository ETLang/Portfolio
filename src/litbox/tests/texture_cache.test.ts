import { describe, expect, it, vi } from 'vitest';
import { TextureCache } from '../texture_cache.ts';
import { createFakeGpuDevice } from './test_gpu_stubs.ts';
import type { TextureAtlasKey, UvTransform } from '../scene.ts';

const IDENTITY: UvTransform = { a: 1, b: 0, c: 0, d: 0, e: 1, f: 0 };

function makeAtlasKey(textureName: string, atlasName: string, uvTransform: UvTransform): TextureAtlasKey {
    return { textureName, atlasName, uvTransform };
}

describe('TextureCache', () => {
    it('resolves the empty name to the built-in solid texture with an identity transform, per fallback', async () => {
        const textureCache = new TextureCache(createFakeGpuDevice() as unknown as GPUDevice);

        const white = await textureCache.resolve('', 'white');
        expect(white.texture).toBe(textureCache.getWhiteTexture());
        expect(white.uvTransform).toEqual(IDENTITY);

        const black = await textureCache.resolve('', 'black');
        expect(black.texture).toBe(textureCache.getBlackTexture());
        expect(black.uvTransform).toEqual(IDENTITY);
    });

    it('resolves an atlassed name to its atlas uvTransform', async () => {
        const uvTransform: UvTransform = { a: 0.5, b: 0, c: 0, d: 0, e: 0.5, f: 0.5 };
        const textureCache = new TextureCache(createFakeGpuDevice() as unknown as GPUDevice);
        textureCache.loadScene('', [makeAtlasKey('Moon', 'Atlas1', uvTransform)]);

        vi.spyOn(console, 'error').mockImplementation(() => {}); // Atlas1 won't actually fetch in this test environment
        const resolved = await textureCache.resolve('Moon', 'white');
        expect(resolved.uvTransform).toEqual(uvTransform);
        vi.restoreAllMocks();
    });

    it('resolves an unmatched non-empty name to an identity transform', async () => {
        const textureCache = new TextureCache(createFakeGpuDevice() as unknown as GPUDevice);

        vi.spyOn(console, 'error').mockImplementation(() => {}); // "Unknown" won't actually fetch in this test environment
        const resolved = await textureCache.resolve('Unknown', 'white');
        expect(resolved.uvTransform).toEqual(IDENTITY);
        expect(resolved.texture).toBe(textureCache.getWhiteTexture()); // fetch failure falls back to the requested solid color
        vi.restoreAllMocks();
    });

    it('substitutes an atlas entry named "white" for the empty name, but not for "black"', async () => {
        const uvTransform: UvTransform = { a: 0.25, b: 0, c: 0, d: 0, e: 0.25, f: 0 };
        const textureCache = new TextureCache(createFakeGpuDevice() as unknown as GPUDevice);
        textureCache.loadScene('', [makeAtlasKey('white', 'Atlas1', uvTransform)]);

        vi.spyOn(console, 'error').mockImplementation(() => {});
        const white = await textureCache.resolve('', 'white');
        expect(white.uvTransform).toEqual(uvTransform);

        const black = await textureCache.resolve('', 'black');
        expect(black.uvTransform).toEqual(IDENTITY);
        expect(black.texture).toBe(textureCache.getBlackTexture());
        vi.restoreAllMocks();
    });

    it('substitutes an atlas entry named "black" for the empty name, but not for "white"', async () => {
        const uvTransform: UvTransform = { a: 0.1, b: 0, c: 0.2, d: 0, e: 0.1, f: 0.3 };
        const textureCache = new TextureCache(createFakeGpuDevice() as unknown as GPUDevice);
        textureCache.loadScene('', [makeAtlasKey('black', 'Atlas1', uvTransform)]);

        vi.spyOn(console, 'error').mockImplementation(() => {});
        const black = await textureCache.resolve('', 'black');
        expect(black.uvTransform).toEqual(uvTransform);

        const white = await textureCache.resolve('', 'white');
        expect(white.uvTransform).toEqual(IDENTITY);
        expect(white.texture).toBe(textureCache.getWhiteTexture());
        vi.restoreAllMocks();
    });

    it('caches a resolved texture by name, reusing it across calls', async () => {
        const textureCache = new TextureCache(createFakeGpuDevice() as unknown as GPUDevice);

        vi.spyOn(console, 'error').mockImplementation(() => {});
        const first = await textureCache.resolve('Unknown', 'white');
        const second = await textureCache.resolve('Unknown', 'white');
        expect(first.texture).toBe(second.texture);
        vi.restoreAllMocks();
    });

    it('fetches a texture path relative to the scene directory loaded via loadScene, not BASE_URL directly', async () => {
        const textureCache = new TextureCache(createFakeGpuDevice() as unknown as GPUDevice);
        const uvTransform: UvTransform = { a: 0.5, b: 0, c: 0, d: 0, e: 0.5, f: 0.5 };
        textureCache.loadScene('scenes/', [makeAtlasKey('Moon', 'scenes_atlases/atlas_0.png', uvTransform)]);

        const fetchMock = vi.fn().mockRejectedValue(new Error('no network in tests'));
        vi.stubGlobal('fetch', fetchMock);
        vi.spyOn(console, 'error').mockImplementation(() => {});

        await textureCache.resolve('Moon', 'white');
        expect(fetchMock).toHaveBeenCalledWith(`${import.meta.env.BASE_URL}scenes/scenes_atlases/atlas_0.png`);

        vi.unstubAllGlobals();
        vi.restoreAllMocks();
    });
});
