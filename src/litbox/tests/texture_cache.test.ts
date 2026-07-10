import { describe, expect, it, vi } from 'vitest';
import { TextureCache } from '../texture_cache.ts';
import { createFakeGpuDevice } from './test_gpu_stubs.ts';
import type { TextureAtlasKey, UvTransform } from '../scene.ts';

const IDENTITY: UvTransform = { a: 1, b: 0, c: 0, d: 0, e: 1, f: 0 };

function makeAtlasKey(textureName: string, atlasName: string, uvTransform: UvTransform): TextureAtlasKey {
    return { textureName, atlasName, uvTransform };
}

/** Builds a well-formed ".bc1" file buffer matching TextureCache's expected header layout. */
function makeBc1Buffer(width: number, height: number, blockData: Uint8Array, options: { magic?: string; version?: number } = {}): ArrayBuffer {
    const magic = options.magic ?? 'BC11';
    const version = options.version ?? 1;
    const buffer = new ArrayBuffer(20 + blockData.length);
    const view = new DataView(buffer);
    for (let i = 0; i < 4; i++) {
        view.setUint8(i, magic.charCodeAt(i));
    }
    view.setUint16(4, version, true);
    view.setUint16(6, 0, true);
    view.setUint32(8, width, true);
    view.setUint32(12, height, true);
    view.setUint32(16, blockData.length, true);
    new Uint8Array(buffer, 20).set(blockData);
    return buffer;
}

function stubFetchArrayBuffer(buffer: ArrayBuffer): void {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ arrayBuffer: () => Promise.resolve(buffer) }));
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

describe('TextureCache BC1 loading', () => {
    it('loads a .bc1 file as a compressed texture when the device supports texture-compression-bc', async () => {
        const device = createFakeGpuDevice();
        device.features.add('texture-compression-bc');
        const textureCache = new TextureCache(device as unknown as GPUDevice);

        const blockData = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]); // one 4x4 BC1 block
        stubFetchArrayBuffer(makeBc1Buffer(4, 4, blockData));

        const resolved = await textureCache.resolve('atlas.bc1', 'white');

        expect(resolved.texture).not.toBe(textureCache.getWhiteTexture());
        expect(device.createTextureCalls).toContainEqual(
            expect.objectContaining({ size: [4, 4], format: 'bc1-rgba-unorm' }),
        );
        // The constructor already wrote the two built-in solid-color textures; this call's
        // write is the one after those.
        const bc1Write = device.writeTextureCalls.at(-1)!;
        expect(bc1Write.dataLayout).toEqual({ bytesPerRow: 8, rowsPerImage: 1 });
        expect(bc1Write.data).toEqual(blockData);

        vi.unstubAllGlobals();
    });

    it('uses the sRGB BC1 format when the file name contains "srgb"', async () => {
        const device = createFakeGpuDevice();
        device.features.add('texture-compression-bc');
        const textureCache = new TextureCache(device as unknown as GPUDevice);

        stubFetchArrayBuffer(makeBc1Buffer(4, 4, new Uint8Array(8)));
        await textureCache.resolve('atlas_srgb.bc1', 'white');

        expect(device.createTextureCalls).toContainEqual(
            expect.objectContaining({ format: 'bc1-rgba-unorm-srgb' }),
        );

        vi.unstubAllGlobals();
    });

    it('computes bytesPerRow/rowsPerImage in whole 4x4 blocks for dimensions not a multiple of 4', async () => {
        const device = createFakeGpuDevice();
        device.features.add('texture-compression-bc');
        const textureCache = new TextureCache(device as unknown as GPUDevice);

        // 5x5 pixels -> 2x2 blocks (ceil(5/4)), 8 bytes/block -> 16 bytes/row, 2 rows
        const blockData = new Uint8Array(32);
        stubFetchArrayBuffer(makeBc1Buffer(5, 5, blockData));
        await textureCache.resolve('odd.bc1', 'white');

        const bc1Write = device.writeTextureCalls.at(-1)!;
        expect(bc1Write.dataLayout).toEqual({ bytesPerRow: 16, rowsPerImage: 2 });
        expect(bc1Write.size).toEqual([5, 5]);

        vi.unstubAllGlobals();
    });

    it('reverses block-row order on upload (the exporter writes rows bottom-up; every other path here is top-down)', async () => {
        const device = createFakeGpuDevice();
        device.features.add('texture-compression-bc');
        const textureCache = new TextureCache(device as unknown as GPUDevice);

        // 4x8 pixels -> 1 block wide, 2 block rows. Row 0 = all 0x11 bytes, row 1 = all 0x22 bytes.
        const row0 = new Uint8Array(8).fill(0x11);
        const row1 = new Uint8Array(8).fill(0x22);
        const blockData = new Uint8Array([...row0, ...row1]);
        stubFetchArrayBuffer(makeBc1Buffer(4, 8, blockData));

        await textureCache.resolve('rows.bc1', 'white');

        const bc1Write = device.writeTextureCalls.at(-1)!;
        expect(bc1Write.data).toEqual(new Uint8Array([...row1, ...row0])); // rows swapped
    });

    it('falls back to the solid color texture when the device lacks texture-compression-bc', async () => {
        const device = createFakeGpuDevice(); // features empty by default
        const textureCache = new TextureCache(device as unknown as GPUDevice);
        stubFetchArrayBuffer(makeBc1Buffer(4, 4, new Uint8Array(8)));

        vi.spyOn(console, 'error').mockImplementation(() => {});
        const resolved = await textureCache.resolve('atlas.bc1', 'black');
        expect(resolved.texture).toBe(textureCache.getBlackTexture());
        expect(device.createTextureCalls).toHaveLength(2); // just the two built-in solid-color textures, no compressed texture

        vi.unstubAllGlobals();
        vi.restoreAllMocks();
    });

    it('falls back to the solid color texture when the file has a bad magic or unsupported version', async () => {
        const device = createFakeGpuDevice();
        device.features.add('texture-compression-bc');
        const textureCache = new TextureCache(device as unknown as GPUDevice);
        vi.spyOn(console, 'error').mockImplementation(() => {});

        stubFetchArrayBuffer(makeBc1Buffer(4, 4, new Uint8Array(8), { magic: 'XXXX' }));
        const badMagic = await textureCache.resolve('bad_magic.bc1', 'white');
        expect(badMagic.texture).toBe(textureCache.getWhiteTexture());

        stubFetchArrayBuffer(makeBc1Buffer(4, 4, new Uint8Array(8), { version: 2 }));
        const badVersion = await textureCache.resolve('bad_version.bc1', 'white');
        expect(badVersion.texture).toBe(textureCache.getWhiteTexture());

        vi.unstubAllGlobals();
        vi.restoreAllMocks();
    });
});
