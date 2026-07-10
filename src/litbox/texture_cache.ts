import type { TextureAtlasKey, UvTransform } from './scene.ts';

const IDENTITY_UV_TRANSFORM: UvTransform = { a: 1, b: 0, c: 0, d: 0, e: 1, f: 0 };

// ".bc1" file layout, written by the scene exporter's C# BinaryWriter (little-endian):
//   char[4]  magic    "BC11"
//   uint16   version  (currently always 1)
//   uint16   reserved
//   uint32   width
//   uint32   height
//   uint32   blockDataLength
//   byte[]   blockData (raw BC1/DXT1 compressed blocks, 8 bytes per 4x4 pixel block)
const BC1_MAGIC = 'BC11';
const BC1_HEADER_SIZE_BYTES = 4 + 2 + 2 + 4 + 4 + 4;
const BC1_BLOCK_SIZE_BYTES = 8;
const BC1_BLOCK_DIM = 4;

export interface ResolvedTexture {
    texture: GPUTexture;
    uvTransform: UvTransform;
}

interface AtlasKeyEntry {
    atlasName: string;
    uvTransform: UvTransform;
}

/**
 * Loads image file names (paths relative to the active scene's own JSON file - see
 * loadScene) into GPUTextures and caches them by name, and tracks which atlas (if any) each
 * name has been packed into - see scene.ts's TextureAtlasKey - so callers get both the right
 * GPUTexture and the right sub-rectangle of it from a single lookup. A name ending in ".bc1"
 * is loaded as a BC1/DXT1-compressed texture (see loadBc1Texture); everything else goes
 * through the browser's own image decoder. Also provides 1x1 white/black default textures,
 * matching the Unity reference shader's own material defaults (_MainTex = "white", _LightMap
 * = "black").
 */
export class TextureCache {
    private device: GPUDevice;
    private cache = new Map<string, GPUTexture>();
    private atlasKeys = new Map<string, AtlasKeyEntry>();
    // Directory (relative to Vite's BASE_URL) containing the active scene's JSON file - see
    // LitboxScene.baseUrl. Every texture path, atlas-member or not, is a path relative to the
    // scene's own JSON file (per the scene exporter's convention), not to BASE_URL directly.
    private baseUrl = '';
    private whiteTexture: GPUTexture;
    private blackTexture: GPUTexture;

    /** Trilinear filtering (linear min/mag/mip), clamped to edge. */
    public readonly trilinearClamped: GPUSampler;
    /** Bilinear filtering (linear min/mag, nearest mip level), clamped to edge. */
    public readonly bilinearClamped: GPUSampler;
    /** Nearest-neighbor filtering (min/mag/mip), clamped to edge. */
    public readonly nearestClamped: GPUSampler;

    constructor(device: GPUDevice) {
        this.device = device;
        this.trilinearClamped = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
        });
        this.bilinearClamped = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'nearest',
        });
        this.nearestClamped = device.createSampler({
            magFilter: 'nearest',
            minFilter: 'nearest',
            mipmapFilter: 'nearest',
        });
        this.whiteTexture = this.createSolidColorTexture([255, 255, 255, 255]);
        this.blackTexture = this.createSolidColorTexture([0, 0, 0, 0]);
    }

    public getWhiteTexture(): GPUTexture {
        return this.whiteTexture;
    }

    public getBlackTexture(): GPUTexture {
        return this.blackTexture;
    }

    /**
     * (Re)loads the active scene's texture path base directory (see LitboxScene.baseUrl) and
     * atlas key table (see scene.ts's textureAtlasKeys).
     */
    public loadScene(baseUrl: string, textureAtlasKeys: readonly TextureAtlasKey[]): void {
        this.baseUrl = baseUrl;
        this.atlasKeys = new Map(textureAtlasKeys.map(key => [key.textureName, { atlasName: key.atlasName, uvTransform: key.uvTransform }]));
    }

    /**
     * Resolves an image/atlas-member name to its GPUTexture and the UV transform needed to
     * sample it there (identity if the name isn't atlassed). An empty name ("no texture
     * assigned") resolves to `fallback`'s built-in 1x1 solid-color texture, unless the loaded
     * atlas keys include an entry literally named "white"/"black", in which case that's used
     * instead - e.g. an atlas entry named "white" lets a scene swap in a proper default
     * texture for every unassigned-image sprite/map that falls back to white.
     */
    public async resolve(name: string, fallback: 'white' | 'black' = 'white'): Promise<ResolvedTexture> {
        const atlasKey = this.atlasKeys.get(name || fallback);
        const texture = await this.resolveTexture(atlasKey?.atlasName ?? name, fallback);
        return { texture, uvTransform: atlasKey?.uvTransform ?? IDENTITY_UV_TRANSFORM };
    }

    private async resolveTexture(name: string, fallback: 'white' | 'black'): Promise<GPUTexture> {
        if (!name) {
            return fallback === 'white' ? this.whiteTexture : this.blackTexture;
        }

        const cached = this.cache.get(name);
        if (cached) {
            return cached;
        }

        try {
            const url = `${import.meta.env.BASE_URL}${this.baseUrl}${name}`;
            const texture = name.toLowerCase().endsWith('.bc1')
                ? await this.loadBc1Texture(url, name)
                : await this.loadImageTexture(url);

            this.cache.set(name, texture);
            return texture;
        } catch (error) {
            console.error(`Litbox texture cache: failed to load "${name}":`, error);
            const fallbackTexture = fallback === 'white' ? this.whiteTexture : this.blackTexture;
            this.cache.set(name, fallbackTexture);
            return fallbackTexture;
        }
    }

    private async loadImageTexture(url: string): Promise<GPUTexture> {
        const response = await fetch(url);
        const blob = await response.blob();
        const bitmap = await createImageBitmap(blob);

        const texture = this.device.createTexture({
            size: [bitmap.width, bitmap.height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height]);
        return texture;
    }

    /** Loads a ".bc1" file (see the header layout documented above `BC1_MAGIC`) as a BC1/DXT1-compressed GPUTexture. */
    private async loadBc1Texture(url: string, name: string): Promise<GPUTexture> {
        if (!this.device.features.has('texture-compression-bc')) {
            throw new Error(`"${name}" is BC1-compressed, but this GPUDevice wasn't given the "texture-compression-bc" feature.`);
        }

        const response = await fetch(url);
        const buffer = await response.arrayBuffer();
        const view = new DataView(buffer);

        const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
        if (magic !== BC1_MAGIC) {
            throw new Error(`"${name}": expected .bc1 magic "${BC1_MAGIC}", got "${magic}".`);
        }
        const version = view.getUint16(4, true);
        if (version !== 1) {
            throw new Error(`"${name}": unsupported .bc1 version ${version}.`);
        }
        const width = view.getUint32(8, true);
        const height = view.getUint32(12, true);
        const blockDataLength = view.getUint32(16, true);
        const blockData = new Uint8Array(buffer, BC1_HEADER_SIZE_BYTES, blockDataLength);

        // "srgb" in the file name (matching this project's other atlas-naming convention, e.g.
        // "atlas_rgba32_srgb_0.png") opts into sRGB-aware sampling; otherwise the data is read
        // as linear.
        const format: GPUTextureFormat = name.toLowerCase().includes('srgb') ? 'bc1-rgba-unorm-srgb' : 'bc1-rgba-unorm';
        const texture = this.device.createTexture({
            size: [width, height],
            format,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        const blocksPerRow = Math.ceil(width / BC1_BLOCK_DIM);
        const blockRows = Math.ceil(height / BC1_BLOCK_DIM);
        const bytesPerRow = blocksPerRow * BC1_BLOCK_SIZE_BYTES;

        // The exporter writes block ROWS in Unity's own bottom-up texture memory order (row 0
        // = the bottom of the image) - the opposite of every other texture path here, where
        // decoding a standard image format (createImageBitmap) always yields top-down data.
        // Reverse block-row order (each block's own 4x4 pixel data is untouched - only *which*
        // block row it ends up at moves) so this ends up top-down like everything else, and
        // the shared vertex-shader V-flip (which corrects for the exporter's UV convention,
        // not raw file storage order) applies uniformly across atlas textures regardless of
        // file format.
        const flippedBlockData = new Uint8Array(blockData.length);
        for (let row = 0; row < blockRows; row++) {
            const srcOffset = row * bytesPerRow;
            const dstOffset = (blockRows - 1 - row) * bytesPerRow;
            flippedBlockData.set(blockData.subarray(srcOffset, srcOffset + bytesPerRow), dstOffset);
        }

        this.device.queue.writeTexture(
            { texture },
            flippedBlockData,
            { bytesPerRow, rowsPerImage: blockRows },
            [width, height],
        );
        return texture;
    }

    private createSolidColorTexture(rgba: [number, number, number, number]): GPUTexture {
        const texture = this.device.createTexture({
            size: [1, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.device.queue.writeTexture(
            { texture },
            new Uint8Array(rgba),
            { bytesPerRow: 4 },
            [1, 1],
        );
        return texture;
    }
}
