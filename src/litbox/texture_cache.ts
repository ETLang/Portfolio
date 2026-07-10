import type { TextureAtlasKey, UvTransform } from './scene.ts';

const IDENTITY_UV_TRANSFORM: UvTransform = { a: 1, b: 0, c: 0, d: 0, e: 1, f: 0 };

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
 * GPUTexture and the right sub-rectangle of it from a single lookup. Also provides 1x1
 * white/black default textures, matching the Unity reference shader's own material defaults
 * (_MainTex = "white", _LightMap = "black").
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
            const response = await fetch(`${import.meta.env.BASE_URL}${this.baseUrl}${name}`);
            const blob = await response.blob();
            const bitmap = await createImageBitmap(blob);

            const texture = this.device.createTexture({
                size: [bitmap.width, bitmap.height],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
            });
            this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height]);

            this.cache.set(name, texture);
            return texture;
        } catch (error) {
            console.error(`Litbox texture cache: failed to load "${name}":`, error);
            const fallbackTexture = fallback === 'white' ? this.whiteTexture : this.blackTexture;
            this.cache.set(name, fallbackTexture);
            return fallbackTexture;
        }
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
