/**
 * Loads image paths into GPUTextures and caches them by path. Also provides
 * 1x1 white/black default textures, matching the Unity reference shader's
 * own material defaults (_MainTex = "white", _LightMap = "black").
 */
export class TextureCache {
    private device: GPUDevice;
    private cache = new Map<string, GPUTexture>();
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
     * Resolves an image path to a GPUTexture. Empty paths resolve to the
     * requested fallback without any network access.
     */
    public async resolve(path: string, fallback: 'white' | 'black' = 'white'): Promise<GPUTexture> {
        if (!path) {
            return fallback === 'white' ? this.whiteTexture : this.blackTexture;
        }

        const cached = this.cache.get(path);
        if (cached) {
            return cached;
        }

        try {
            const response = await fetch(path);
            const blob = await response.blob();
            const bitmap = await createImageBitmap(blob);

            const texture = this.device.createTexture({
                size: [bitmap.width, bitmap.height],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
            });
            this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height]);

            this.cache.set(path, texture);
            return texture;
        } catch (error) {
            console.error(`Litbox texture cache: failed to load "${path}":`, error);
            const fallbackTexture = fallback === 'white' ? this.whiteTexture : this.blackTexture;
            this.cache.set(path, fallbackTexture);
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
