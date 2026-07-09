/**
 * Loads image paths into GPUTextures and caches them by path. Also provides
 * 1x1 white/black fallback textures, matching the Unity reference shader's
 * own material defaults (_MainTex = "white", _LightMap = "black").
 */
export class TextureCache {
    private device: GPUDevice;
    private cache = new Map<string, GPUTexture>();
    private whiteFallback: GPUTexture;
    private blackFallback: GPUTexture;
    public readonly sampler: GPUSampler;

    constructor(device: GPUDevice) {
        this.device = device;
        this.sampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
        });
        this.whiteFallback = this.createSolidColorTexture([255, 255, 255, 255]);
        this.blackFallback = this.createSolidColorTexture([0, 0, 0, 0]);
    }

    public getWhiteFallback(): GPUTexture {
        return this.whiteFallback;
    }

    public getBlackFallback(): GPUTexture {
        return this.blackFallback;
    }

    /**
     * Resolves an image path to a GPUTexture. Empty paths resolve to the
     * requested fallback without any network access.
     */
    public async resolve(path: string, fallback: 'white' | 'black' = 'white'): Promise<GPUTexture> {
        if (!path) {
            return fallback === 'white' ? this.whiteFallback : this.blackFallback;
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
            const fallbackTexture = fallback === 'white' ? this.whiteFallback : this.blackFallback;
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
