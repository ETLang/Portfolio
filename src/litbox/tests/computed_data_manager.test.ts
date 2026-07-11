import { describe, expect, it } from 'vitest';
import { ComputedDataManager } from '../computed_data_manager.ts';
import { createFakeGpuDevice, type FakeGpuBuffer, type FakeGpuTexture } from './test_gpu_stubs.ts';

describe('ComputedDataManager', () => {
    describe('textures', () => {
        it('creates a new texture on first acquire', () => {
            const device = createFakeGpuDevice();
            const manager = new ComputedDataManager(device as unknown as GPUDevice);

            const pooled = manager.acquireTexture(64, 32, 'rgba16float', GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING);

            expect(device.createTextureCalls).toHaveLength(1);
            expect(device.createTextureCalls[0]).toEqual({
                size: [64, 32],
                format: 'rgba16float',
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                mipLevelCount: 1,
            });
            expect(pooled.width).toBe(64);
            expect(pooled.height).toBe(32);
            expect(pooled.view).toBeDefined();
        });

        it('reuses a released texture with a matching descriptor instead of creating a new one', () => {
            const device = createFakeGpuDevice();
            const manager = new ComputedDataManager(device as unknown as GPUDevice);
            const usage = GPUTextureUsage.TEXTURE_BINDING;

            const first = manager.acquireTexture(64, 64, 'rgba8unorm', usage);
            manager.releaseTexture(first);
            const second = manager.acquireTexture(64, 64, 'rgba8unorm', usage);

            expect(second).toBe(first);
            expect(device.createTextureCalls).toHaveLength(1);
        });

        it('does not reuse a released texture across a different descriptor', () => {
            const device = createFakeGpuDevice();
            const manager = new ComputedDataManager(device as unknown as GPUDevice);
            const usage = GPUTextureUsage.TEXTURE_BINDING;

            const first = manager.acquireTexture(64, 64, 'rgba8unorm', usage);
            manager.releaseTexture(first);
            const second = manager.acquireTexture(64, 64, 'rgba16float', usage);

            expect(second).not.toBe(first);
            expect(device.createTextureCalls).toHaveLength(2);
        });

        it('acquireTextureLike matches width/height/mips of an existing texture, with an overridable format', () => {
            const device = createFakeGpuDevice();
            const manager = new ComputedDataManager(device as unknown as GPUDevice);
            const usage = GPUTextureUsage.TEXTURE_BINDING;

            const original = manager.acquireTexture(128, 96, 'rgba8unorm', usage, 4);
            const like = manager.acquireTextureLike(original, 'rgba16float');

            expect(like.width).toBe(128);
            expect(like.height).toBe(96);
            expect(like.mipLevelCount).toBe(4);
            expect(like.format).toBe('rgba16float');
        });

        it('getMipView caches views per level', () => {
            const device = createFakeGpuDevice();
            const manager = new ComputedDataManager(device as unknown as GPUDevice);
            const pooled = manager.acquireTexture(64, 64, 'rgba16float', GPUTextureUsage.RENDER_ATTACHMENT, 3);

            const viewA = pooled.getMipView(1);
            const viewB = pooled.getMipView(1);
            const viewC = pooled.getMipView(2);

            expect(viewA).toBe(viewB);
            expect(viewA).not.toBe(viewC);
        });

        it('purge destroys every pooled texture and clears the pool', () => {
            const device = createFakeGpuDevice();
            const manager = new ComputedDataManager(device as unknown as GPUDevice);
            const usage = GPUTextureUsage.TEXTURE_BINDING;

            const pooled = manager.acquireTexture(16, 16, 'rgba8unorm', usage);
            manager.releaseTexture(pooled);

            manager.purge();

            expect((pooled.texture as unknown as FakeGpuTexture).destroyed).toBe(true);

            const after = manager.acquireTexture(16, 16, 'rgba8unorm', usage);
            expect(after).not.toBe(pooled);
            expect(device.createTextureCalls).toHaveLength(2);
        });
    });

    describe('buffers', () => {
        it('creates a new buffer on first acquire, aligned to 4 bytes', () => {
            const device = createFakeGpuDevice();
            const manager = new ComputedDataManager(device as unknown as GPUDevice);

            const pooled = manager.acquireBuffer(15, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

            expect(pooled.size).toBe(16);
            expect(pooled.buffer.size).toBe(16);
        });

        it('reuses a released buffer with a matching size/usage instead of creating a new one', () => {
            const device = createFakeGpuDevice();
            const manager = new ComputedDataManager(device as unknown as GPUDevice);
            const usage = GPUBufferUsage.STORAGE;

            const first = manager.acquireBuffer(64, usage);
            manager.releaseBuffer(first);
            const second = manager.acquireBuffer(64, usage);

            expect(second).toBe(first);
        });

        it('does not reuse a released buffer across a different usage', () => {
            const device = createFakeGpuDevice();
            const manager = new ComputedDataManager(device as unknown as GPUDevice);

            const first = manager.acquireBuffer(64, GPUBufferUsage.STORAGE);
            manager.releaseBuffer(first);
            const second = manager.acquireBuffer(64, GPUBufferUsage.UNIFORM);

            expect(second).not.toBe(first);
        });

        it('acquireBufferWithData uploads the data to a newly-acquired buffer', () => {
            const device = createFakeGpuDevice();
            const manager = new ComputedDataManager(device as unknown as GPUDevice);
            const data = new Uint32Array([1, 2, 3, 4]);

            const pooled = manager.acquireBufferWithData(data, GPUBufferUsage.STORAGE);

            expect(device.writeCalls).toHaveLength(1);
            expect(device.writeCalls[0].buffer).toBe(pooled.buffer);
            expect(device.writeCalls[0].bufferOffset).toBe(0);
            expect(new Uint32Array(device.writeCalls[0].data)).toEqual(data);
        });

        it('purge destroys every pooled buffer and clears the pool', () => {
            const device = createFakeGpuDevice();
            const manager = new ComputedDataManager(device as unknown as GPUDevice);

            const pooled = manager.acquireBuffer(32, GPUBufferUsage.STORAGE);
            manager.releaseBuffer(pooled);

            manager.purge();

            expect((pooled.buffer as unknown as FakeGpuBuffer).destroyed).toBe(true);

            const after = manager.acquireBuffer(32, GPUBufferUsage.STORAGE);
            expect(after).not.toBe(pooled);
        });
    });

    describe('idle sweep', () => {
        it('purgeStale destroys and evicts a released resource once it has been idle longer than maxIdleMs', () => {
            const device = createFakeGpuDevice();
            let clock = 0;
            const manager = new ComputedDataManager(device as unknown as GPUDevice, 5000, () => clock);

            const texture = manager.acquireTexture(16, 16, 'rgba8unorm', GPUTextureUsage.TEXTURE_BINDING);
            const buffer = manager.acquireBuffer(32, GPUBufferUsage.STORAGE);
            manager.releaseTexture(texture);
            manager.releaseBuffer(buffer);

            clock = 4999;
            manager.purgeStale();
            expect((texture.texture as unknown as FakeGpuTexture).destroyed).toBe(false);
            expect((buffer.buffer as unknown as FakeGpuBuffer).destroyed).toBe(false);

            clock = 5001;
            manager.purgeStale();
            expect((texture.texture as unknown as FakeGpuTexture).destroyed).toBe(true);
            expect((buffer.buffer as unknown as FakeGpuBuffer).destroyed).toBe(true);

            const newTexture = manager.acquireTexture(16, 16, 'rgba8unorm', GPUTextureUsage.TEXTURE_BINDING);
            const newBuffer = manager.acquireBuffer(32, GPUBufferUsage.STORAGE);
            expect(newTexture).not.toBe(texture);
            expect(newBuffer).not.toBe(buffer);
        });

        it('does not purge a resource that is still within its idle window', () => {
            const device = createFakeGpuDevice();
            let clock = 0;
            const manager = new ComputedDataManager(device as unknown as GPUDevice, 5000, () => clock);

            const texture = manager.acquireTexture(16, 16, 'rgba8unorm', GPUTextureUsage.TEXTURE_BINDING);
            manager.releaseTexture(texture);

            clock = 1000;
            manager.purgeStale();

            const reused = manager.acquireTexture(16, 16, 'rgba8unorm', GPUTextureUsage.TEXTURE_BINDING);
            expect(reused).toBe(texture);
        });

        it('never destroys a resource that is currently acquired (not released)', () => {
            const device = createFakeGpuDevice();
            let clock = 0;
            const manager = new ComputedDataManager(device as unknown as GPUDevice, 5000, () => clock);

            const texture = manager.acquireTexture(16, 16, 'rgba8unorm', GPUTextureUsage.TEXTURE_BINDING);

            clock = 1_000_000;
            manager.purgeStale();

            expect((texture.texture as unknown as FakeGpuTexture).destroyed).toBe(false);
        });

        it('sweeps opportunistically during acquire/release, throttled to at most once per second', () => {
            const device = createFakeGpuDevice();
            let clock = 0;
            const manager = new ComputedDataManager(device as unknown as GPUDevice, 5000, () => clock);

            const stale = manager.acquireTexture(16, 16, 'rgba8unorm', GPUTextureUsage.TEXTURE_BINDING);
            manager.releaseTexture(stale);

            clock = 5001;
            // First activity after the throttle window triggers the sweep automatically.
            manager.acquireBuffer(32, GPUBufferUsage.STORAGE);

            expect((stale.texture as unknown as FakeGpuTexture).destroyed).toBe(true);
        });
    });
});
