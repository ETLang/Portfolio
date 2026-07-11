import { describe, expect, it } from 'vitest';
import { PackedUniformArray } from '../packed_uniform_array.ts';
import { createFakeGpuDevice, type FakeGpuBuffer, type FakeGpuDevice } from './test_gpu_stubs.ts';

const STRIDE = 4; // one u32 per entry - small and easy to reason about in these tests

function fillValue(value: number): (view: DataView, byteOffset: number) => void {
    return (view, byteOffset) => view.setUint32(byteOffset, value, true);
}

/** Replays every recorded write to `buffer` (in order) into a same-sized scratch buffer, so tests can read back its effective final content without PackedUniformArray needing a read API of its own. */
function readBack(device: FakeGpuDevice, buffer: GPUBuffer): DataView {
    const fakeBuffer = buffer as unknown as FakeGpuBuffer;
    const scratch = new Uint8Array(fakeBuffer.size);
    for (const call of device.writeCalls) {
        if (call.buffer !== fakeBuffer) {
            continue;
        }
        scratch.set(new Uint8Array(call.data), call.bufferOffset);
    }
    return new DataView(scratch.buffer);
}

function makeArray(device: FakeGpuDevice, initialCapacity = 8): PackedUniformArray<number> {
    return new PackedUniformArray<number>(device as unknown as GPUDevice, STRIDE, initialCapacity);
}

describe('PackedUniformArray', () => {
    it('insertStatic appends when there is no dynamic region', () => {
        const device = createFakeGpuDevice();
        const array = makeArray(device);

        const a = array.insertStatic(fillValue(11));
        const b = array.insertStatic(fillValue(22));
        array.flush();

        expect(array.getStaticCount()).toBe(2);
        expect(array.getCount()).toBe(2);
        expect(a.index).toBe(0);
        expect(b.index).toBe(1);

        const view = readBack(device, array.getBuffer());
        expect(view.getUint32(a.index * STRIDE, true)).toBe(11);
        expect(view.getUint32(b.index * STRIDE, true)).toBe(22);
    });

    it('insertStatic displaces the first dynamic entry to the tail and preserves its data', () => {
        const device = createFakeGpuDevice();
        const array = makeArray(device);

        const a = array.insertStatic(fillValue(1));
        const b = array.insertStatic(fillValue(2));
        array.markDynamic(a); // statics: [b]; dynamic: [a]

        const c = array.insertStatic(fillValue(3));
        array.flush();

        expect(array.getStaticCount()).toBe(2); // b, c
        expect(array.getCount()).toBe(3);
        expect(a.index).toBe(2); // displaced to the very end
        expect(c.index).toBe(1); // took over the vacated boundary slot

        const view = readBack(device, array.getBuffer());
        expect(view.getUint32(a.index * STRIDE, true)).toBe(1); // a's data survived the move
        expect(view.getUint32(b.index * STRIDE, true)).toBe(2);
        expect(view.getUint32(c.index * STRIDE, true)).toBe(3);
    });

    it('markDynamic moves an entry into the dynamic region and is a no-op if already dynamic', () => {
        const device = createFakeGpuDevice();
        const array = makeArray(device);
        const a = array.insertStatic(fillValue(1));
        const b = array.insertStatic(fillValue(2));
        const c = array.insertStatic(fillValue(3));

        array.markDynamic(b);
        expect(array.getStaticCount()).toBe(2);
        expect(b.index).toBe(2); // front of the dynamic region

        const bIndexBefore = b.index;
        array.markDynamic(b); // already dynamic - idempotent no-op
        expect(b.index).toBe(bIndexBefore);
        expect(array.getStaticCount()).toBe(2);

        array.flush();
        const view = readBack(device, array.getBuffer());
        expect(view.getUint32(a.index * STRIDE, true)).toBe(1);
        expect(view.getUint32(b.index * STRIDE, true)).toBe(2);
        expect(view.getUint32(c.index * STRIDE, true)).toBe(3);
    });

    it('remove from the static region keeps both regions contiguous', () => {
        const device = createFakeGpuDevice();
        const array = makeArray(device);
        const a = array.insertStatic(fillValue(1));
        const b = array.insertStatic(fillValue(2));
        const c = array.insertStatic(fillValue(3));
        const d = array.insertStatic(fillValue(4));
        array.markDynamic(d); // statics: [a, b, c]; dynamic: [d]

        array.remove(b); // remove from the middle of statics
        array.flush();

        expect(array.getStaticCount()).toBe(2);
        expect(array.getCount()).toBe(3);

        const view = readBack(device, array.getBuffer());
        expect(a.index).toBeLessThan(array.getStaticCount());
        expect(c.index).toBeLessThan(array.getStaticCount());
        expect(view.getUint32(a.index * STRIDE, true)).toBe(1);
        expect(view.getUint32(c.index * STRIDE, true)).toBe(3);

        expect(d.index).toBeGreaterThanOrEqual(array.getStaticCount());
        expect(d.index).toBeLessThan(array.getCount());
        expect(view.getUint32(d.index * STRIDE, true)).toBe(4);
    });

    it('remove from the dynamic region keeps the dynamic region contiguous', () => {
        const device = createFakeGpuDevice();
        const array = makeArray(device);
        const a = array.insertStatic(fillValue(1));
        const b = array.insertStatic(fillValue(2));
        const c = array.insertStatic(fillValue(3));
        array.markDynamic(b);
        array.markDynamic(c); // statics: [a]; dynamic: [b, c] in some order

        array.remove(b);
        array.flush();

        expect(array.getStaticCount()).toBe(1);
        expect(array.getCount()).toBe(2);

        const view = readBack(device, array.getBuffer());
        expect(view.getUint32(a.index * STRIDE, true)).toBe(1);
        expect(c.index).toBeGreaterThanOrEqual(array.getStaticCount());
        expect(c.index).toBeLessThan(array.getCount());
        expect(view.getUint32(c.index * STRIDE, true)).toBe(3);
    });

    it('remove of the sole entry empties the array', () => {
        const device = createFakeGpuDevice();
        const array = makeArray(device);
        const a = array.insertStatic(fillValue(1));

        array.remove(a);

        expect(array.getStaticCount()).toBe(0);
        expect(array.getCount()).toBe(0);
    });

    it('buffer growth preserves existing entry data and calls onBufferReplaced', () => {
        const device = createFakeGpuDevice();
        const array = makeArray(device, 2);
        let replacedCount = 0;
        array.onBufferReplaced(() => { replacedCount++; });

        const a = array.insertStatic(fillValue(1));
        const b = array.insertStatic(fillValue(2));
        const bufferBeforeGrowth = array.getBuffer();

        const c = array.insertStatic(fillValue(3)); // exceeds initial capacity of 2
        array.flush();

        expect(replacedCount).toBe(1);
        expect(array.getBuffer()).not.toBe(bufferBeforeGrowth);

        const view = readBack(device, array.getBuffer());
        expect(view.getUint32(a.index * STRIDE, true)).toBe(1);
        expect(view.getUint32(b.index * STRIDE, true)).toBe(2);
        expect(view.getUint32(c.index * STRIDE, true)).toBe(3);
    });

    it('flush issues exactly one writeBuffer call covering everything touched since the last flush, and no-ops when nothing changed', () => {
        const device = createFakeGpuDevice();
        const array = makeArray(device);

        array.insertStatic(fillValue(1));
        array.insertStatic(fillValue(2));
        array.insertStatic(fillValue(3));
        expect(device.writeCalls).toHaveLength(0); // nothing reaches the GPU before flush()

        array.flush();
        expect(device.writeCalls).toHaveLength(1);

        device.writeCalls = [];
        array.flush(); // nothing changed since the last flush
        expect(device.writeCalls).toHaveLength(0);
    });

    it('writeEntry stages a change that is only uploaded on the next flush', () => {
        const device = createFakeGpuDevice();
        const array = makeArray(device);
        const a = array.insertStatic(fillValue(1));
        array.flush();
        device.writeCalls = [];

        array.writeEntry(a, fillValue(42));
        expect(device.writeCalls).toHaveLength(0);

        array.flush();
        expect(device.writeCalls).toHaveLength(1);
        const view = readBack(device, array.getBuffer());
        expect(view.getUint32(a.index * STRIDE, true)).toBe(42);
    });
});
