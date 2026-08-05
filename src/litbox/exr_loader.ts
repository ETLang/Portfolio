// Minimal OpenEXR reader for the scene exporter's own atlas output (see
// TextureAtlasBaker.cs's RFloat/RgbaFloat buckets, written via
// Texture2D.EncodeToEXR(EXRFlags.CompressZIP | EXRFlags.OutputAsFloat)). This is not a
// general-purpose EXR reader - it only supports what that one producer ever emits:
// single-part scanline (no tiles, no deep data, no multipart), ZIP compression, four
// FLOAT channels named R/G/B/A. Anything else throws, since a mismatch here means the
// exporter and this loader have drifted apart, not that some other legitimate EXR
// variant showed up.
//
// Row 0 of the returned pixels is the top of the image, matching every other path
// through TextureCache (createImageBitmap-decoded formats are always top-down) - no
// extra flip needed here, unlike the .bc1 loader's block-row reversal, because Unity's
// EncodeToEXR already writes scanlines top-down.

const MAGIC = 0x01312f76;
const ZIP_COMPRESSION = 3;
const FLOAT_PIXEL_TYPE = 2;
const SCANLINES_PER_ZIP_BLOCK = 16;
const CHANNEL_SLOT: Record<string, number> = { R: 0, G: 1, B: 2, A: 3 };

export interface DecodedExr {
    width: number;
    height: number;
    /** RGBA float32, interleaved, row-major, row 0 = top. */
    pixels: Float32Array;
}

export async function decodeExr(buffer: ArrayBuffer): Promise<DecodedExr> {
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== MAGIC) {
        throw new Error('not an OpenEXR file (bad magic number)');
    }
    const versionField = view.getUint32(4, true);
    const version = versionField & 0xff;
    const flags = versionField & ~0xff;
    if (version !== 2 || flags !== 0) {
        throw new Error(`unsupported EXR version/flags 0x${versionField.toString(16)} (only single-part scanline images are supported)`);
    }

    let pos = 8;
    const readCString = (): string => {
        const start = pos;
        while (view.getUint8(pos) !== 0) pos++;
        const str = new TextDecoder('ascii').decode(new Uint8Array(buffer, start, pos - start));
        pos++;
        return str;
    };

    let channelNames: string[] | null = null;
    let compression: number | null = null;
    let width = 0;
    let height = 0;

    while (true) {
        const name = readCString();
        if (name === '') break;
        const type = readCString();
        const size = view.getInt32(pos, true);
        pos += 4;
        const attrStart = pos;

        if (name === 'channels' && type === 'chlist') {
            const names: string[] = [];
            while (view.getUint8(pos) !== 0) {
                const cStart = pos;
                while (view.getUint8(pos) !== 0) pos++;
                names.push(new TextDecoder('ascii').decode(new Uint8Array(buffer, cStart, pos - cStart)));
                pos++; // channel name's terminator
                const pixelType = view.getInt32(pos, true);
                pos += 4 + 4 + 4 + 4; // pixelType (already read) + pLinear/reserved(4) + xSampling(4) + ySampling(4)
                if (pixelType !== FLOAT_PIXEL_TYPE) {
                    throw new Error(`EXR channel "${names[names.length - 1]}" is not FLOAT (pixelType ${pixelType})`);
                }
            }
            pos++; // chlist terminator (empty name)
            channelNames = names;
        } else if (name === 'compression' && type === 'compression') {
            compression = view.getUint8(pos);
        } else if (name === 'dataWindow' && type === 'box2i') {
            const xMin = view.getInt32(pos, true);
            const yMin = view.getInt32(pos + 4, true);
            const xMax = view.getInt32(pos + 8, true);
            const yMax = view.getInt32(pos + 12, true);
            width = xMax - xMin + 1;
            height = yMax - yMin + 1;
        }

        pos = attrStart + size;
    }

    if (!channelNames) throw new Error('EXR file has no "channels" attribute');
    if (compression === null) throw new Error('EXR file has no "compression" attribute');
    if (width === 0 || height === 0) throw new Error('EXR file has no "dataWindow" attribute');
    if (compression !== ZIP_COMPRESSION) {
        throw new Error(`unsupported EXR compression type ${compression} (only ZIP is supported)`);
    }
    if (channelNames.length !== 4 || channelNames.some(n => !(n in CHANNEL_SLOT))) {
        throw new Error(`expected exactly channels R/G/B/A, got [${channelNames.join(', ')}]`);
    }

    const numBlocks = Math.ceil(height / SCANLINES_PER_ZIP_BLOCK);
    pos += numBlocks * 8; // skip the scanline offset table; chunks are read sequentially below

    const pixels = new Float32Array(width * height * 4);

    for (let block = 0; block < numBlocks; block++) {
        const y = view.getInt32(pos, true);
        pos += 4;
        const size = view.getInt32(pos, true);
        pos += 4;
        const compressed = new Uint8Array(buffer.slice(pos, pos + size));
        pos += size;

        const rows = Math.min(SCANLINES_PER_ZIP_BLOCK, height - y);
        const raw = await inflateAndUnfilter(compressed);
        const rawView = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);

        // Channel-major layout within the block: for each channel (header declaration order),
        // `rows` scanlines of `width` FLOATs each, contiguous.
        const floatsPerChannel = rows * width;
        for (let ci = 0; ci < channelNames.length; ci++) {
            const slot = CHANNEL_SLOT[channelNames[ci]];
            const channelByteOffset = ci * floatsPerChannel * 4;
            for (let r = 0; r < rows; r++) {
                const destRowOffset = (y + r) * width * 4;
                const srcRowOffset = channelByteOffset + r * width * 4;
                for (let x = 0; x < width; x++) {
                    pixels[destRowOffset + x * 4 + slot] = rawView.getFloat32(srcRowOffset + x * 4, true);
                }
            }
        }
    }

    return { width, height, pixels };
}

/**
 * Reverses OpenEXR's ZIP compression: zlib inflate, then undo the predictor and
 * byte-interleave passes the encoder applies before deflating (see upstream OpenEXR's
 * ImfZip.cpp - `compression` here is zlib/RFC 1950 framing, i.e. Web Compression
 * Streams' "deflate", not "deflate-raw").
 */
async function inflateAndUnfilter(compressed: Uint8Array<ArrayBuffer>): Promise<Uint8Array> {
    const decompressor = new DecompressionStream('deflate');
    const writer = decompressor.writable.getWriter();
    void writer.write(compressed);
    void writer.close();
    const inflated = new Uint8Array(await new Response(decompressor.readable).arrayBuffer());

    // Predictor: undo the running byte delta (each byte was stored as its difference from
    // the previous output byte, offset by 128 to stay unsigned).
    for (let i = 1; i < inflated.length; i++) {
        inflated[i] = inflated[i - 1] + inflated[i] - 128;
    }

    // Un-interleave: the encoder split the buffer into two halves and alternated bytes from
    // each into the compressed stream (t1 = buffer[0::2], t2 = buffer[1::2]).
    const out = new Uint8Array(inflated.length);
    const half = Math.ceil(inflated.length / 2);
    let s = 0;
    for (let i = 0; i < half; i++) {
        out[s++] = inflated[i];
        if (s < out.length) out[s++] = inflated[half + i];
    }
    return out;
}
