// Perceptual pixel diff between two screenshots (typically two take-screenshot outputs) using
// pixelmatch+pngjs. No daemon dependency - operates purely on two already-saved PNG files.
import pixelmatch from 'pixelmatch';
import { PNG } from 'pngjs';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
    const args = { threshold: 0.1, includeAA: false, out: path.join(__dirname, 'output', 'diff.png') };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--a') args.a = argv[++i];
        else if (a === '--b') args.b = argv[++i];
        else if (a === '--out') args.out = argv[++i];
        else if (a === '--threshold') args.threshold = Number(argv[++i]);
        else if (a === '--include-aa') args.includeAA = true;
        else throw new Error(`Unknown argument: ${a}`);
    }
    if (!args.a || !args.b) throw new Error('--a and --b are both required');
    return args;
}

function main() {
    const args = parseArgs(process.argv.slice(2));
    const imgA = PNG.sync.read(readFileSync(args.a));
    const imgB = PNG.sync.read(readFileSync(args.b));

    if (imgA.width !== imgB.width || imgA.height !== imgB.height) {
        // Hard-fail rather than auto-resize: a dimension mismatch is most likely two screenshots
        // taken at different --width/--height, which is the actual bug worth surfacing, not
        // something to paper over by rescaling.
        throw new Error(
            `Dimension mismatch: ${args.a} is ${imgA.width}x${imgA.height}, ${args.b} is ${imgB.width}x${imgB.height} - `
            + 'screenshots must be taken at the same --width/--height to diff them.',
        );
    }

    const { width, height } = imgA;
    const diffImg = new PNG({ width, height });
    const diffPixels = pixelmatch(imgA.data, imgB.data, diffImg.data, width, height, {
        threshold: args.threshold,
        includeAA: args.includeAA,
    });

    mkdirSync(path.dirname(args.out), { recursive: true });
    writeFileSync(args.out, PNG.sync.write(diffImg));

    const totalPixels = width * height;
    console.log(JSON.stringify({
        ok: true,
        diffPixels,
        totalPixels,
        diffPercent: (diffPixels / totalPixels) * 100,
        width,
        height,
        out: args.out,
    }));
}

try {
    main();
} catch (err) {
    console.error(JSON.stringify({ ok: false, error: String(err && err.message || err) }));
    process.exit(1);
}
