// Crops and nearest-neighbor-zooms a region of a PNG (typically screenshot.mjs's own output) for
// closer visual inspection. Uses Playwright's own bundled Chromium (not the msedge channel
// screenshot.mjs needs) since this only touches a 2D canvas - no WebGPU device involved, so none of
// that skill's dxil.dll workaround is needed here.
//
// Output path is fixed (like screenshot.mjs's own render.png) so the same literal command line
// works on every invocation regardless of which region is being cropped - only the numeric args
// change, which keeps this allowlist-able in permission settings.
import { chromium } from 'playwright';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
    const args = { scale: 3, out: path.join(__dirname, 'output', 'crop.png') };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--in') args.in = argv[++i];
        else if (a === '--x') args.x = Number(argv[++i]);
        else if (a === '--y') args.y = Number(argv[++i]);
        else if (a === '--w') args.w = Number(argv[++i]);
        else if (a === '--h') args.h = Number(argv[++i]);
        else if (a === '--scale') args.scale = Number(argv[++i]);
        else if (a === '--out') args.out = argv[++i];
        else throw new Error(`Unknown argument: ${a}`);
    }
    for (const required of ['in', 'x', 'y', 'w', 'h']) {
        if (args[required] === undefined) throw new Error(`--${required} is required`);
    }
    return args;
}

async function main() {
    const args = parseArgs(process.argv.slice(2));
    const imageBuffer = readFileSync(args.in);
    const base64 = imageBuffer.toString('base64');

    const browser = await chromium.launch({ headless: true });
    try {
        const page = await browser.newPage();
        const outW = Math.round(args.w * args.scale);
        const outH = Math.round(args.h * args.scale);
        await page.setViewportSize({ width: outW, height: outH });
        await page.setContent(`
            <html><body style="margin:0;padding:0;">
            <canvas id="c" width="${outW}" height="${outH}"></canvas>
            <script>
                window.__done = false;
                const img = new Image();
                img.onload = () => {
                    const ctx = document.getElementById('c').getContext('2d');
                    ctx.imageSmoothingEnabled = false;
                    ctx.drawImage(img, ${args.x}, ${args.y}, ${args.w}, ${args.h}, 0, 0, ${outW}, ${outH});
                    window.__done = true;
                };
                img.src = "data:image/png;base64,${base64}";
            </script>
            </body></html>
        `);
        await page.waitForFunction(() => window.__done, { timeout: 10000 });
        await page.locator('canvas').screenshot({ path: args.out });
        console.log(JSON.stringify({ ok: true, out: args.out }));
    } finally {
        await browser.close();
    }
}

main().catch((err) => {
    console.error(JSON.stringify({ ok: false, error: String(err && err.stack || err) }));
    process.exit(1);
});
