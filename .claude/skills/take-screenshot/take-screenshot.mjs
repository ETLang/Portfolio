// Thin client for the start-server daemon's /screenshot endpoint.
import { callDaemon } from '../_shared/litbox-daemon-client.mjs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const OUTPUT_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'output');

// Default filename is timestamped (down to the millisecond), not a fixed name - screenshots
// accumulate in OUTPUT_DIR across a session instead of each call silently overwriting the last
// one (that was the old `render.png`-always behavior; multi-shot comparisons had to hand-roll a
// distinct --out path, which is exactly how a mangled Git-Bash-style path once leaked a stray file
// into the repo root - see clear-screenshots for the cleanup counterpart to this). No colons (not
// valid in a Windows filename).
function timestampedFilename() {
    const now = new Date();
    const pad = (n, width = 2) => String(n).padStart(width, '0');
    return `render-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}`
        + `-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}-${pad(now.getMilliseconds(), 3)}.png`;
}

function parseArgs(argv) {
    const args = { settleMs: 5000 };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--out') args.out = argv[++i];
        else if (a === '--settle-ms') args.settleMs = Number(argv[++i]);
        else if (a === '--width') args.width = Number(argv[++i]);
        else if (a === '--height') args.height = Number(argv[++i]);
        else throw new Error(`Unknown argument: ${a}`);
    }
    if (!args.out) {
        args.out = path.join(OUTPUT_DIR, timestampedFilename());
    }
    return args;
}

async function main() {
    const args = parseArgs(process.argv.slice(2));
    const result = await callDaemon('/screenshot', args);
    console.log(JSON.stringify(result));
}

main().catch((err) => {
    console.error(JSON.stringify({ ok: false, error: String(err && err.message || err) }));
    // exitCode, not exit() - see set-config-property.mjs's identical comment: exit() right after an
    // in-flight fetch() can crash with a libuv assertion on Windows.
    process.exitCode = 1;
});
