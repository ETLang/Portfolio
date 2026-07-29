// Thin client for the start-server daemon's /screenshot endpoint.
import { callDaemon } from '../_shared/litbox-daemon-client.mjs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const DEFAULT_OUT = path.join(path.dirname(fileURLToPath(import.meta.url)), 'output', 'render.png');

function parseArgs(argv) {
    const args = { out: DEFAULT_OUT, settleMs: 5000 };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--out') args.out = argv[++i];
        else if (a === '--settle-ms') args.settleMs = Number(argv[++i]);
        else if (a === '--width') args.width = Number(argv[++i]);
        else if (a === '--height') args.height = Number(argv[++i]);
        else throw new Error(`Unknown argument: ${a}`);
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
