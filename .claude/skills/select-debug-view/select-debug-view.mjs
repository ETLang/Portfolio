// Thin client for the start-server daemon's /select-debug-view endpoint.
import { callDaemon } from '../_shared/litbox-daemon-client.mjs';

const VALID_VIEWS = [
    'albedo', 'density', 'normal', 'roughness', 'lightmap',
    'irradiance-a', 'irradiance-b', 'combined-irradiance', 'raw-variance', 'filtered-variance',
];

function parseArgs(argv) {
    const args = {};
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--view') args.view = argv[++i];
        else if (a === '--scale') args.scale = Number(argv[++i]);
        else if (a === '--mip-level') args.mipLevel = Number(argv[++i]);
        else if (a === '--solid-color') args.solidColor = true;
        else if (a === '--no-solid-color') args.solidColor = false;
        else throw new Error(`Unknown argument: ${a}`);
    }
    if (!args.view) throw new Error('--view is required (a debug view key, or "none" to clear it)');
    return args;
}

async function main() {
    const args = parseArgs(process.argv.slice(2));
    const view = args.view === 'none' ? null : args.view;
    if (view !== null && !VALID_VIEWS.includes(view)) {
        // Not fatal - the renderer silently falls back to normal rendering on an unknown key rather
        // than throwing - but warn loudly since that silent fallback is exactly the kind of footgun
        // this project's own docs call out.
        console.error(`Warning: "${view}" is not a known debug view (${VALID_VIEWS.join(', ')}). The renderer will silently fall back to normal rendering rather than erroring.`);
    }
    const result = await callDaemon('/select-debug-view', { view, scale: args.scale, mipLevel: args.mipLevel, solidColor: args.solidColor });
    console.log(JSON.stringify(result));
}

main().catch((err) => {
    console.error(JSON.stringify({ ok: false, error: String(err && err.message || err) }));
    // exitCode, not exit() - see set-config-property.mjs's identical comment: exit() right after an
    // in-flight fetch() can crash with a libuv assertion on Windows.
    process.exitCode = 1;
});
