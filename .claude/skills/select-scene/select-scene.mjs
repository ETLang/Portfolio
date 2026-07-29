// Thin client for the start-server daemon's /select-scene endpoint - see start-server/SKILL.md for
// how the daemon works. This script holds no Playwright/browser state of its own.
import { callDaemon } from '../_shared/litbox-daemon-client.mjs';

function parseArgs(argv) {
    const args = {};
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--scene') args.scene = argv[++i];
        else throw new Error(`Unknown argument: ${a}`);
    }
    if (!args.scene) throw new Error('--scene is required');
    return args;
}

async function main() {
    const args = parseArgs(process.argv.slice(2));
    const result = await callDaemon('/select-scene', { scene: args.scene });
    console.log(JSON.stringify(result));
}

main().catch((err) => {
    console.error(JSON.stringify({ ok: false, error: String(err && err.message || err), validOptions: err && err.validOptions }));
    // exitCode, not exit() - see set-config-property.mjs's identical comment: exit() right after an
    // in-flight fetch() can crash with a libuv assertion on Windows.
    process.exitCode = 1;
});
