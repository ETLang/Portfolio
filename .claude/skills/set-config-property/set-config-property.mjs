// Thin client for the start-server daemon's /set-config-property endpoint. Does a fast local
// pre-validation pass against the same whitelist the daemon enforces authoritatively (see
// _shared/config-property-schema.mjs's own comment for why both sides parse) so a bad --property
// name fails immediately with a helpful message instead of round-tripping to the daemon first.
import { callDaemon } from '../_shared/litbox-daemon-client.mjs';
import { parsePropertyArg } from '../_shared/config-property-schema.mjs';

function parseArgs(argv) {
    const args = {};
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--property') args.property = argv[++i];
        else if (a === '--value') args.value = argv[++i];
        else throw new Error(`Unknown argument: ${a}`);
    }
    if (!args.property || args.value === undefined) throw new Error('--property and --value are both required');
    return args;
}

async function main() {
    const args = parseArgs(process.argv.slice(2));

    // Local pre-check only - not trusted as the real gate, see daemon.mjs's setConfigProperty().
    if (!args.property.startsWith('scene-slider.')) {
        parsePropertyArg(args.property, args.value);
    }

    const result = await callDaemon('/set-config-property', { property: args.property, value: args.value });
    console.log(JSON.stringify(result));
}

main().catch((err) => {
    console.error(JSON.stringify({ ok: false, error: String(err && err.message || err), validOptions: err && err.validOptions }));
    // exitCode, not exit() - calling exit() right after an in-flight fetch() (undici) can abort its
    // internal handles mid-teardown and crash with a libuv assertion on Windows. Setting exitCode
    // lets the event loop drain those handles naturally before the process exits.
    process.exitCode = 1;
});
