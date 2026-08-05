// Deletes every file inside take-screenshot's own output/ folder - the cleanup counterpart to
// take-screenshot now defaulting to a fresh timestamped filename per call (see its own SKILL.md)
// instead of a single always-overwritten render.png. Pure filesystem operation - no daemon/browser
// dependency at all, unlike every other Litbox skill in this library, so this works even if
// start-server was never run (or has already idled out).
import { readdirSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

// Deliberately hardcoded to take-screenshot's own output/ folder, not a caller-supplied path -
// this skill's whole purpose is a single, narrowly-scoped, always-safe cleanup, not a general
// "rm -rf whatever you tell me" tool.
const OUTPUT_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'take-screenshot', 'output');

function main() {
    let entries;
    try {
        entries = readdirSync(OUTPUT_DIR);
    } catch (err) {
        if (err.code === 'ENOENT') {
            console.log(JSON.stringify({ ok: true, deletedCount: 0 }));
            return;
        }
        throw err;
    }

    for (const name of entries) {
        rmSync(path.join(OUTPUT_DIR, name), { recursive: true, force: true });
    }
    console.log(JSON.stringify({ ok: true, deletedCount: entries.length, deleted: entries }));
}

try {
    main();
} catch (err) {
    console.error(JSON.stringify({ ok: false, error: String(err && err.message || err) }));
    process.exitCode = 1;
}
