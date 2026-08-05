// Client for the start-server daemon's /shutdown endpoint - see ../start-server/SKILL.md for how
// the daemon works. Complements the daemon's own idle-timeout self-shutdown: this exists for a
// caller that knows it's done and wants the headless browser (and the GPU/CPU load that comes with
// it) gone immediately, rather than waiting out the idle timer (up to --idle-timeout-ms, 10 minutes
// by default).
import { readState, statePath, bestEffortKill, healthCheck } from '../_shared/litbox-daemon-client.mjs';
import { unlinkSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';

async function main() {
    // readState() itself already confirms the daemon is alive via /health (and cleans up a
    // stale/crashed one) - nothing running means there's nothing to do, and that's success, not an
    // error, same idempotent philosophy as start-server.mjs's alreadyRunning check.
    const state = await readState();
    if (!state) {
        console.log(JSON.stringify({ ok: true, alreadyStopped: true }));
        return;
    }

    try {
        await fetch(`http://127.0.0.1:${state.port}/shutdown`, { method: 'POST' });
    } catch {
        // A failed request here is handled the same as an unconfirmed shutdown below - the
        // polling/fallback logic covers it either way.
    }

    // /shutdown responds before the daemon actually finishes tearing down (browser.close() plus the
    // Vite taskkill /T teardown take a moment) - poll for it to actually go away instead of trusting
    // the response alone.
    let stopped = false;
    for (let i = 0; i < 20; i++) {
        await sleep(250);
        if (!(await healthCheck(state.port))) {
            stopped = true;
            break;
        }
    }

    if (!stopped) {
        // The daemon didn't confirm its own shutdown within ~5s - fall back to the same
        // best-effort taskkill /T the daemon-client already uses for a stale/crashed daemon, so
        // this skill still succeeds even if the graceful path hung.
        bestEffortKill(state.pid);
        try { unlinkSync(statePath()); } catch { /* already gone */ }
    }

    console.log(JSON.stringify({ ok: true, alreadyStopped: false, forcedKill: !stopped }));
}

main().catch((err) => {
    console.error(JSON.stringify({ ok: false, error: String(err && err.message || err) }));
    // exitCode, not exit() - see set-config-property.mjs's identical comment: exit() right after an
    // in-flight fetch() can crash with a libuv assertion on Windows.
    process.exitCode = 1;
});
