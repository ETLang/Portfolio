// Idempotent entry point: ensures the daemon (daemon.mjs) is running and the Litbox app inside it
// is ready, then exits - the daemon itself keeps running detached. See SKILL.md for the full
// picture of why this is split from the rest of the skills the way it is.
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { mkdirSync, writeFileSync, unlinkSync, openSync, statSync } from 'node:fs';
import path from 'node:path';
import { readState } from '../_shared/litbox-daemon-client.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const STATE_DIR = path.join(__dirname, 'state');
const LOCK_PATH = path.join(STATE_DIR, 'starting.lock');
const LOG_PATH = path.join(STATE_DIR, 'daemon.log');
const LOCK_STALE_MS = 60000; // generously longer than the daemon's own startup timeout below

function parseArgs(argv) {
    const args = { idleTimeoutMs: undefined, width: undefined, height: undefined };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--idle-timeout-ms') args.idleTimeoutMs = argv[++i];
        else if (a === '--width') args.width = argv[++i];
        else if (a === '--height') args.height = argv[++i];
        else throw new Error(`Unknown argument: ${a}`);
    }
    return args;
}

function acquireStartLock() {
    mkdirSync(STATE_DIR, { recursive: true });
    try {
        writeFileSync(LOCK_PATH, JSON.stringify({ pid: process.pid, at: Date.now() }), { flag: 'wx' });
        return true;
    } catch (err) {
        if (err.code !== 'EEXIST') throw err;
        try {
            const age = Date.now() - statSync(LOCK_PATH).mtimeMs;
            if (age > LOCK_STALE_MS) {
                // Abandoned lock from a crashed/killed prior attempt - safe to take over.
                unlinkSync(LOCK_PATH);
                return acquireStartLock();
            }
        } catch { /* lock disappeared between checks - fine, treat as not ours */ }
        return false;
    }
}

async function waitUntilHealthy(timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const state = await readState();
        if (state) return state;
        await sleep(1000);
    }
    return null;
}

async function main() {
    const args = parseArgs(process.argv.slice(2));

    const existing = await readState();
    if (existing) {
        console.log(JSON.stringify({ ok: true, alreadyRunning: true, port: existing.port, viteUrl: existing.url }));
        return;
    }

    const gotLock = acquireStartLock();
    if (!gotLock) {
        // Someone else (another near-simultaneous start-server invocation) is already starting the
        // daemon - wait for them instead of racing a second one into existence.
        const state = await waitUntilHealthy(60000);
        if (!state) throw new Error('Another start-server invocation appears stuck starting the daemon (lock held, but no healthy daemon appeared in time).');
        console.log(JSON.stringify({ ok: true, alreadyRunning: true, port: state.port, viteUrl: state.url }));
        return;
    }

    try {
        const daemonScript = path.join(__dirname, 'daemon.mjs');
        const daemonArgs = [daemonScript];
        if (args.idleTimeoutMs) daemonArgs.push('--idle-timeout-ms', args.idleTimeoutMs);
        if (args.width) daemonArgs.push('--width', args.width);
        if (args.height) daemonArgs.push('--height', args.height);

        // Not stdio:'ignore' - redirected to a log file so a post-detach crash is diagnosable (see
        // daemon.mjs's own log() comment). detached+unref was empirically confirmed to survive this
        // parent process exiting on this machine before this skill was built on top of it.
        const logFd = openSync(LOG_PATH, 'a');
        const child = spawn(process.execPath, daemonArgs, { detached: true, stdio: ['ignore', logFd, logFd] });
        child.unref();

        // Vite startup + msedge launch + first render can genuinely take a while - generous timeout.
        const state = await waitUntilHealthy(60000);
        if (!state) {
            throw new Error(`Daemon did not become healthy in time - check ${LOG_PATH} for details.`);
        }
        console.log(JSON.stringify({ ok: true, alreadyRunning: false, port: state.port, viteUrl: state.url }));
    } finally {
        try { unlinkSync(LOCK_PATH); } catch { /* already gone */ }
    }
}

main().catch((err) => {
    console.error(JSON.stringify({ ok: false, error: String(err && err.stack || err) }));
    // exitCode, not exit() - see set-config-property.mjs's identical comment: exit() right after an
    // in-flight fetch() (readState()'s health check) can crash with a libuv assertion on Windows.
    process.exitCode = 1;
});
