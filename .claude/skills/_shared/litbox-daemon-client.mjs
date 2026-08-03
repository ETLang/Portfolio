// Shared client for talking to the start-server daemon (see ../start-server/daemon.mjs). Every
// consumer skill (select-scene, select-debug-view, set-config-property, take-screenshot) imports
// this instead of touching Playwright/the state file directly, so "how do I find/trust the running
// daemon" lives in exactly one place.
import { readFileSync, unlinkSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

// The daemon's own state file lives next to daemon.mjs, not next to this shared file - every
// consumer resolves it the same way (relative to _shared/, i.e. sideways into start-server/state)
// so each worktree/checkout of this repo naturally finds its OWN daemon (see SKILL.md's "why no
// fixed port" - a fixed port would let a second worktree's client mistake a first worktree's
// daemon for its own).
const STATE_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'start-server', 'state', 'daemon-state.json');

export function bestEffortKill(pid) {
    if (!pid) return;
    try {
        if (process.platform === 'win32') {
            spawnSync('taskkill', ['/pid', String(pid), '/T', '/F']);
        } else {
            process.kill(pid, 'SIGKILL');
        }
    } catch {
        // Already gone, or never existed - fine either way, this is just cleanup.
    }
}

export async function healthCheck(port) {
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 2000);
        const res = await fetch(`http://127.0.0.1:${port}/health`, { signal: controller.signal });
        clearTimeout(timeout);
        if (!res.ok) return null;
        return await res.json();
    } catch {
        return null;
    }
}

/**
 * Reads the daemon's state file and confirms it's actually alive via /health (not just present -
 * a crashed/killed daemon can leave a stale file behind). Returns null, and cleans up the stale
 * state (deletes the file, best-effort kills the recorded pid), if no healthy daemon is found -
 * every caller goes through this so staleness is only ever handled in this one spot.
 */
export async function readState() {
    let state;
    try {
        state = JSON.parse(readFileSync(STATE_PATH, 'utf-8'));
    } catch {
        return null;
    }
    const health = await healthCheck(state.port);
    if (!health) {
        try { unlinkSync(STATE_PATH); } catch { /* already gone */ }
        bestEffortKill(state.pid);
        return null;
    }
    return state;
}

export function statePath() {
    return STATE_PATH;
}

/**
 * POSTs a command to the running daemon. Throws a clear, actionable error (rather than a raw
 * fetch/connection error) if no healthy daemon is found, since that's the single most likely
 * mistake a caller will make (forgetting to run Start-Server first).
 */
export async function callDaemon(pathname, body) {
    const state = await readState();
    if (!state) {
        throw new Error('No running Litbox automation server found. Run the start-server skill first.');
    }
    const res = await fetch(`http://127.0.0.1:${state.port}${pathname}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body ?? {}),
    });
    const json = await res.json();
    if (!res.ok || json.ok === false) {
        const err = new Error(json.error || `Daemon request to ${pathname} failed with status ${res.status}`);
        err.validOptions = json.validOptions;
        throw err;
    }
    return json;
}
