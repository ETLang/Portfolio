// The long-lived process behind the start-server skill: owns one Vite dev server + one headless
// Edge WebGPU browser + one page for as long as anything needs it, and exposes that page to every
// other litbox skill (select-scene, select-debug-view, set-config-property, take-screenshot) over a
// tiny localhost-only HTTP server. Spawned detached by start-server.mjs, which exits immediately
// after this process reports healthy - this process itself keeps running (self-shuts-down on an
// idle timer) rather than being tied to the lifetime of whichever tool call started it.
//
// The msedge WebGPU launch is ported from the old screenshot-litbox skill's screenshot.mjs verbatim
// - see that file's history for why msedge (not Playwright's own bundled Chromium, which fails
// requestDevice() on this machine). Dev-server start/teardown started from that same script too but
// spawns Vite's own JS entry directly rather than `npm run dev` - see startDevServer()'s comment.
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { mkdirSync, writeFileSync, unlinkSync } from 'node:fs';
import path from 'node:path';
import { parsePropertyArg } from '../_shared/config-property-schema.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const STATE_DIR = path.join(__dirname, 'state');
const STATE_PATH = path.join(STATE_DIR, 'daemon-state.json');
mkdirSync(STATE_DIR, { recursive: true });

function log(...args) {
    // Deliberately not stdio:'ignore' from the spawning side (see start-server.mjs) - this ends up
    // in state/daemon.log, the only diagnostic trail available if this process crashes after its
    // parent tool call has already exited.
    console.log(new Date().toISOString(), ...args);
}

function parseArgs(argv) {
    const args = { idleTimeoutMs: 10 * 60 * 1000, width: 1280, height: 800 };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--idle-timeout-ms') args.idleTimeoutMs = Number(argv[++i]);
        else if (a === '--width') args.width = Number(argv[++i]);
        else if (a === '--height') args.height = Number(argv[++i]);
        else throw new Error(`Unknown argument: ${a}`);
    }
    return args;
}

const REPO_ROOT = path.join(__dirname, '..', '..', '..');

function startDevServer() {
    return new Promise((resolve, reject) => {
        // Spawn Vite's own JS entry directly - NOT `npm run dev` via shell:true. That route wraps
        // through cmd.exe/npm.cmd, and that wrapper layer can exit before vite.js itself does,
        // orphaning it past whatever PID stopDevServer's taskkill /T was given (observed: a daemon
        // shut down cleanly per its own log, but Vite kept running - taskkill /T could no longer
        // find it once the wrapper had already exited). Spawning vite.js directly keeps a single
        // parent-child hop that taskkill /T can reliably walk.
        const viteBin = path.join(REPO_ROOT, 'node_modules', 'vite', 'bin', 'vite.js');
        const child = spawn(process.execPath, [viteBin], { cwd: REPO_ROOT });
        let resolved = false;
        let buffer = '';
        const onData = (data) => {
            // eslint-disable-next-line no-control-regex
            buffer += data.toString().replace(/\x1b\[[0-9;]*m/g, '');
            const match = buffer.match(/Local:\s+(https:\/\/localhost:\d+\/[^\s]*)/);
            if (match && !resolved) {
                resolved = true;
                resolve({ child, url: match[1] });
            }
        };
        child.stdout.on('data', onData);
        child.stderr.on('data', onData);
        child.on('error', reject);
        setTimeout(() => {
            if (!resolved) reject(new Error('Timed out waiting for Vite dev server to start:\n' + buffer));
        }, 20000);
    });
}

function stopDevServer(child) {
    if (process.platform === 'win32') {
        return new Promise((resolve) => {
            const killer = spawn('taskkill', ['/pid', String(child.pid), '/T', '/F']);
            killer.on('exit', resolve);
            killer.on('error', resolve);
        });
    }
    child.kill('SIGTERM');
    return Promise.resolve();
}

// Serializes every mutating request against the one shared `page` - two overlapping handlers
// (e.g. a scene switch still in flight when a screenshot request lands) would otherwise be a real,
// cheap-to-hit bug since Playwright doesn't serialize concurrent calls against the same page itself.
function makeMutex() {
    let tail = Promise.resolve();
    return (fn) => {
        const result = tail.then(fn, fn);
        tail = result.then(() => {}, () => {});
        return result;
    };
}

async function main() {
    const args = parseArgs(process.argv.slice(2));
    log('starting dev server...');
    const { child: viteChild, url } = await startDevServer();
    log('dev server ready at', url);

    const browser = await chromium.launch({ headless: true, channel: 'msedge', args: ['--enable-unsafe-webgpu'] });
    const page = await browser.newPage({
        ignoreHTTPSErrors: true,
        viewport: { width: args.width, height: args.height },
    });
    const consoleIssues = [];
    page.on('pageerror', (e) => consoleIssues.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') consoleIssues.push(m.text()); });

    await page.goto(url, { waitUntil: 'load' });
    await page.waitForFunction(() => !!(window).litboxRenderer, { timeout: 15000 });
    log('litboxRenderer ready');

    const withPage = makeMutex();
    let lastRequestAt = Date.now();
    let shuttingDown = false;

    async function shutdown(reason) {
        if (shuttingDown) return;
        shuttingDown = true;
        log('shutting down:', reason);
        try { unlinkSync(STATE_PATH); } catch { /* already gone */ }
        try { await browser.close(); } catch { /* best effort */ }
        try { await stopDevServer(viteChild); } catch { /* best effort */ }
        process.exit(0);
    }

    // Independent of the idle timer below - a bug in the idle-shutdown logic itself shouldn't be
    // able to leak this process (and its Vite/Edge children) forever.
    const MAX_LIFETIME_MS = 6 * 60 * 60 * 1000;
    setTimeout(() => shutdown('max lifetime reached'), MAX_LIFETIME_MS);

    setInterval(() => {
        if (Date.now() - lastRequestAt > args.idleTimeoutMs) {
            shutdown(`idle for over ${args.idleTimeoutMs}ms`);
        }
    }, 30000);

    async function selectScene(sceneKey) {
        await page.click('[data-view="litbox"]');
        await page.waitForSelector('#scene-select');
        await page.selectOption('#scene-select', sceneKey);
        await sleep(500);
    }

    async function selectDebugView({ view, scale, mipLevel, solidColor }) {
        await page.evaluate(({ view, scale, mipLevel, solidColor }) => {
            const renderer = window.litboxRenderer;
            renderer.debugView = view;
            if (scale !== undefined) renderer.debugViewScale = scale;
            if (mipLevel !== undefined) renderer.debugViewMipLevel = mipLevel;
            if (solidColor !== undefined) renderer.debugSolidColor = solidColor;
        }, { view, scale, mipLevel, solidColor });
    }

    async function setConfigProperty({ property, value }) {
        // Authoritative parse/validation happens here, not on the client - a network caller's own
        // pre-validation (set-config-property.mjs does the same parse for a fast local error) must
        // never be trusted as the real gate.
        const command = parsePropertyArg(property, value);
        if (command.kind === 'exposure') {
            await page.evaluate((value) => { window.litboxRenderer.exposureOverride = value; }, command.value);
            return { property: 'exposure', value: command.value };
        }
        if (command.kind === 'tonemap') {
            await page.evaluate((value) => { window.litboxRenderer.tonemapEnabled = value; }, command.value);
            return { property: 'tonemap', value: command.value };
        }
        if (command.kind === 'denoiser') {
            await page.evaluate((value) => { window.litboxRenderer.getSimulationResources().denoiserEnabled = value; }, command.value);
            return { property: 'denoiser', value: command.value };
        }
        if (command.kind === 'denoiser-tunable') {
            await page.evaluate(({ key, value }) => {
                window.litboxRenderer.getSimulationResources().denoiserTunables[key] = value;
            }, { key: command.key, value: command.value });
            return { property: `denoiser.${command.key}`, value: command.value };
        }
        if (command.kind === 'scene-slider') {
            const result = await page.evaluate(({ label, value }) => {
                const scene = window.litboxRenderer.getActiveScene();
                if (!scene) return { ok: false, error: 'No active scene.' };
                const sliders = scene.getSliders();
                const slider = sliders.find((s) => s.label === label);
                if (!slider) {
                    return { ok: false, error: `No slider named "${label}" on the active scene.`, validOptions: sliders.map((s) => s.label) };
                }
                if (value < slider.min || value > slider.max) {
                    return { ok: false, error: `${label} must be between ${slider.min} and ${slider.max}, got ${value}.` };
                }
                slider.setValue(value);
                return { ok: true };
            }, { label: command.label, value: command.value });
            if (!result.ok) {
                const err = new Error(result.error);
                err.validOptions = result.validOptions;
                throw err;
            }
            return { property: `scene-slider.${command.label}`, value: command.value };
        }
        throw new Error(`Unhandled command kind: ${command.kind}`);
    }

    async function takeScreenshot({ out, settleMs, width, height }) {
        if (width && height) {
            await page.setViewportSize({ width, height });
        }
        await sleep(settleMs);
        mkdirSync(path.dirname(out), { recursive: true });
        await page.locator('canvas').screenshot({ path: out });
        return { out };
    }

    function readBody(req) {
        return new Promise((resolve, reject) => {
            let data = '';
            req.on('data', (chunk) => { data += chunk; });
            req.on('end', () => {
                try { resolve(data ? JSON.parse(data) : {}); } catch (err) { reject(err); }
            });
            req.on('error', reject);
        });
    }

    const server = createServer(async (req, res) => {
        lastRequestAt = Date.now();
        try {
            if (req.method === 'GET' && req.url === '/health') {
                res.writeHead(200, { 'content-type': 'application/json' });
                res.end(JSON.stringify({ ok: true, url, pid: process.pid }));
                return;
            }

            const body = await readBody(req);
            let result;
            if (req.method === 'POST' && req.url === '/select-scene') {
                result = await withPage(() => selectScene(body.scene));
            } else if (req.method === 'POST' && req.url === '/select-debug-view') {
                result = await withPage(() => selectDebugView(body));
            } else if (req.method === 'POST' && req.url === '/set-config-property') {
                result = await withPage(() => setConfigProperty(body));
            } else if (req.method === 'POST' && req.url === '/screenshot') {
                result = await withPage(() => takeScreenshot(body));
            } else {
                res.writeHead(404, { 'content-type': 'application/json' });
                res.end(JSON.stringify({ ok: false, error: `No such endpoint: ${req.method} ${req.url}` }));
                return;
            }
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(JSON.stringify({ ok: true, ...result, consoleIssues }));
        } catch (err) {
            log('request failed:', err);
            res.writeHead(400, { 'content-type': 'application/json' });
            res.end(JSON.stringify({ ok: false, error: String(err && err.message || err), validOptions: err && err.validOptions }));
        }
    });

    // Loopback-only and OS-assigned port - see litbox-daemon-client.mjs for why a fixed port would
    // be wrong (worktree collisions).
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = server.address().port;
    writeFileSync(STATE_PATH, JSON.stringify({ pid: process.pid, port, url, startedAt: new Date().toISOString() }, null, 2));
    log('daemon listening on port', port);

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));
}

main().catch((err) => {
    log('fatal:', err && err.stack || err);
    process.exit(1);
});
