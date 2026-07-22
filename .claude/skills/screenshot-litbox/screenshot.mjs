// Renders the Litbox WebGPU app in a real (headless) browser and saves a screenshot of the
// canvas. See SKILL.md in this directory for why this drives the OS-installed Edge instead of
// Playwright's own downloaded Chromium, and for the desktop-only limitation of this approach.
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { mkdirSync } from 'node:fs';
import path from 'node:path';

// Fixed, always-overwritten default location (not a caller-supplied path) so the invocation is
// identical on every run - see SKILL.md's "Why the output path is fixed" for why that matters.
const DEFAULT_OUT = path.join(path.dirname(fileURLToPath(import.meta.url)), 'output', 'render.png');

function parseArgs(argv) {
    const args = { settleMs: 5000, width: 1280, height: 800, out: DEFAULT_OUT };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--scene') args.scene = argv[++i];
        else if (a === '--out') args.out = argv[++i];
        else if (a === '--settle-ms') args.settleMs = Number(argv[++i]);
        else if (a === '--width') args.width = Number(argv[++i]);
        else if (a === '--height') args.height = Number(argv[++i]);
        else throw new Error(`Unknown argument: ${a}`);
    }
    return args;
}

function startDevServer() {
    return new Promise((resolve, reject) => {
        // shell:true is required on Windows to resolve npm.cmd; safe here since no part of this
        // command line is derived from user input or CLI args.
        const child = spawn('npm', ['run', 'dev'], { shell: true });
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
    // Vite spawns its own child processes (esbuild optimizer, etc.) that a plain SIGTERM to the
    // `npm` shell wrapper won't reliably reach on Windows - taskkill /T walks the whole tree.
    // Awaited by the caller so the process is actually gone before this script exits - firing
    // taskkill without waiting let Node's event loop end before it completed, leaking dev servers.
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

async function main() {
    const args = parseArgs(process.argv.slice(2));
    const { child, url } = await startDevServer();

    try {
        // Playwright's own downloaded Chromium build fails to get a WebGPU device on this
        // machine (Dawn's D3D12 backend can't load dxil.dll - see SKILL.md). The OS-installed
        // Edge does not have this problem, so drive that instead via channel: 'msedge'.
        const browser = await chromium.launch({ headless: true, channel: 'msedge', args: ['--enable-unsafe-webgpu'] });
        try {
            const page = await browser.newPage({
                ignoreHTTPSErrors: true, // dev server uses a self-signed mkcert cert
                viewport: { width: args.width, height: args.height },
            });
            const consoleIssues = [];
            page.on('pageerror', (e) => consoleIssues.push(String(e)));
            page.on('console', (m) => { if (m.type() === 'error') consoleIssues.push(m.text()); });

            await page.goto(url, { waitUntil: 'load' });
            await page.waitForFunction(() => !!(window).litboxRenderer, { timeout: 15000 });

            if (args.scene) {
                await page.click('[data-view="litbox"]');
                await page.waitForSelector('#scene-select');
                await page.selectOption('#scene-select', args.scene);
                // Let the new scene's first frames land before settling further below.
                await sleep(500);
            }

            await sleep(args.settleMs);
            mkdirSync(path.dirname(args.out), { recursive: true });
            await page.locator('canvas').screenshot({ path: args.out });

            console.log(JSON.stringify({ ok: true, out: args.out, consoleIssues }));
        } finally {
            await browser.close();
        }
    } finally {
        await stopDevServer(child);
    }
}

main().catch((err) => {
    console.error(JSON.stringify({ ok: false, error: String(err && err.stack || err) }));
    process.exit(1);
});
