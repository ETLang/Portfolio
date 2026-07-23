// TEMP investigation script - zoomed crop of the canvas at a fixed region, for inspecting
// fine-grained denoiser artifacts (banding/Moire) that a full-scene screenshot compresses away.
// Same drive-Edge approach as screenshot.mjs. Delete after use.
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { mkdirSync } from 'node:fs';
import path from 'node:path';

const OUT = path.join(path.dirname(fileURLToPath(import.meta.url)), 'output', 'zoom.png');
const SCENE = process.argv[2] || 'cornell-square';
const SETTLE_MS = Number(process.argv[3] || 15000);
const W = Number(process.argv[4] || 3200);
const H = Number(process.argv[5] || 2000);
const CLIP = { x: Number(process.argv[6]), y: Number(process.argv[7]), width: Number(process.argv[8]), height: Number(process.argv[9]) };

function startDevServer() {
    return new Promise((resolve, reject) => {
        const child = spawn('npm', ['run', 'dev'], { shell: true });
        let resolved = false;
        let buffer = '';
        const onData = (data) => {
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

async function main() {
    const { child, url } = await startDevServer();
    try {
        const browser = await chromium.launch({ headless: true, channel: 'msedge', args: ['--enable-unsafe-webgpu'] });
        try {
            const page = await browser.newPage({ ignoreHTTPSErrors: true, viewport: { width: W, height: H } });
            const consoleIssues = [];
            page.on('pageerror', (e) => consoleIssues.push(String(e)));
            page.on('console', (m) => { if (m.type() === 'error') consoleIssues.push(m.text()); });

            await page.goto(url, { waitUntil: 'load' });
            await page.waitForFunction(() => !!(window).litboxRenderer, { timeout: 15000 });

            await page.click('[data-view="litbox"]');
            await page.waitForSelector('#scene-select');
            await page.selectOption('#scene-select', SCENE);
            await sleep(500);

            await sleep(SETTLE_MS);
            mkdirSync(path.dirname(OUT), { recursive: true });
            // page.screenshot's `clip` is in page (viewport) coordinates, not element-relative -
            // locator.screenshot() doesn't support `clip` at all, so offset by the canvas's own
            // on-page bounding box first.
            let clipOpt = {};
            if (Number.isFinite(CLIP.x)) {
                const box = await page.locator('canvas').boundingBox();
                clipOpt = { clip: { x: box.x + CLIP.x, y: box.y + CLIP.y, width: CLIP.width, height: CLIP.height } };
            }
            await page.screenshot({ path: OUT, ...clipOpt });
            console.log(JSON.stringify({ ok: true, out: OUT, consoleIssues }));
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
