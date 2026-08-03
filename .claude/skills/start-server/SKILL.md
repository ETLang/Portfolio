---
name: start-server
description: Start the persistent Litbox automation server - a headless Edge browser with WebGPU rendering the app, backed by its own independent dev server - that the select-scene, select-debug-view, set-config-property, and take-screenshot skills all act on. Run this once before using any of those; it stays running in the background and self-shuts-down after a period of inactivity, or can be stopped immediately with the stop-server skill.
---

# Start the Litbox automation server

This is the entry point for the atomized Litbox skill library (select-scene, select-debug-view,
set-config-property, take-screenshot, crop-zoom, diff-screenshots). Run it once at the start of a
session of Litbox visual work; the other skills will fail with a clear error telling you to run this
first if you skip it or if the server has since idled out.

## How to run it

```
node .claude/skills/start-server/start-server.mjs [--idle-timeout-ms 600000] [--width 1280] [--height 800]
```

- **Idempotent - safe to call redundantly.** If a healthy server is already running, this returns
  immediately with `{ok: true, alreadyRunning: true, port, viteUrl}` instead of starting a second
  one. Call it freely before any sequence of the other skills rather than trying to track server
  state yourself.
- `--width`/`--height` only take effect on a fresh start (they set the browser's initial viewport);
  they're ignored on an `alreadyRunning: true` response. `take-screenshot` can still change the
  viewport per-shot later.
- `--idle-timeout-ms` (default 10 minutes) only takes effect on a fresh start too.

## What this actually does

Spawns a detached background process (`daemon.mjs`) that:
1. Starts its own independent Vite dev server (spawned directly via `node_modules/vite/bin/vite.js`,
   not `npm run dev` - the npm/cmd.exe wrapper that shell:true would introduce breaks Windows
   process-tree teardown, see daemon.mjs's `startDevServer()`. Vite auto-picks a free port, so this
   never collides with a dev server you're already running yourself for other purposes).
2. Launches headless Microsoft Edge with WebGPU enabled (`channel: 'msedge'` - Playwright's own
   bundled Chromium fails `requestDevice()` on this machine, see the retired screenshot-litbox
   skill's history) and opens one page, waiting for `window.litboxRenderer` to exist.
3. Starts a small HTTP server bound to `127.0.0.1` on an OS-assigned port (not a fixed port - see
   "why no fixed port" below) that the other skills send commands to.
4. Writes `state/daemon-state.json` (pid, port, dev server URL) once healthy, then keeps running
   until either the idle timeout elapses with no requests, or a hard 6-hour lifetime cap is hit as
   a safety net independent of the idle timer.

This process is deliberately **not** tied to the lifetime of the tool call that started it - it has
to survive across every separate `node ...` invocation the other skills make. It cleans itself up on
the idle timer by default, or can be stopped immediately with [stop-server](../stop-server/SKILL.md)
once you're done with a batch of Litbox visual work and don't want it competing for the GPU any
longer than necessary.

## Why no fixed port

A fixed port would let a second checkout of this repo (e.g. a git worktree) falsely detect the
*first* checkout's daemon as "already running" and silently act on the wrong code. Instead every
consumer skill locates its own daemon via `state/daemon-state.json`, which lives next to this
skill's own files - each checkout naturally finds only its own daemon.

## If something goes wrong

Check `.claude/skills/start-server/state/daemon.log` - the daemon logs there (not to a discarded
stdio stream) specifically so a crash after this script has already exited is still diagnosable.
