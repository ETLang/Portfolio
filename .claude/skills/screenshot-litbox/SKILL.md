---
name: screenshot-litbox
description: Render the Litbox WebGPU portfolio app in a real headless browser, capture a screenshot of the canvas, and visually inspect it. Use after a change to shaders, the denoiser, or scene/rendering code, to verify the change actually renders correctly instead of just trusting that it compiles.
---

# Screenshot-verify the Litbox renderer

Use this after touching anything that affects what appears on screen (shaders, `LitboxSceneRenderer`,
denoiser passes, scene definitions) - render it and *look* at the result instead of declaring victory
on a clean typecheck/build.

## How to run it

```
node .claude/skills/screenshot-litbox/screenshot.mjs --out <absolute-path.png> [--scene <key>] [--settle-ms 5000] [--width 1280] [--height 800]
```

- `--out` is required. **Use a native Windows path (`C:\...`), not a Git-Bash-style path
  (`/c/...`)** - the script runs as a plain Windows Node process, so a `/c/...` path gets
  reinterpreted as a relative path (`.\c\...`) instead of erroring, which silently writes the
  screenshot to the wrong place. Prefer the scratchpad directory for the output path.
- `--scene` is optional and must be a key from `SCENE_REGISTRY` in
  [litbox_scene_registry.ts](../../../src/litbox_scene_registry.ts). Omit it to just screenshot
  whatever the default scene renders. When given, the script clicks the "Litbox" nav button and
  picks the scene from the `#scene-select` dropdown, same as a user would.
- The script starts its own `npm run dev` and tears it down when it exits - it does not touch
  any dev server you already have running (Vite auto-picks a free port if 5173 is taken).
- `--settle-ms` (default 5000) is how long to let photon accumulation run before the screenshot.
  This is a Monte Carlo renderer - very short settle times will look noisier than the app looks
  in normal use. Increase it if you need to judge converged quality rather than early noise.

After it runs, **read the resulting PNG with the Read tool and actually look at it.** The script
reports `consoleIssues` (any `console.error`/uncaught page errors) in its JSON stdout, which is a
useful early check (a black canvas + console errors is over-determined), but a black canvas *with
no console errors at all* is exactly the kind of failure this project has hit before (see
CLAUDE.md's WebGPU JS-API gotchas) - only the image itself tells you if it actually rendered the
scene.

## Why this drives Edge, not Playwright's own Chromium

Playwright downloads its own Chromium build (`npx playwright install chromium`), but on this
machine that build's WebGPU support is broken: `navigator.gpu.requestAdapter()` succeeds, but
`requestDevice()` throws:

```
OperationError: Failed to execute 'requestDevice' on 'GPUAdapter': DynamicLib.Open: dxil.dll Windows Error: 87
```

`dxil.dll`/`dxcompiler.dll` (the DirectX Shader Compiler, which Dawn's D3D12 backend needs) are
present right next to `chrome.exe` in that build, but fail to load anyway - root cause not fully
isolated (not a missing-file or MOTW-block issue; ruled both out). The OS-installed Microsoft Edge
(`channel: 'msedge'` in `chromium.launch(...)`, already present on this Windows 11 machine at the
default install path) does not have this problem and gets a working WebGPU device in headless
mode. If Edge is ever removed from this machine, re-run the same adapter/device probe against
Playwright's bundled Chromium first before assuming it's fixed itself.

## Limitations - desktop-only signal

This only proves the **desktop** WebGPU path works. Per CLAUDE.md, this project has two confirmed
classes of bug that produce **zero signal on desktop** and only reproduce on real mobile GPU
drivers: WGSL dynamic-indexing corruption, and `copyExternalImageToTexture` silently leaving a
texture black on Android. A clean screenshot from this skill says nothing about either - real
device testing (see the CDP-over-adb mobile debugging setup) is still required before trusting a
shader or texture-upload change on mobile.

**This skill cannot run in a headless cloud/CI environment - do not attempt it from a `remote`
agent sandbox, GitHub Actions, or similar.** It needs a real GPU plus a working WebGPU-capable
browser on the host machine. The `msedge` workaround above only helped because this machine
already has real GPU hardware and drivers - Playwright's own Chromium build was the broken part,
not the hardware. A standard cloud VM has no GPU device at all (no `/dev/dri`, no vendor driver),
so `navigator.gpu.requestAdapter()` returns `null` unconditionally in every browser there - there
is no "try a different browser" fix for missing hardware. Only attempt this on a machine known to
have a real GPU and a browser already validated against it (see the probe history above).
