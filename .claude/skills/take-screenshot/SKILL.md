---
name: take-screenshot
description: Screenshot the current state of the running Litbox app's canvas. Requires the start-server skill to have been run first. Use after select-scene/select-debug-view/set-config-property to see the effect, or any time you want to visually verify the current render.
---

# Screenshot the Litbox canvas

```
node .claude/skills/take-screenshot/take-screenshot.mjs [--out <path>] [--settle-ms 5000] [--width 1280] [--height 800]
```

- Requires [start-server](../start-server/SKILL.md) to already be running.
- The screenshot always lands at `.claude/skills/take-screenshot/output/render.png` unless `--out`
  overrides it - a fixed path, overwritten on every run, so the same literal command line can be
  allowlisted once. `--out`, if given, must be a native Windows path (`C:\...`), not a Git-Bash-style
  path (`/c/...`) - this is a plain Windows Node process, so a `/c/...` path silently gets
  reinterpreted as relative instead of erroring.
- **No `--scene` flag** - that's [select-scene](../select-scene/SKILL.md)'s job now. Run that first
  if you need a specific scene; this skill only screenshots whatever's currently loaded.
- `--settle-ms` (default 5000) is how long to let photon accumulation run before capturing. This is
  a Monte Carlo renderer - short settle times look noisier than the app looks in normal use.
- `--width`/`--height`, if given, resize the browser's viewport before capturing (persists for
  subsequent screenshots too, since the browser/page stay open across calls).
- After it runs, **read the resulting PNG with the Read tool and actually look at it.** The response
  includes `consoleIssues` (console errors / uncaught page errors accumulated since the server
  started), which is a useful early check, but a black canvas with *no* console errors is exactly the
  kind of failure this project has hit before (see CLAUDE.md's WebGPU JS-API gotchas) - only the
  image itself tells you if it actually rendered.

## Limitations - desktop-only signal

Same caveat as the retired screenshot-litbox skill this replaces: this only proves the desktop
WebGPU path works. This project has confirmed bug classes (WGSL dynamic-indexing corruption,
`copyExternalImageToTexture` silently failing) that produce zero signal on desktop and only reproduce
on real mobile GPU drivers - a clean screenshot here says nothing about either. This also cannot run
in a headless cloud/CI environment - it needs a real GPU and a working WebGPU browser on the host.
