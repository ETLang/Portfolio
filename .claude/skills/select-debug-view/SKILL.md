---
name: select-debug-view
description: Enable (or clear) one of the Litbox renderer's debug visualizations - G-Buffer channels, the raw lightmap, or denoiser evidence textures. Requires the start-server skill to have been run first. Use to inspect an intermediate rendering/denoising stage instead of the final composited image.
---

# Select a Litbox debug view

```
node .claude/skills/select-debug-view/select-debug-view.mjs --view <key|none> [--scale <n>] [--mip-level <n>] [--solid-color]
```

- Requires [start-server](../start-server/SKILL.md) to already be running.
- `--view none` clears the debug view (reverts to normal composited rendering) - this is the only
  way to undo a debug view within this skill library, since there's no generic reset skill.
- Valid `<key>` values, set on `litboxRenderer.debugView`
  ([litbox_scene_renderer.ts](../../../src/litbox_scene_renderer.ts)):
  - G-Buffer channels: `albedo`, `density`, `normal`, `roughness`
  - `lightmap` - the raw HDR lightmap, pre-tonemap/pre-exposure
  - Denoiser pipeline evidence: `irradiance-a`, `irradiance-b`, `combined-irradiance`,
    `raw-variance`, `filtered-variance` - see this project's CLAUDE.md denoiser-architecture section
    for what each of these represents.
  - `blur-size` - decideBlurSize's continuous per-pixel result (denoise.wgsl), i.e. the actual
    "how much blur" decision at each pixel - lets you see that decision directly instead of
    inferring it from how noisy/smooth the final composited image looks. `--scale` should
    typically be set near `maxBlurMip` (~6 by default) to use the full displayable range, since
    that's this value's own natural ceiling.
- **An unrecognized `--view` key does not error** - the renderer silently falls back to normal
  rendering. This script prints a warning to stderr if you pass something outside the known list
  above, but double-check spelling regardless; a clean-looking screenshot after a typo'd view name
  is a false negative, not confirmation the view exists.
- `--scale <n>` sets `debugViewScale` (default 0.5 in the renderer) - consumed by `density` and
  every HDR-scaled view (`lightmap`, `irradiance-a`/`irradiance-b`, `combined-irradiance`,
  `raw-variance`/`filtered-variance`, `blur-size`); ignored by the rest.
- `--mip-level <n>` sets `debugViewMipLevel` (default 0) - which mip of the active view's source
  texture to display. Out-of-range values are clamped harmlessly by the GPU, not rejected.
- `--solid-color` / `--no-solid-color` toggles `debugSolidColor` (flat shape-colored quads, bypassing
  opacity/shading) - independent of which debug view (if any) is active.
- Does not take a screenshot itself - follow with [take-screenshot](../take-screenshot/SKILL.md).
