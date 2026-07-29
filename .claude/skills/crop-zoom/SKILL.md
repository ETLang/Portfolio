---
name: crop-zoom
description: Crop a region out of an existing screenshot PNG and nearest-neighbor-zoom it for closer inspection. No server required - operates purely on an already-saved image file. Use after take-screenshot when a detail is too small to judge at full-image scale.
---

# Crop and zoom a screenshot

```
node .claude/skills/crop-zoom/crop-zoom.mjs --in <path.png> --x <n> --y <n> --w <n> --h <n> [--scale 3] [--out <path.png>]
```

- No dependency on start-server - this only touches an already-saved PNG file with Playwright's own
  bundled Chromium (plain 2D canvas, no WebGPU, so none of take-screenshot's msedge workaround is
  needed here).
- `--x --y --w --h` define the crop rectangle in the *source* image's pixel coordinates (e.g. from
  take-screenshot's output).
- `--scale` (default 3) is the nearest-neighbor zoom factor applied after cropping - output image is
  `(w*scale) x (h*scale)`. Nearest-neighbor, not smoothed, so individual source pixels stay legible
  at high zoom rather than blurring together.
- Output always lands at `.claude/skills/crop-zoom/output/crop.png` unless `--out` overrides it -
  fixed path, overwritten each run, same allowlist-ability rationale as the other skills here.
