---
name: diff-screenshots
description: Compute a perceptual pixel diff between two screenshot PNGs and produce a visual diff-highlight image. No server required - operates purely on two already-saved image files. Use to compare a before/after render, or to check whether a change actually altered the output.
---

# Diff two screenshots

```
node .claude/skills/diff-screenshots/diff-screenshots.mjs --a <path.png> --b <path.png> [--out <path.png>] [--threshold 0.1] [--include-aa]
```

- No dependency on start-server - operates purely on two already-saved PNG files, using
  `pixelmatch`+`pngjs`.
- **Hard-fails on a width/height mismatch between `--a` and `--b`** rather than auto-resizing - a
  dimension mismatch almost always means the two screenshots were taken with different
  `--width`/`--height`, which is the actual bug worth surfacing, not something to paper over.
- `--threshold` (pixelmatch's own option, 0-1, default `0.1`) is how different a pixel pair must be
  to count as changed. This project's renderer is Monte Carlo, so two screenshots of the identical
  scene/settings will still show some noise-driven pixel drift even with nothing actually changed -
  see "Limitations" below for the measured noise floor before assuming a nonzero `diffPercent` means
  something real changed.
- `--include-aa` includes anti-aliased edge pixels in the diff (pixelmatch excludes them by default,
  which reduces false positives from subpixel jitter unrelated to Monte Carlo noise).
- Reports `{diffPixels, totalPixels, diffPercent, width, height, out}`. This is a **binary
  per-pixel threshold count, not a continuous magnitude-of-difference metric** - for "how different,
  and where," read the diff-highlight PNG at `--out` (default
  `.claude/skills/diff-screenshots/output/diff.png`), same "look at the image" philosophy as the
  other skills in this library.

## Limitations

Same-scene noise floor, measured: two `take-screenshot` calls back-to-back against the same running
`battle` scene (default `--settle-ms 5000`, default `--threshold 0.1`) produced `diffPercent` ≈
**0.46%** (2794/605900 pixels), concentrated in the beam and haze regions where Monte Carlo noise is
naturally highest - the flat dark background differed almost nowhere. Treat `diffPercent` well below
roughly 1% as "no meaningful change" for this renderer; a value well above that on a comparison where
you expect no change is a real signal, not noise. This floor was measured on continuously-accumulating
live renders (the daemon keeps refining the image between calls), not a restarted render, which is
the realistic way this skill gets used in practice.
