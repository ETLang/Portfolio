---
name: set-config-property
description: Set one Litbox renderer/denoiser/scene-slider config value to a new value, by name, on the running app. Requires the start-server skill to have been run first. Use to tune exposure, tonemap/denoiser toggles, individual denoiser tunables, or a per-scene slider without going through the UI.
---

# Set a Litbox config property

```
node .claude/skills/set-config-property/set-config-property.mjs --property <name> --value <v>
```

- Requires [start-server](../start-server/SKILL.md) to already be running.
- `<name>` must be one of the whitelisted properties below - not arbitrary JS, so a typo fails
  loudly rather than silently doing nothing. The whitelist is enforced on both this script (a fast
  local error) and the server (the authoritative check - never trust the client-side one alone).

| `--property` | Value | Notes |
|---|---|---|
| `exposure` | number, or `auto`/`null` | Sets `exposureOverride`. `auto`/`null` clears the override back to automatic - the only way to undo an exposure override in this skill library. |
| `tonemap` | `true`/`false`/`1`/`0` | Tonemap enabled. |
| `denoiser` | `true`/`false`/`1`/`0` | Denoiser enabled. |
| `denoiser.<key>` | number | One of the 17 denoiser tunables - see [denoiser_tunables_panel.ts](../../../src/denoiser_tunables_panel.ts) for the full list and their ranges (`varianceScale`, `darknessNoiseFloor`, `maxBlurMip`, `albedoSensitivity`, `densitySensitivity`, `normalSensitivity`, `sigmaLuminanceTight`, `sigmaLuminanceLoose`, `kLuminance`, `maxSplitDistance`, `detailMaxSplitDistance`, `albedoLuminanceThreshold`, `albedoChromaThreshold`, `logDensityThreshold`, `volatilityThreshold`, `detailThreshold`, `varianceGateScale`). |
| `scene-slider.<label>` | number | A slider on the *currently active* scene (see select-scene), matched by its exact label. Out-of-range values are **rejected**, not clamped. An unknown label's error response lists the active scene's real slider labels. |

- **`DENSITY_SCALE` is deliberately NOT settable here.** It's a compile-time WGSL constant
  duplicated manually between `LitboxCommon.wgsl` and `RaytracedResources.DENSITY_SCALE` (see this
  project's CLAUDE.md) - there is no runtime property for it.
- **`debugView`/`debugViewScale`/`debugViewMipLevel`/`debugSolidColor` are NOT here either** - those
  belong to [select-debug-view](../select-debug-view/SKILL.md), since they're meaningless without an
  active debug view.
- Does not take a screenshot itself - follow with [take-screenshot](../take-screenshot/SKILL.md) to
  see the effect.
