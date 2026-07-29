---
name: select-scene
description: Switch the running Litbox app to a different scene, by key. Requires the start-server skill to have been run first. Use after start-server and before take-screenshot when you need to look at a specific scene rather than whatever's currently loaded.
---

# Select a Litbox scene

```
node .claude/skills/select-scene/select-scene.mjs --scene <key>
```

- Requires [start-server](../start-server/SKILL.md) to already be running - this fails with a clear
  error telling you to run it first if not.
- `<key>` must be a key from `SCENE_REGISTRY` in
  [litbox_scene_registry.ts](../../../src/litbox_scene_registry.ts) (currently `cornell-square`,
  `basic`, `battle` - check that file for the current list, it changes as scenes are added).
- Drives the same UI path a real user would (clicks the "Litbox" nav tab, picks the option from the
  `#scene-select` dropdown) rather than calling the renderer's `setScene()` directly, so it exercises
  the actual production code path.
- Does not take a screenshot itself - follow with [take-screenshot](../take-screenshot/SKILL.md).
