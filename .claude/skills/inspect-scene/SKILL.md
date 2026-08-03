---
name: inspect-scene
description: Inspect the structure of a scene JSON (public/scenes/*.json) - list top-level keys, dump a section, list/search object names, look up everything referencing an id, or print the parent/child tree. No server required - operates purely on the JSON file. Use instead of ad hoc node/python one-liners when checking a scene's contents.
---

# Inspect a Litbox scene JSON

```
node .claude/skills/inspect-scene/inspect-scene.mjs --scene <name-or-path> [mode]
```

- `<name-or-path>` can be a bare scene key (`battle`, `cornell_square`, `basic`), a filename
  (`battle.json`), or a full path - resolved against `public/scenes/` if it isn't found as-is.
- Exactly one mode flag (defaults to `--summary` if omitted):
  - `--keys` - top-level keys and their type/length.
  - `--summary` - like `--keys`, plus the property names on each array's first item (a quick schema
    peek).
  - `--section <key>` - pretty-prints the full contents of one top-level key (e.g. `cameras`,
    `raytraced`, `sprites`, `simulations`, `textureAtlasKeys`, `pointLights`, `laserLights`).
  - `--names [--array <key>]` - lists `.name` values from an array (default `objects`), with counts,
    sorted most-common first. Replaces the old `grep -o '"name"...' | sort | uniq -c` pattern.
  - `--id <id>` - scans every top-level array and prints anything whose `id`, `objectId`, `ownerId`,
    or `parentId` equals `<id>` - i.e. the object itself plus its children, sprites, and raytraced
    entries in one shot.
  - `--tree [--root <id>]` - prints the `objects` parent/child hierarchy (via `parentId`), indented,
    as `id<TAB>name`. `--root` starts from that node instead of every root.
  - `--find <substring> [--array <key>]` - case-insensitive substring search on `.name` across one
    array, or all arrays if `--array` is omitted.

All args are used only as property/id lookups - never evaluated - so this is safe to allowlist with
a wildcard on the args regardless of which scene/id/name is being queried.
