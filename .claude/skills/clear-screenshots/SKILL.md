---
name: clear-screenshots
description: Delete every file in take-screenshot's output folder. No server required - a pure filesystem cleanup. Use once a batch of Litbox visual work is done and the accumulated screenshots aren't needed anymore, so they don't pile up indefinitely.
---

# Clear the take-screenshot output folder

```
node .claude/skills/clear-screenshots/clear-screenshots.mjs
```

- Deletes every file inside `.claude/skills/take-screenshot/output/` - the folder
  [take-screenshot](../take-screenshot/SKILL.md) writes each timestamped screenshot into. Deletes
  only the folder's *contents*, never the folder itself and nothing outside it - the path is
  hardcoded, not caller-supplied, so this can't be pointed at anything else.
- Does **not** require [start-server](../start-server/SKILL.md) to be running - this is a plain
  filesystem operation, independent of the daemon/browser, and works even after the daemon has
  idled out.
- Takes no arguments. Safe to call even if the folder is already empty or doesn't exist yet
  (`deletedCount: 0` either way).
- Only clears `take-screenshot`'s own output folder - `crop-zoom` and `diff-screenshots` each keep
  their own separate `output/` folders (their own crops/diffs) and are unaffected.
