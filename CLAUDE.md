# Project notes for Claude Code

## WGSL / shader gotchas

- **Never dynamically index a function-local WGSL array literal**
  (`var x = array<T, N>(...); x[runtimeIndex]`), where `runtimeIndex` is anything not
  known at compile time (`vertex_index`, `local_invocation_index`, a loop variable, a
  value read from a buffer, etc.). Confirmed on a Pixel 10 Pro (both Chrome and Brave,
  so it's the shared Android GPU driver, not a browser feature) that this can silently
  corrupt geometry/output, with **zero** validation error, exception, or device loss to
  catch it. See `src/litbox/shaders/tonemap.wgsl`'s `vertex_main` for the workaround
  (branching instead of indexing) and `src/litbox_scene_renderer.ts`'s class doc comment
  for more context.
  - Applies to vertex, fragment, and compute shaders alike - the bug is in how the
    driver lowers the indexing instruction, not specific to one shader stage.
  - Prefer buffer-backed data (storage buffer, uniform buffer, or `var<workgroup>` for
    compute shared memory) for anything indexed at runtime - that goes through ordinary
    memory loads, a completely different and far more battle-tested code path.
  - For a genuinely tiny, fixed-size lookup (a handful of entries), branching/`select`
    is the safe fallback.
  - Any new shader trick like this needs to be confirmed on real mobile hardware, not
    just desktop - this class of bug produces no signal in desktop-only testing.
