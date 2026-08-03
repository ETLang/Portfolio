# Stop the Litbox automation server

Stops the daemon started by [start-server](../start-server/SKILL.md) immediately, instead of
waiting for its idle timeout. Use this once you're done with a batch of Litbox visual work (screenshots/
scene switches/debug views/config tweaks) and don't need the headless browser hanging around anymore -
it's a real Edge process doing real WebGPU rendering, and it competes for the same GPU as anything
else running on the machine (including a human tester's own browser tab).

## How to run it

```
node .claude/skills/stop-server/stop-server.mjs
```

- **Idempotent - safe to call even if nothing is running.** If there's no healthy daemon (never
  started, already idled out, already stopped), this returns `{ok: true, alreadyStopped: true}`
  immediately rather than erroring.
- Otherwise it asks the daemon to shut down gracefully (closes the browser, tears down the Vite dev
  server via the same process-tree-aware teardown `start-server` already uses) and polls briefly to
  confirm it actually went away. If the daemon doesn't confirm within ~5 seconds, this falls back to
  a forced kill of the recorded PID (`forcedKill: true` in the response) so the call still succeeds
  either way.
- Cleans up `start-server/state/daemon-state.json` itself - after this returns, `start-server` will
  start a fresh daemon on next use rather than mistaking anything for still-running.

## When to call this vs. just leaving it

The daemon self-shuts-down on an idle timer by design (see `start-server/SKILL.md`), so calling this
is never required for correctness - only for freeing the GPU/CPU sooner than the idle timeout would.
Call it once you're confident you're done with Litbox visual work for the current task, not after
every individual screenshot/scene-switch (the other skills all reuse the same daemon across calls;
stopping it between every call would just force a slow restart on the next one).
