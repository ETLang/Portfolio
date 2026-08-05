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

A `Stop` hook (`.claude/settings.json`) already calls this automatically at the end of every Claude
Code turn, so the server never sits idle on the GPU waiting for a human to notice - you shouldn't
normally need to call this yourself. Manual calls are still useful mid-turn if you're confident a
whole batch of Litbox visual work (screenshots/scene switches/debug views/config tweaks) is done and
want the GPU freed before the turn ends, or when debugging the server itself. The daemon's own idle
timer (30s, see `start-server/SKILL.md`) is a backstop for cases the hook can't reach (a
crashed/killed session), not the primary teardown path anymore.
