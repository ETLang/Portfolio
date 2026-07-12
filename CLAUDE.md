# Project notes for Claude Code

## Unity project gotchas

- The Unity Litbox project contains a significant amount of vestigial code that was
  used for past experiments that didn't work out. Not all of it has been removed, 
  so do not assume that merely because the Unity version does something, that the
  WebGPU version must also do the same thing. 

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

## WebGPU JS-API gotchas (mobile)

- **Never use `GPUQueue.copyExternalImageToTexture` to upload an `ImageBitmap` on mobile.**
  Confirmed on a Pixel 10 Pro XL (Imagination PowerVR GPU, Chrome for Android) that it
  silently leaves the destination texture black/empty - texture creation succeeds, the copy
  call raises no validation error or exception, and `textureSample` just reads back zero
  everywhere. Works fine on desktop (Windows/D3D12), which is exactly why this class of bug
  is dangerous - it produces zero signal in desktop-only testing, same as the WGSL indexing
  gotcha above. Root-caused by isolating it in a standalone test page
  (`copyExternalImageToTexture` vs `writeTexture`, same decoded pixels, side by side) since
  there was no error to trace from.
  - Fix: decode the `ImageBitmap` to raw RGBA bytes via a 2D canvas (`drawImage` +
    `getImageData`), then upload with `device.queue.writeTexture(...)` instead. See
    `src/litbox/texture_cache.ts`'s `loadImageTexture`.
  - `writeTexture` doesn't need `GPUTextureUsage.RENDER_ATTACHMENT` on the destination
    texture; `copyExternalImageToTexture` does (Chrome enforces this - it's implemented as
    an internal blit). Drop the flag when migrating a texture off the old path.
  - Any future image-upload code path needs the same real-mobile-hardware confirmation
    before being trusted - this bug and the WGSL one above are both invisible without it.

## Compute-shader operation architecture

The raytracing simulation's compute passes (photon emission/tracing/accumulation, etc. - see
`SimulationResources.run()` in `src/litbox/simulation.ts`, currently a stub) will each be a
small subclass of a shared abstract `ComputeOperation` base (planned:
`src/litbox/compute_operation.ts` - build it when the first real compute pass needs it, not
before). Goal: whatever orchestrates a sequence of these passes reads as a plumbing-free list
of steps, with all bind group/pipeline/dispatch mechanics hidden inside the operation:

```ts
if (uniformsDirty) { op.updateUniforms(...); }
if (buffersDirty) {
    op.updateInputs(...);
    op.updateOutputs(...);
}
op.execute(encoder);
```

- **One subclass per operation, not one generic config-driven class.** Each operation's
  inputs/outputs/uniform struct are bespoke enough that a fully generic class would just
  relocate the same complexity into a config object instead of removing it. The shared
  `ComputeOperation` base should own only what's genuinely identical across every operation:
  compute pipeline creation, per-group bind-group caching/dirty-tracking, and dispatch math.
  Leave everything operation-specific to the subclass.
- **Bind groups are fixed by convention: group 0 = uniforms, group 1 = inputs, group 2 =
  outputs.** Each gets its own independent dirty flag, lazily rebuilt on the next
  `execute()` - mirror `RaytracedResources`' `sharedBindGroupDirty`/`rebuildSharedBindGroup()`
  pattern (`src/litbox/raytraced_resources.ts:105-130,574-589`).
- **`updateInputs`/`updateOutputs` each take bespoke, named parameters for exactly the
  buffers/textures that operation reads/writes** - never a generic `entries:
  GPUBindGroupEntry[]` bag. `updateUniforms` takes a single struct parameter describing all
  of that operation's uniform values (never raw bytes/`BufferSource` at the call site). For
  example:
  ```ts
  class ClearLightmapOperation extends ComputeOperation {
      public updateUniforms(uniforms: ClearLightmapUniforms): void { ... }
      public updateInputs(density: ComputedTexture, normalRoughness: ComputedTexture): void { ... }
      public updateOutputs(lightmap: ComputedTexture): void { ... }
  }
  ```
  Each method translates its typed parameters into whatever the base class needs
  internally (bind group entries / raw bytes) and hands them to protected base helpers (e.g.
  `this.setInputs([...])`) that do the actual dirty-tracking and byte-writing.
- **Workgroup size is parsed from the shader's own `@workgroup_size(...)` attribute**
  (a small regex over the WGSL source, done once by the base class), never duplicated as a
  separate JS constant that could silently drift from the shader.
- **Each operation has exactly one uniform struct, not an array of them.** Unlike
  `PackedUniformArray` (built for an arbitrary, growable count of per-object entries), there
  are only a handful of simulation operations - no indexing, no byte offsets. `updateUniforms`
  always writes to offset 0 of a single buffer, sized lazily from whatever's first written to
  it - no separately-declared size constant to keep in sync with the shader's struct.
- **Dispatch size is computed by the operation itself, not passed in by the caller.**
  `execute(encoder)` takes no width/height/item-count parameters - each subclass knows its own
  dispatch extent from whatever it was last given via `updateOutputs` (e.g. an output
  `ComputedTexture`'s width/height) and reports it internally before the base class dispatches.
- **Samplers are internal to the operation, never caller-configurable.** An operation creates
  whatever sampler(s) it needs (a fixed filter/wrap mode, inherent to what the shader does)
  itself, at construction - never accepts one through `updateInputs`.
- **Compile-time switches (WGSL sections enabled/disabled at compile time, e.g. via
  `override` constants) go through their own method, `updateSwitches(...)`** - same
  bespoke-named-parameters shape as `updateInputs`/`updateOutputs`, never a generic flags
  bag. Unlike uniforms/inputs/outputs, a switch change invalidates the compute *pipeline*
  itself, not just a bind group, since it can change which shader code actually gets
  compiled/specialized - so `updateSwitches` marks its own `pipelineDirty` flag, checked and
  rebuilt by `execute()` before dispatch, the same lazy-rebuild pattern used for the three
  bind groups.
