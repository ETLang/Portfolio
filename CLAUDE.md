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
  catch it. See `src/litbox/shaders/LitboxCommon.wgsl`'s `fullscreenQuadPosition` for the
  workaround (branching instead of indexing) and `src/litbox_scene_renderer.ts`'s class doc
  comment for more context.
  - Applies to vertex, fragment, and compute shaders alike - the bug is in how the
    driver lowers the indexing instruction, not specific to one shader stage.
  - Prefer buffer-backed data (storage buffer, uniform buffer, or `var<workgroup>` for
    compute shared memory) for anything indexed at runtime - that goes through ordinary
    memory loads, a completely different and far more battle-tested code path.
  - For a genuinely tiny, fixed-size lookup (a handful of entries), branching/`select`
    is the safe fallback.
  - Any new shader trick like this needs to be confirmed on real mobile hardware, not
    just desktop - this class of bug produces no signal in desktop-only testing.

- WGSL has no native `#include`, `#define`, or `#ifdef`. This project emulates a minimal
  C-style subset of all three in `src/litbox/shaders/shader_preprocessor.ts`'s
  `preprocessShader(source, defines?)`, run on every shader's raw `?raw` import before the
  result reaches `createShaderModule` - see any of `tonemap.ts`, `debug_view.ts`,
  `raytraced_resources.ts`, `sprite_resources.ts`, `simulation.ts`,
  `convert_photon_irradiance_to_hdr.ts` for the pattern. See the file header for the exact
  supported directive set (`#include`, `#define`/`#undef`, `#ifdef`/`#ifndef`/`#else`/`#endif` -
  no `#elif`, no `#if <expression>`) and its limits (single-pass, non-recursive macro
  substitution; `#include` is deduped per-file automatically rather than via opt-in guards).
  Any `.wgsl` file in `src/litbox/shaders/` should open with `#include "LitboxCommon.wgsl"`
  even if it doesn't yet use anything from it, so a future shared declaration doesn't require
  retrofitting every file. `LitboxCommon.wgsl` currently holds `DENSITY_SCALE` and the
  `fullscreenQuadPosition`/`clipSpaceToUv` helpers - it's the single WGSL-side source of
  truth for both (though `DENSITY_SCALE` still needs a separate, manually kept-in-sync copy on
  the TS side, `RaytracedResources.DENSITY_SCALE`, since WGSL and TS can't share a literal
  across that language boundary). `preprocessShader`'s `defines` parameter is also how
  `ComputeOperation.updateSwitches(...)` should be implemented once it's built - see
  "Compute-shader operation architecture" below.

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
- **Compile-time switches (WGSL sections enabled/disabled at compile time) go through their
  own method, `updateSwitches(...)`** - same bespoke-named-parameters shape as
  `updateInputs`/`updateOutputs`, never a generic flags bag. Implement it via
  `shader_preprocessor.ts`'s `#define`/`#ifdef` support: translate the typed switch parameters
  into a `ShaderDefines` object and re-run `preprocessShader(rawShaderSource, defines)` to get
  the shader text for this particular switch combination (not WGSL `override` constants -
  those can only gate runtime branches with `if`, they can't omit code from compilation the way
  `#ifdef` can, e.g. dropping an entire unused binding or entry point). Unlike uniforms/inputs/
  outputs, a switch change invalidates the compute *pipeline* itself, not just a bind group,
  since it changes which shader code actually gets compiled - so `updateSwitches` marks its own
  `pipelineDirty` flag, checked and rebuilt by `execute()` before dispatch, the same lazy-rebuild
  pattern used for the three bind groups.

## Denoiser architecture and philosophy (`denoise.wgsl`)

This section **is** "this project's denoiser plan" that comments across `denoise.wgsl`,
`build_denoiser_quadtree.wgsl`, `filter_variance.wgsl`, `compute_volatility.wgsl`,
`compute_variance_and_mips.wgsl`, and `LitboxCommon.wgsl` point to - it's the one place the full
argument lives, not just those files' own inline notes.

### Pipeline

Photon tracing produces two independent noisy irradiance samples per pixel (an A/B split), never
one, because that's what makes a valid per-pixel noise estimate possible at all (see "A/B split"
below). From there, evidence accumulates in stages, each coarser and cheaper than the last, before
the per-pixel blur ever runs:

1. `compute_variance_and_mips.wgsl` - fuses A/B into `combinedIrradiance`'s mip0-2 mean pyramid and
   derives `rawVariance` (quarter-res) from that same A/B pair.
2. `filter_variance.wgsl` - bilateral-filters `rawVariance` against mip2 albedo/irradiance into
   `filteredVariance`, so one noisy pixel doesn't read as an isolated "blur here" spike.
3. `compute_volatility.wgsl` - a normal-based edge detector, computed once at full-res mip0 and
   propagated upward by max-reduction, never re-derived at coarser levels.
4. `build_denoiser_quadtree.wgsl` - bakes a min/max-range quadtree (albedo, density, volatility,
   plus an irradiance-detail trigger) into a per-mip "must split" texture at half G-Buffer
   resolution.
5. `denoise.wgsl` - the guided blur itself: `decideBlurSize` picks a per-pixel starting mip,
   `shouldSplit` (backed by step 4's baked quadtree) drives a hierarchical descent from there, and
   `decideWeight` combines whatever the descent visits into the blurred result, optionally folding
   albedo/density back in (`COMBINE_ALBEDO_DENSITY`) to produce the final lit image.

### Core philosophy: three separate questions, three separate mechanisms

The blur deliberately never answers "how much should this pixel blur" with one signal or one
mechanism. It's split into three independent questions, each backed by different evidence, because
each question has a different failure mode if it's asked with the wrong evidence:

- **How much evidence of noise is there at all, at this pixel?** (`decideBlurSize`) - a *scalar*
  question, answered from variance + local brightness.
- **Where, spatially, does refining the blur further still change the answer?** (`shouldSplit`,
  driven by the baked quadtree) - a *where* question, answered from material/normal/density range
  and an irradiance-detail trigger, because none of those individually can see every kind of
  feature that matters.
- **Given a gathered set of samples, how much should each one count?** (`decideWeight`) - a
  *weighting* question, answered from structural similarity, radiance similarity, and spatial
  falloff multiplied together, because each rejects a different, non-overlapping kind of incorrect
  bleed.

Collapsing these into a single heuristic (e.g. "blur harder near edges") would conflate "there's
real detail here" with "there's noise here" with "this sample is relevant to the center pixel" -
three claims that are frequently true independently of each other (a laser beam through uniform
haze has real detail with no material-edge signature at all; a dark corner has high-confidence
low-variance MC noise that variance itself can't see; a same-material sample two seed-radii away is
irrelevant to the average regardless of how well it resolved). Keeping the questions separate is
also what makes the quadtree's evidence *reusable*: it's baked once per frame and shared across
every pixel's independent descent, rather than re-derived per-pixel inside the blur, because a
descent through mostly-flat regions of the image would otherwise re-pay for the same "is this
region interesting" answer thousands of times over.

### The evidence, and the argument for each piece

**`decideBlurSize` - how much to blur, from `filteredVariance` and the irradiance mip chain itself:**
- `combinedIrradiance`'s own mip chain is a genuine box filter (built once in
  `compute_variance_and_mips.wgsl`), so a coarse mip already *is* the local mean - there's no need
  for a second, separate blur pass just to estimate "what does this region look like on average"
  the way the Unity reference re-derives one by hand.
- Relative variance under-reports noise in rarely-hit dark regions: both A/B half-samples can land
  near zero, making their difference deceptively small - it's a difference of two things wrong in
  the same direction, not an absolute confidence measure. Mean brightness (`darknessShortfall`)
  doesn't share that blind spot, so the two signals are combined with `max()`, not a blend:
  whichever one says "blur more" wins, since either one being right is sufficient justification to
  blur, and averaging them would let a confident "blur more" from one signal get diluted by the
  other's blind spot.

**`shouldSplit` / the baked quadtree - where further refinement can still change the result:**
- **Albedo range (luma + chroma, decorrelated)** - the direct signature of a material/color
  boundary. Luma and chroma are checked separately (not a single RGB distance) so a chroma-only
  boundary (equal brightness, different hue) isn't washed out by a luma-dominated combined metric.
- **Density range, compared as optical depth, not raw coverage** - `opticalDepth` = `-log(1 -
  density)` because density's *perceptual* effect on the image is nonlinear (going from 90% to 95%
  coverage matters far more to the rendered result than going from 10% to 15%), so a raw linear
  density-value comparison would over- or under-trigger splits depending on where in the [0,1] range
  the difference sits.
- **Volatility (normal-based edge detector)** - protects a smooth Lambertian shading gradient on a
  uniform-albedo curved surface: real, converged detail that has *no* albedo or density signature
  at all, only a normal one. Without it, a curved object with one material would get over-blurred
  into faceted-looking bands, because nothing else in the evidence set would flag "this region's
  appearance is still changing."
- **Irradiance-detail trigger (Laplacian-pyramid-style, same-uv adjacent-mip comparison)** - catches
  the complementary blind spot: a feature with *no* G-Buffer signature whatsoever (the canonical
  example is a laser beam through uniform haze - same albedo, same density, same normal on both
  sides of the beam, only the lit result differs). `combinedIrradiance`'s mip chain is a genuine box
  filter, so a real feature produces a real disagreement between adjacent mips at the same
  location, while a flat region's adjacent mips agree almost exactly - the trigger is gated by
  nearby `filteredVariance` (a higher effective threshold where variance is already high) so plain
  MC noise in a flat, under-sampled region doesn't get misread as detail.
- **Distance bias, normalized to the candidate node's own current footprint (not the fixed seed
  footprint `decideWeight`'s spatial term uses)** - as a candidate's distance from the query pixel
  grows *relative to its own size*, `decideWeight`'s `spatialWeight` term is going to discount it
  heavily no matter how accurately its fine structure resolves, so refining it further is wasted
  work. Node-relative normalization (dividing by the *current* node's own texel size, which halves
  every split) is what makes this depth-aware rather than a blunt per-branch on/off switch: a branch
  that starts near the query keeps a roughly constant distance-in-its-own-texels ratio as it
  descends (each split's children move by a fixed fraction of the *parent's* footprint, a fraction
  that doesn't grow with depth), so it's free to keep splitting as deep as the quadtree wants; a
  branch that starts far away roughly *doubles* that ratio with every split (distance stays near-
  constant while its footprint keeps halving), so it gets cut off increasingly fast the farther out
  it already was - exactly "fine detail far away doesn't matter, no matter how deep you'd have to go
  to see it," made computable per-node instead of asserted globally.
- All of the above are baked once, at half-resolution, into a shared quadtree - not recomputed per
  query pixel - specifically because the same regional evidence ("is there a material boundary
  here," "is there real irradiance detail here") is relevant to every nearby pixel's independent
  descent; amortizing it once per frame is what keeps the hierarchical descent affordable at all.

**`decideWeight` - how much a gathered sample counts, once the descent has visited it:**
- **`structuralWeight` (albedo x normal x density)** rejects cross-*material* bleed: hard edges and
  silhouettes where a neighboring sample belongs to visibly different surface. Squaring the
  albedo/density terms (`w *= w`) sharpens the falloff near the tolerance boundary rather than
  letting it fade linearly, and the normal term is specifically what protects a smooth shading
  gradient on a curved, uniform-albedo surface - the same real-but-invisible-to-albedo detail
  `shouldSplit`'s volatility channel exists to catch on the "where" side.
- **`radianceWeight`** rejects cross-*feature* bleed that `structuralWeight` cannot see at all: a
  region that's uniform in every G-Buffer channel (the laser-through-haze case again) can still
  contain a real, sharp irradiance feature that a material-only weight would happily blur across.
  Its sigma is adaptive on the *center* pixel's own variance (tight/selective where the center is
  already trustworthy, loose/permissive where it isn't), reusing `filter_variance.wgsl`'s validated
  adaptive-sigma pattern verbatim - specifically so that a region which is genuinely noisy but flat
  doesn't get its real neighbors rejected by its own Monte Carlo noise being mistaken for a feature
  boundary.
- **`spatialWeight`**, normalized by the *fixed seed* texel size (deliberately different from
  `shouldSplit`'s node-relative distance metric) - this term answers "how far is this sample from
  the query, relative to the whole blur kernel's radius," a question about the final weighted
  average, not "is it worth resolving this branch further," which is a question that needs to scale
  with how deep the branch already is. Using the same metric for both would either make the split
  cutoff too aggressive near the seed or too permissive far from it.
- **`nodeSize` (`4^-depth-below-seed`)**, a partition-of-unity area weight, is multiplied in rather
  than stored per-node, because it's a pure function of `(startMip - node.mip)`: it guarantees a
  region that fully resolves down to mip 0 contributes the same total weight mass to the average as
  it would have unsplit, so a heavily-detailed (hence heavily-split, hence many-stack-entries)
  region doesn't numerically dominate the average simply by producing more terms in the sum.
- The three weight terms are **multiplied together, not summed or blended**, because each one is a
  veto for a distinct, non-overlapping failure mode (wrong material, wrong feature, wrong location)
  - a sample only deserves weight if it survives *all three* tests, not merely one of them.

**Correctness fix vs. the Unity reference: stack-overflow leaves resolve at their current mip
instead of dropping energy.** When the traversal stack has no room left to push four children
(`stackCount + 4 > STACK_SIZE`), the Unity reference falls through to leaf code that only
accumulates at mip 0, silently discarding that branch's energy entirely. Resolving it as a leaf at
its *current* (coarser) mip instead bounds the worst case to a quality degradation (a locally
blurrier estimate), never an energy loss (a locally dark/wrong estimate) - the same
partition-of-unity `nodeSize` weighting keeps its contribution correctly scaled even though it
stopped short of full resolution.

**`COMBINE_ALBEDO_DENSITY` multiplies by the *center* pixel's own albedo/density, never a per-sample
or blurred value**, because albedo/density describe the material property *at the output pixel*,
independent of which neighbors' irradiance fed the blur - the blur only ever operates on the
lighting signal (`combinedIrradiance`), and material re-application happens once, after, at full
sharpness. Left toggleable (raw irradiance vs. final lit image) purely for debugging.

**`FORCE_FULL_SPLIT` exists to isolate `decideWeight`'s own quality from `shouldSplit`'s split
heuristic.** Forcing every stack node to fully descend to mip 0 regardless of the baked quadtree
removes the "where to refine" question entirely, so any remaining quality issue must be in the
weighting math, not the split decision - the mode this project's mobile-perf-tuning work used to
validate `decideWeight` before the quadtree existed, kept around specifically for re-isolating the
two failure modes from each other later, at the cost of being extremely slow by design (worst-case
full descent from every one of the 9 seeds).
