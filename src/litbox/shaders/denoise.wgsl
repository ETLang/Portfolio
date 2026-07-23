// Hierarchical guided blur over combinedIrradiance, via a quadtree-style "consideration stack"
// per output pixel. See this project's denoiser plan for the full design argument; only the
// load-bearing rationale is repeated here as comments.
//
// ShouldSplit() normally consults the baked min/max-range quadtree built by
// build_denoiser_quadtree.wgsl (see shouldSplit() below). FORCE_FULL_SPLIT (see
// DenoiseOperation.updateSwitches) is a debug switch, not the normal path: when defined, it forces
// every stack node to split down to mip 0 regardless of the quadtree - the mode used to validate
// DecideBlurSize/DecideWeight's quality in isolation before the quadtree was built (extremely
// slow by design). Kept around for re-isolating weight-quality regressions from split-heuristic
// regressions later.
//
// combineAlbedoDensity is a compile-time switch (see DenoiseOperation.updateSwitches): this is
// where albedo/density get folded into the final image, post-blur, using the CENTER pixel's own
// material properties (not a per-sample value - albedo/density describe the material AT the
// output pixel, independent of which neighbors' irradiance fed the blur). Kept toggleable for
// debugging (raw irradiance vs. final lit image).
#include "LitboxCommon.wgsl"

struct DenoiseUniforms {
    varianceScale: f32,
    darknessNoiseFloor: f32,
    maxBlurMip: f32,
    albedoSensitivity: f32,
    densitySensitivity: f32,
    normalSensitivity: f32,
    sigmaLuminanceTight: f32,
    sigmaLuminanceLoose: f32,
    kLuminance: f32,
    // Distance-bias split cutoff (this project's denoiser plan) - see shouldSplit()'s doc comment
    // for the exact normalization (node-relative texels, not seed-relative).
    maxSplitDistance: f32,
}
@group(0) @binding(0) var<uniform> uniforms: DenoiseUniforms;

// Full mip chains (not just mip0) - the guided blur samples arbitrary levels of each. albedo/
// normalRoughness/density are sampled with nearestSampler (material identity shouldn't blend
// across a boundary); combinedIrradiance with linearSampler (smoother reconstruction of the
// signal actually being blurred). filteredVariance has only one (quarter-res) level.
@group(1) @binding(0) var combinedIrradiance: texture_2d<f32>;
@group(1) @binding(1) var albedo: texture_2d<f32>;
@group(1) @binding(2) var normalRoughness: texture_2d<f32>;
@group(1) @binding(3) var density: texture_2d<f32>;
@group(1) @binding(4) var filteredVarianceTex: texture_2d<f32>;
@group(1) @binding(5) var linearSampler: sampler;
@group(1) @binding(6) var nearestSampler: sampler;
// Baked min/max-range quadtree (see build_denoiser_quadtree.wgsl) - r32float, own 0-indexed mip
// chain at half G-Buffer resolution. Declared unconditionally (used only when FORCE_FULL_SPLIT is
// off) - same pattern as albedo/density's conditional use under COMBINE_ALBEDO_DENSITY below;
// 'auto' bind group layout includes a binding for every declared resource regardless of whether
// it's reachable under the active #ifdef branch, so updateInputs always provides it.
@group(1) @binding(7) var quadtreeMustSplit: texture_2d<f32>;

@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

const SEED_RADIUS: i32 = 1; // 3x3 seed neighborhood at the chosen starting mip.

// Cheap per-pixel hash (Hash without Sine, David Hoskins) backing the seed-placement jitter below -
// see seedJitter's own doc comment for why this exists.
fn hash2(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// See this project's denoiser plan for the derivation: sized to survive Phase 1's forced-full-
// split worst case (every one of the 9 seeds fully descending from maxBlurMip to mip 0 in a
// depth-first push-4/pop-1 traversal), not copied from the Unity reference's arbitrary 64.
const STACK_SIZE: u32 = 32u;

struct TreeSampleNode {
    uv: vec2<f32>,
    mip: i32,
}

// PERF EXPERIMENT (mobile occupancy floor): this used to be a THREAD_COUNT*STACK_SIZE
// var<workgroup> array with a per-thread stackBase slice, to route around CLAUDE.md's documented
// WGSL indexing bug. That bug is specifically about dynamically indexing a *literal*-initialized
// local array (`var x = array<T, N>(...); x[runtimeIndex]`) - this is a plain, uninitialized local
// declaration (no array(...) constructor), a different WGSL construct, so it's not expected to hit
// that lowering bug. Trying function-local here because the workgroup-shared version reserved
// THREAD_COUNT*STACK_SIZE*16 bytes of shared memory per workgroup unconditionally (whether or not
// any thread's traversal used it), which likely capped resident-workgroup occupancy on this GPU
// regardless of actual per-pixel work - see this project's denoiser plan / mobile-perf-tuning
// notes. STILL NEEDS REAL-HARDWARE CORRECTNESS VERIFICATION per CLAUDE.md - the known bug produces
// silent output corruption, not a validation error or crash, so a clean run isn't proof by itself.

// Evidence available: filteredVariance (relative-variance, quarter res) and combinedIrradiance's
// own mip chain (already box-filtered, so a coarse mip IS the local mean - no separate blur pass
// needed the way the Unity reference re-derives one by hand). Combined with max(), not a blend:
// relative variance under-reports noise in rarely-hit dark regions (both A/B half-samples land
// near zero, so their difference is deceptively small - it's a difference of two things wrong in
// the same direction, not an absolute confidence measure), but mean brightness doesn't share that
// blind spot, so whichever signal says "blur more" wins.
fn decideBlurSize(centerVariance: f32, localLuminance: f32) -> f32 {
    let darknessShortfall = saturate((uniforms.darknessNoiseFloor - localLuminance) / (uniforms.darknessNoiseFloor + 1e-3));
    let adjustedVariance = uniforms.maxBlurMip * max(centerVariance * uniforms.varianceScale, darknessShortfall);
    return clamp(adjustedVariance, 0.0, uniforms.maxBlurMip);
}

// O(1) lookup into the baked min/max-range quadtree (see build_denoiser_quadtree.wgsl and this
// project's denoiser plan), gated by a distance bias against splitting far from the query pixel -
// see this project's denoiser plan. quadtreeMustSplit's own mip index is offset by -1 from
// G-Buffer/irradiance mip space (it's allocated at half the G-Buffer's resolution with its own
// 0-indexed chain - level i there answers "should G-Buffer mip (i+1) split into mip-i children") -
// mip - 1 is the correct index into quadtreeMustSplit's own space, not mip. FORCE_FULL_SPLIT
// (Phase 1's debug mode) still takes priority when defined, ignoring both the baked quadtree and
// the distance bias entirely - that mode exists specifically to isolate DecideWeight's own quality
// from any split heuristic, distance-based or otherwise.
//
// Distance bias rationale: as a candidate node's distance from the query pixel grows relative to
// the whole blur kernel's radius, further refining it matters less and less to the final weighted
// result, since decideWeight's spatialWeight will heavily discount it regardless of how accurately
// its fine structure gets resolved - resolving detail nobody's going to weight is wasted work.
//
// Bug fix: this used to normalize by the CURRENT node's own (shrinking) texel size instead of the
// fixed seedTexelSize passed in now, on the theory that dividing by a shrinking footprint made the
// cutoff "depth-aware": a branch starting AT the query (distance 0) keeps a ~0 ratio forever, so
// that part checked out - but every other one of the 3x3 neighborhood's 8 seeds (see SEED_RADIUS)
// starts a full node-width away, and a split's children are only ever offset from their parent by
// a FIXED FRACTION of the *parent's* (not the query's) footprint - a geometric series that
// converges, so any branch's descendants stay within a bounded distance of where their seed
// started, no matter how deep they split. That means the numerator (true distance from the query)
// stays roughly constant for an off-center seed while the old denominator (current node's texel
// size) shrinks to 0 every split - the ratio was mathematically guaranteed to diverge to infinity
// for every non-center seed, regardless of the threshold. Raising the threshold only bought a few
// more split levels before the identical failure would recur one level deeper on a scene needing
// more of them (a noisier/darker scene, or any change that pushes decideBlurSize's chosen mip
// higher) - a strong sign the *formula*, not the constant, was wrong; the fix belongs here, not in
// DEFAULT_DENOISER_TUNABLES.maxSplitDistance. Normalizing by the fixed
// seedTexelSize instead - the same quantity decideWeight's spatialWeight already uses (see that
// function's own doc comment) - gives every branch a stable, depth-independent ratio approximating
// its true final distance from the query, so the cutoff now actually answers the question it's
// supposed to ("will decideWeight care about this branch at all"), instead of one that happens to
// answer "has this branch split more than ~log2(maxSplitDistance) times," which is a different
// question that doesn't track real relevance.
fn shouldSplit(uv: vec2<f32>, mip: i32, queryUv: vec2<f32>, texelSize: vec2<f32>) -> bool {
#ifdef FORCE_FULL_SPLIT
    return mip > 0;
#else
    if (mip <= 0) {
        return false;
    }

    // The split distance condition scales with the texel size of each mip level.
    // This is by design and works quite nicely.
    let nodeTexelSize = texelSize * f32(1u << u32(mip));
    let distanceInNodeTexels = length(uv - queryUv) / max(nodeTexelSize.x, nodeTexelSize.y);
    if (distanceInNodeTexels > uniforms.maxSplitDistance) {
        return false;
    }

    let quadSize = vec2<i32>(textureDimensions(quadtreeMustSplit, mip - 1));
    let quadCoords = clamp(vec2<i32>(uv * vec2<f32>(quadSize)), vec2<i32>(0), quadSize - vec2<i32>(1));
    return textureLoad(quadtreeMustSplit, quadCoords, mip - 1).r != 0.0;
#endif
}

// spatialWeight * structuralWeight * radianceWeight * node.size - see the denoiser plan for the
// full justification of each term; summary:
// - structuralWeight (G-Buffer similarity) rejects cross-material bleed (hard edges,
//   silhouettes) and, via the normal term, protects a smooth Lambertian shading gradient on a
//   uniform-albedo curved object - real, converged detail with no albedo/density signature.
// - radianceWeight rejects cross-*feature* bleed structuralWeight cannot see at all: a G-Buffer-
//   uniform region (e.g. a laser beam through haze) can still contain a real, sharp irradiance
//   feature. Its sigma is adaptive on the *center* pixel's own variance (tighter where the center
//   is already trustworthy, looser where it isn't) - reusing filter_variance.wgsl's validated
//   pattern verbatim, so a genuinely noisy-but-flat region doesn't get rejected by its own MC
//   noise.
//   (spatialWeight is normalized by the fixed seedTexelSize - the same metric shouldSplit's own
//   distance bias now uses, see that function's doc comment for why an earlier, node-relative
//   version of that cutoff was a bug, not a deliberate difference from this one.)
// - node.size is a partition-of-unity area weight (4^-depth-below-seed), not stored on the node
//   since it's a pure function of (startMip - node.mip): guarantees a region that fully resolves
//   to mip 0 contributes the same total weight mass as it would have unsplit, so a heavily-
//   detailed (hence heavily-split) region doesn't numerically dominate the average just by
//   producing more stack entries.
fn decideWeight(
    queryUv: vec2<f32>,
    centerAlbedo: vec3<f32>,
    centerNormal: vec3<f32>,
    centerOpticalDepth: f32,
    centerLuminance: f32,
    sigmaAdaptive: f32,
    seedTexelSize: vec2<f32>,
    node: TreeSampleNode,
    nodeSize: f32,
) -> vec4<f32> {
    let nodeMipF = f32(node.mip);
    let sampleAlbedo = textureSampleLevel(albedo, nearestSampler, node.uv, nodeMipF).rgb;
    let sampleNormal = textureSampleLevel(normalRoughness, nearestSampler, node.uv, nodeMipF).xyz;
    let sampleDensityValue = textureSampleLevel(density, nearestSampler, node.uv, nodeMipF).r / DENSITY_SCALE;
    let sampleColor = textureSampleLevel(combinedIrradiance, linearSampler, node.uv, nodeMipF).rgb;
    let sampleLuminance = luminance(sampleColor);

    let albedoDiff = distance(centerAlbedo, sampleAlbedo);
    var wAlbedo = 1.0 - saturate(albedoDiff / uniforms.albedoSensitivity);
    wAlbedo *= wAlbedo;

    let normalDot = saturate(dot(normalize(centerNormal), normalize(sampleNormal)));
    let wNormal = pow(normalDot, uniforms.normalSensitivity);

    let opticalDepthDiff = abs(centerOpticalDepth - opticalDepth(sampleDensityValue));
    var wDensity = 1.0 - saturate(opticalDepthDiff / uniforms.densitySensitivity);
    wDensity *= wDensity;

    let structuralWeight = wAlbedo * wNormal * wDensity;
    let radianceWeight = gaussianWeight(abs(centerLuminance - sampleLuminance), sigmaAdaptive);

    let distanceInSeeds = length(node.uv - queryUv) / max(seedTexelSize.x, seedTexelSize.y);
    let spatialWeight = exp(-distanceInSeeds * distanceInSeeds);

    let weight = spatialWeight * structuralWeight * radianceWeight * nodeSize;
    return vec4<f32>(sampleColor * weight, weight);
}

@compute @workgroup_size(8, 2, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let size = textureDimensions(output);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let texelSize = 1.0 / vec2<f32>(size);

    let centerAlbedo = textureLoad(albedo, coords, 0).rgb;
    let centerNormal = textureLoad(normalRoughness, coords, 0).xyz;
    let centerDensityValue = textureLoad(density, coords, 0).r / DENSITY_SCALE;
    let centerOpticalDepth = opticalDepth(centerDensityValue);
    // r32float textures aren't filterable in core WebGPU (no float32-filterable feature enabled
    // here) - textureSample/textureSampleLevel would fail bind-group creation (Float sampleType
    // required vs. the texture's actual UnfilterableFloat). textureLoad only, matching
    // filter_variance.wgsl's existing convention for its own r32float inputs.
    let varianceSize = vec2<i32>(textureDimensions(filteredVarianceTex));
    let varianceCoords = clamp(coords / 4, vec2<i32>(0), varianceSize - vec2<i32>(1));
    let centerVariance = textureLoad(filteredVarianceTex, varianceCoords, 0).r;
    let centerIrradiance = textureLoad(combinedIrradiance, coords, 0).rgb;

    let maxMip = i32(textureNumLevels(combinedIrradiance)) - 1;
    let localLuminanceMip = min(3, maxMip);
    let localLuminance = luminance(textureSampleLevel(combinedIrradiance, linearSampler, uv, f32(localLuminanceMip)).rgb);

    let blurSize = decideBlurSize(centerVariance, localLuminance);
    let startMip = clamp(i32(round(blurSize)), 0, maxMip);

    if (startMip == 0) {
#ifdef COMBINE_ALBEDO_DENSITY
        textureStore(output, coords, vec4<f32>(centerIrradiance * centerAlbedo * centerDensityValue, 1.0));
#else
        textureStore(output, coords, vec4<f32>(centerIrradiance, 1.0));
#endif
        return;
    }

    let sigmaAdaptive = mix(uniforms.sigmaLuminanceTight, uniforms.sigmaLuminanceLoose,
        smoothstep(0.0, 1.0 / uniforms.kLuminance, centerVariance));
    let centerLuminance = luminance(centerIrradiance);

    let seedTexelSize = texelSize * f32(1u << u32(startMip));
    // All 9 seeds share the same seedTexelSize wherever startMip is locally constant (true across
    // most of a flat, uniform-material region - see DecideBlurSize), so every seed's position (and
    // therefore its "which quadtree cell am I in" state) advances in exact lockstep as the query
    // pixel moves. Left unjittered, that state flips at a fixed, periodic pixel interval matching
    // the quadtree's own grid pitch wherever the true (correctly smooth) signal happens to cross a
    // cell boundary - visible as coherent banding on an otherwise perfectly flat surface (confirmed:
    // RTRect's flat facets - see primitive_mesh.ts - show this clearly, since there's no real
    // structure there to mask it). A per-pixel hash-based jitter, up to half a seed-texel, decorrelates
    // neighboring pixels' seed positions so that same aliasing turns into unstructured
    // high-frequency noise instead - noise reads as far less objectionable than coherent stripes,
    // the standard fix for this class of discretization artifact (the same principle as dithering a
    // banded gradient). Applied only to the top-level seed placement, not to recursive child
    // splits - decideWeight's structural/radiance weighting already rejects a jittered seed that
    // lands on the wrong side of a real material edge, so there's no correctness cost, only a
    // small softening of genuine edges (acceptable - see DecideWeight's own weighting for why a
    // seed that drifts across a real boundary still gets down-weighted there).
    let seedJitter = (hash2(vec2<f32>(coords)) - 0.5) * seedTexelSize;
    var stack: array<TreeSampleNode, STACK_SIZE>;
    var stackCount: u32 = 0u;

    for (var dy = -SEED_RADIUS; dy <= SEED_RADIUS; dy++) {
        for (var dx = -SEED_RADIUS; dx <= SEED_RADIUS; dx++) {
            stack[stackCount] = TreeSampleNode(uv + vec2<f32>(f32(dx), f32(dy)) * seedTexelSize + seedJitter, startMip);
            stackCount++;
        }
    }

    var accumulated = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    while (stackCount > 0u) {
        stackCount--;
        let current = stack[stackCount];
        let depth = startMip - current.mip;
        let nodeSize = pow(0.25, f32(depth));

        if (shouldSplit(current.uv, current.mip, uv, texelSize) && stackCount + 4u <= STACK_SIZE) {
            let childMip = current.mip - 1;
            let childTexelSize = texelSize * f32(1u << u32(childMip));
            stack[stackCount]      = TreeSampleNode(current.uv + vec2<f32>(-0.5, -0.5) * childTexelSize, childMip);
            stack[stackCount + 1u] = TreeSampleNode(current.uv + vec2<f32>( 0.5, -0.5) * childTexelSize, childMip);
            stack[stackCount + 2u] = TreeSampleNode(current.uv + vec2<f32>(-0.5,  0.5) * childTexelSize, childMip);
            stack[stackCount + 3u] = TreeSampleNode(current.uv + vec2<f32>( 0.5,  0.5) * childTexelSize, childMip);
            stackCount += 4u;
            continue;
        }

        // Leaf: either ShouldSplit said no, or the stack had no room to split further - resolved
        // at its current (coarser) mip either way. Correctness fix vs. the Unity reference: that
        // version silently drops a node's energy when it can't split due to stack overflow
        // (falls through to leaf code that only accumulates at mip 0); resolving as a coarser
        // leaf here instead bounds worst-case quality degradation without losing energy.
        accumulated += decideWeight(uv, centerAlbedo, centerNormal, centerOpticalDepth, centerLuminance,
            sigmaAdaptive, seedTexelSize, current, nodeSize);
    }

    let blurred = select(centerIrradiance, accumulated.rgb / accumulated.a, accumulated.a > 0.0);

#ifdef COMBINE_ALBEDO_DENSITY
    textureStore(output, coords, vec4<f32>(blurred * centerAlbedo * centerDensityValue, 1.0));
#else
    textureStore(output, coords, vec4<f32>(blurred, 1.0));
#endif
}
