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
    densityBlurFalloff: f32,
    albedoSensitivity: f32,
    densitySensitivity: f32,
    normalSensitivity: f32,
    sigmaLuminanceTight: f32,
    sigmaLuminanceLoose: f32,
    kLuminance: f32,
    // Distance-bias split cutoff (this project's denoiser plan) - see shouldSplit()'s doc comment
    // for the exact normalization (node-relative texels, not seed-relative).
    maxSplitDistance: f32,
    // Salts seedJitter's hash below so the dither pattern varies frame-to-frame ("film grain")
    // instead of being a fixed function of pixel coordinate alone - see
    // SimulationResources.frameIndex's doc comment for why that's safe here (no history buffer to
    // disagree with a changing pattern, unlike typical TAA jitter).
    frameIndex: f32,
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
// Diagnostic-only: the continuous blurSize (see decideBlurSize) each pixel actually resolved to,
// written unconditionally (both early-return and normal paths) - lets the 'blur-size' debug view
// (see LitboxSceneRenderer) show this decision directly instead of inferring it from how noisy or
// smooth the final composited image looks, which is what every earlier investigation of a "why is
// this edge sharp" report had to fall back on.
@group(2) @binding(1) var blurSizeOutput: texture_storage_2d<r32float, write>;

const SEED_RADIUS: i32 = 1; // 3x3 seed neighborhood at the chosen starting mip.

// See this project's denoiser plan for the derivation: sized to survive Phase 1's forced-full-
// split worst case (every one of the 9 seeds fully descending from maxBlurMip to mip 0 in a
// depth-first push-4/pop-1 traversal), not copied from the Unity reference's arbitrary 64.
const STACK_SIZE: u32 = 32u;

struct TreeSampleNode {
    uv: vec2<f32>,
    mip: i32,
}

fn decideBlurSize(centerVariance: f32, localLuminance: f32, localOpticalDepth: f32) -> f32 {
    let darknessShortfall = saturate((uniforms.darknessNoiseFloor - localLuminance) / (uniforms.darknessNoiseFloor + 1e-3));
    let adjustedVariance = max(centerVariance * uniforms.varianceScale, darknessShortfall);
    let densityAttenuation = smoothstep(0, 1, uniforms.densityBlurFalloff / localOpticalDepth);
    return saturate(adjustedVariance * densityAttenuation) * uniforms.maxBlurMip;
}

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

fn hasNearbyDetail(uv: vec2<f32>) -> bool {
#ifdef FORCE_FULL_SPLIT
    return false;
#else
    let quadSize = vec2<i32>(textureDimensions(quadtreeMustSplit, 0));
    let quadCoords = clamp(vec2<i32>(uv * vec2<f32>(quadSize)), vec2<i32>(0), quadSize - vec2<i32>(1));
    return textureLoad(quadtreeMustSplit, quadCoords, 0).r != 0.0;
#endif
}

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

    // heightScale: 0 (a deliberately flat/bump-free raytraced object - see raytraced_gbuffer.wgsl's
    // `normal * props.heightScale`) stores an exact zero vector in the G-Buffer, not a unit normal.
    // normalize(vec3(0)) is NaN (0/0 per component), which would otherwise poison wNormal ->
    // structuralWeight -> this function's whole return value -> accumulated.a in main(), silently
    // defeating the blur for every pixel touched by such an object (confirmed: battle_scene's
    // background haze rect has heightScale=0 and covers ~99.9% of the frame - decideWeight NaNed
    // out almost everywhere, and `accumulated.a > 0.0` is false for NaN, so main() fell back to
    // raw, unblurred centerIrradiance for virtually the whole image). compute_volatility.wgsl
    // already treats a near-zero normal as "no orientation data" (dot(n,n) < 0.1) rather than
    // normalizing it blindly - mirror that guard here. Absent any real orientation signal, there's
    // no basis to penalize a normal "mismatch", so this term is a pass-through (1.0), not a reject.
    let hasNormalData = dot(centerNormal, centerNormal) >= 0.1 && dot(sampleNormal, sampleNormal) >= 0.1;
    let normalDot = saturate(dot(normalize(centerNormal), normalize(sampleNormal)));
    let wNormal = select(1.0, pow(normalDot, uniforms.normalSensitivity), hasNormalData);

    let opticalDepthDiff = abs(centerOpticalDepth - opticalDepth(sampleDensityValue));
    var wDensity = 1.0 - saturate(opticalDepthDiff / uniforms.densitySensitivity);
    wDensity *= wDensity;

    let structuralWeight = wAlbedo * wNormal * wDensity;
    // Log-domain comparison (see LitboxCommon.wgsl's logLuminance): a raw linear-luminance
    // difference against a FIXED sigma only means the same thing across scenes if their absolute
    // brightness scales are similar. Comparing in log space instead makes it a ratio, which a fixed
    // sigma can mean consistently regardless of a scene's overall light energy - a smaller,
    // independent robustness improvement alongside the wNormal fix above (this one wasn't confirmed
    // to be the reported bug's root cause, but doesn't regress cornell_square either).
    let radianceWeight = gaussianWeight(abs(logLuminance(centerLuminance, uniforms.darknessNoiseFloor) - logLuminance(sampleLuminance, uniforms.darknessNoiseFloor)), sigmaAdaptive);

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

    // A coarse, bilinearly-filtered density read for decideBlurSize's attenuation term -
    // deliberately NOT centerOpticalDepth (that stays pixel-exact for decideWeight's material-
    // identity comparison and the final COMBINE_ALBEDO_DENSITY multiply below). decideBlurSize
    // wants "how dense is my surroundings," the same kind of broad, low-frequency signal
    // localLuminance already is (hence the matching mip choice) - sampling density at full
    // resolution instead made the attenuation term track the density texture's own pixel-sharp
    // edges (e.g. a cloud sprite's alpha feather), so at an aggressive densityBlurFalloff the
    // attenuation could swing from ~1 to ~0 within the same couple of source pixels that edge
    // spans, which then rounds to a hard mip-0-vs-mip-1+ line instead of a gradual falloff.
    let densityMaxMip = i32(textureNumLevels(density)) - 1;
    let densityBlurMip = min(1, densityMaxMip);
    let localDensityValue = textureSampleLevel(density, linearSampler, uv, f32(densityBlurMip)).r / DENSITY_SCALE;
    let localOpticalDepth = opticalDepth(localDensityValue);

    let blurSize = clamp(decideBlurSize(centerVariance, localLuminance, localOpticalDepth), 0.0, f32(maxMip));
    //let blurSize = decideBlurSize(centerVariance, localLuminance, localOpticalDepth);
    textureStore(blurSizeOutput, coords, vec4<f32>(blurSize, 0.0, 0.0, 0.0));

    if (blurSize <= 0.0) {
#ifdef COMBINE_ALBEDO_DENSITY
        textureStore(output, coords, vec4<f32>(centerIrradiance * centerAlbedo * centerDensityValue, 1.0));
#else
        textureStore(output, coords, vec4<f32>(centerIrradiance, 1.0));
#endif
        return;
    }

    // The actual texture reads still need one discrete mip - there's no hardware trilinear
    // equivalent for "blend the results of two whole quadtree descents" the way there is for a
    // single texture fetch at a fractional LOD, since startMip seeds an entire recursive
    // gather/weight algorithm (see shouldSplit/decideWeight below), not a texel lookup. ceil (not
    // round) so any blurSize in (0, 1] still reads mip 1's data - the near-zero end of that range
    // is instead handled by seedTexelSize's continuous scaling just below, not by falling back to
    // mip 0.
    let startMip = i32(ceil(blurSize));

    // Prefer the quadtree's detail evidence over centerVariance where they disagree: variance
    // alone can't distinguish "there's a real feature here" from "this is just noisy" once it's
    // saturated (the common case in a sparse/dark scene - see hasNearbyDetail's doc comment), so a
    // query pixel the quadtree already flagged as sitting near real detail is kept at the tight
    // sigma regardless of what its own variance says - prioritizing not smearing a genuine edge
    // over suppressing noise there. Confirmed necessary: raising sigmaLuminanceLoose (see
    // DEFAULT_DENOISER_TUNABLES in simulation.ts) to fix battle_scene's noisy haze halo also
    // started dissolving the laser beam itself into that same halo, since decideWeight had no
    // signal other than variance (saturated near the beam too) to tell the two apart.
    let sigmaAdaptive = select(
        mix(uniforms.sigmaLuminanceTight, uniforms.sigmaLuminanceLoose,
            smoothstep(0.0, 1.0 / uniforms.kLuminance, centerVariance)),
        uniforms.sigmaLuminanceTight,
        hasNearbyDetail(uv));
    let centerLuminance = luminance(centerIrradiance);

    // exp2(blurSize), not f32(1u << u32(startMip)) - continuous, not stepped in powers of two.
    // This is what actually makes the blur-size transition smooth despite startMip itself being a
    // discrete mip: decideWeight's spatialWeight (and the seed positions below) are normalized by
    // this value, so as blurSize shrinks toward 0 within a single startMip bracket (e.g. 0.05, still
    // reading mip 1's data), the seeds collapse toward the query pixel's own UV and spatialWeight
    // correspondingly collapses toward "only the center sample matters" - converging smoothly to the
    // unblurred result the old startMip==0 branch used to jump straight to, instead of holding full
    // mip-1-radius weight right up until blurSize crossed 0.5 and then dropping to zero at once.
    let seedTexelSize = texelSize * exp2(blurSize);
    // All 9 seeds share the same seedTexelSize wherever blurSize is locally constant (true across
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
    // uniforms.frameIndex is wrapped at a small bound on the JS side specifically so this offset
    // stays tiny (see SimulationResources.frameIndex's doc comment) - large enough to decorrelate
    // frame-to-frame, nowhere near f32's exact-integer precision cliff where it could instead
    // collide adjacent pixels' hash inputs and undo the spatial decorrelation this jitter exists
    // for in the first place.
    let temporalOffset = vec2<f32>(uniforms.frameIndex * 37.0, uniforms.frameIndex * 17.0);
    let seedJitter = (hash22(vec2<f32>(coords) + temporalOffset) - 0.5) * seedTexelSize;
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
