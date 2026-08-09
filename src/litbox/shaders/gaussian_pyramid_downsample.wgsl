// 5x5 binomial-weighted blur+downsample - the shader half of the lightmap's Gaussian blur pyramid
// (see LightmapBlurPyramid/lightmap_blur_pyramid.ts, the orchestrator that owns looping this over
// a whole mip chain). Every level this feeds blurs the *previous* level, not mip 0 directly, so
// the blur compounds geometrically down the chain - the classic Burt-Adelson Gaussian pyramid
// "reduce" operator - which is what lets a kernel this small still reach a very soft/blobby look
// after a few levels, all the way to essentially the frame mean at the smallest level. Replaces a
// plain box-filtered mip chain (mip_downsample.wgsl, still used elsewhere - see
// lightmap_blur_pyramid.ts) specifically where the destination is meant to be sampled back at a
// much larger size later (sprite.wgsl's blur-size selection): a box-filtered mip has no internal
// smoothing, so bilinear-magnifying it back up produces visibly faceted, blocky gradients - a real
// blur avoids that.
//
// Weights are the outer product of the standard 1-D binomial approximation to a Gaussian,
// [1,4,6,4,1]/16 (each dimension sums to 1, so the 2D kernel sums to 1 overall). Applied via
// accumulate() below as 25 separate compile-time-constant-weighted calls rather than looping over
// a WGSL array literal - see this project's CLAUDE.md: dynamically indexing a function-local array
// literal has silently corrupted output on some mobile GPU drivers, with zero validation error to
// catch it.
//
// Output texel (x,y) reads source texels centered on (2x, 2y) - the standard Burt-Adelson
// decimated-pyramid tap position (every output sample corresponds to an even-indexed input
// sample, post-blur) - clamped to the source's edges (manual clamp-to-edge, since this uses
// textureLoad rather than a filtering sampler). At the pyramid's smallest levels every tap clamps
// to the same texel and the weights (summing to 1) reduce to an exact passthrough, so no
// special-casing is needed near 1x1.
#include "LitboxCommon.wgsl"

@group(1) @binding(0) var sourceTex: texture_2d<f32>;

@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

fn clampedLoad(sourceSize: vec2<i32>, coord: vec2<i32>) -> vec4<f32> {
    return textureLoad(sourceTex, clamp(coord, vec2<i32>(0), sourceSize - vec2<i32>(1)), 0);
}

#ifdef KARIS_WEIGHTED
// Karis average (Brian Karis' firefly fix for Unreal's bloom downsample chain) - re-weights each
// tap by 1/(1+luminance) on top of its binomial weight, then accumulate()'s caller renormalizes
// by the actual weight sum used instead of dividing by the fixed 256. Plain fixed-weight
// downsampling aliases badly on an HDR spike surrounded by near-zero neighbors: depending on
// exactly where the bright source texel falls relative to the kept (even) sample positions, one
// output texel can inherit most of its weight while its immediate neighbor inherits almost none -
// a jump scaled directly by that texel's brightness, with no upper bound (see this project's
// plan). Down-weighting a tap in proportion to its own brightness bounds that jump regardless of
// how bright the spike is (a tap's maximum possible contribution to the average approaches 1
// instead of growing with luminance).
//
// LightmapBlurPyramid uses this at *every* level, not just the first reduce step - a point-like
// firefly (a single bright texel) does get fully absorbed by one Karis-weighted pass, but a
// laser-beam-style feature (bright and only 1-2 texels wide along its whole visible length, but
// long) stays that narrow across many octaves of downsampling as the pyramid keeps halving
// resolution, so it can start re-aliasing at whichever level its width first drops below ~1 texel
// - which level that is depends on the beam's on-screen width and can't be predicted in advance.
// Applying this at every level closes that gap regardless of where it would have occurred.
//
// The energy-loss tradeoff this implies (a bright tap's contribution can be under-represented) is
// smaller than it first looks, and specifically local: for a *locally uniform* bright
// neighborhood (every tap has similar luminance), the per-tap 1/(1+luminance) factor is nearly
// identical across all 25 taps and cancels almost exactly in the renormalized average - the result
// is close to the same value plain weighting would have produced, at any brightness. The
// under-representation only bites where there's real local contrast within one 5x5 window (a
// bright tap next to much darker ones) - i.e. exactly the outlier/edge case that aliases, and
// nowhere else - so applying this at every level doesn't compound into a global dimming bias on
// wide, smoothly-varying bright regions the way it might seem to at first.
fn tapWeight(baseWeight: f32, color: vec4<f32>) -> f32 {
    return baseWeight / (1.0 + luminance(color.rgb));
}
#else
// Plain, energy-preserving binomial weights - kept as the other half of the KARIS_WEIGHTED switch
// (see GaussianPyramidDownsampleOperation) for isolating/comparing the two later, the same reason
// denoise.wgsl keeps FORCE_FULL_SPLIT around - not currently used by LightmapBlurPyramid, which
// applies Karis weighting unconditionally at every level (see above).
fn tapWeight(baseWeight: f32, color: vec4<f32>) -> f32 {
    return baseWeight;
}
#endif

fn accumulate(sum: ptr<function, vec4<f32>>, weightSum: ptr<function, f32>, sourceSize: vec2<i32>, coord: vec2<i32>, baseWeight: f32) {
    let color = clampedLoad(sourceSize, coord);
    let weight = tapWeight(baseWeight, color);
    *sum += weight * color;
    *weightSum += weight;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let outputSize = textureDimensions(output);
    if (id.x >= outputSize.x || id.y >= outputSize.y) {
        return;
    }

    let sourceSize = vec2<i32>(textureDimensions(sourceTex));
    let center = vec2<i32>(id.xy) * 2;

    var sum = vec4<f32>(0.0);
    var weightSum = 0.0;

    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>(-2, -2), 1.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>(-1, -2), 4.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 0, -2), 6.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 1, -2), 4.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 2, -2), 1.0 / 256.0);

    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>(-2, -1), 4.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>(-1, -1), 16.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 0, -1), 24.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 1, -1), 16.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 2, -1), 4.0 / 256.0);

    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>(-2,  0), 6.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>(-1,  0), 24.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 0,  0), 36.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 1,  0), 24.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 2,  0), 6.0 / 256.0);

    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>(-2,  1), 4.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>(-1,  1), 16.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 0,  1), 24.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 1,  1), 16.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 2,  1), 4.0 / 256.0);

    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>(-2,  2), 1.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>(-1,  2), 4.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 0,  2), 6.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 1,  2), 4.0 / 256.0);
    accumulate(&sum, &weightSum, sourceSize, center + vec2<i32>( 2,  2), 1.0 / 256.0);

    textureStore(output, vec2<i32>(id.xy), sum / max(weightSum, 1e-5));
}
