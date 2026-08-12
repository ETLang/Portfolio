// Fused mean/variance/mip-generation pass - a structural port of Unity's confirmed-live
// ComputeVarianceAndNMipsFromSamplePair (TracerPostProcessing.compute), not a workaround for it -
// see this project's denoiser plan. One workgroup covers one 16x16 tile of irradianceA/
// irradianceB (mip0), reducing both halves independently through workgroup shared memory
// (var<workgroup> - safe/recommended for runtime-indexed data per this project's CLAUDE.md; the
// array-literal mobile bug this project has hit before is specifically about function-local array
// *literals*, not shared memory) down to mip2 (quarter resolution) - as far as this fused pass
// goes; see MipDownsampleOperation for continuing past mip2, once irradianceA/B no longer need to
// stay separate (that's this project's design choice - a 16-wide tile could reduce down to mip4
// in principle, but this pass is capped at 4 simultaneous storage-texture outputs
// (combinedMip0/1/2 + rawVariance) to stay within WebGPU's guaranteed-minimum
// maxStorageTexturesPerShaderStage of 4, which some real hardware enforces - going to mip4 would
// need 6).
//
// Two mutually exclusive modes, chosen by COMBINE_ALBEDO_DENSITY (set when
// SimulationResources.denoiserEnabled is false):
//
// - Denoiser enabled (COMBINE_ALBEDO_DENSITY undefined): combinedMip0/1/2 hold the raw mean(A,B)
//   irradiance signal used everywhere downstream by the guided blur - each level is a genuine
//   box-filtered pyramid of A and of B *independently*, only combined into a mean at the point of
//   writing each level's output texture (never a combination of already-computed variances -
//   seemingly-plausible but not statistically valid, see this project's denoiser plan). rawVariance
//   (binding 3) is also produced, combining two complementary readings at mip2 (see that stage for
//   the full rationale): a mean-of-(temporal A-vs-B)-variances, computed per pixel via
//   relativeVariance() (a squared log-luminance difference, not Unity's original mean-normalized
//   ratio) and propagated/averaged down to mip2, plus a fresh spatial variance-of-means computed
//   directly from the 16 mip0 pixels' own logLuminance(mean) values at mip2.
//
// - Denoiser disabled (COMBINE_ALBEDO_DENSITY defined): irradiance itself is never the wanted
//   output - the final lit image (irradiance * albedo * density) is. combinedMip0/1/2 are bound
//   directly to the top 3 mip levels of `lightmap` (see SimulationResources.run()) and each level
//   stores mean * albedo * density instead of raw mean, where albedo/density are reduced through
//   the same shared-memory box-filter pyramid as the irradiance mean itself (sharedAlbedo/
//   sharedDensity below) - not sampled from a separately-generated albedo/density mip chain, since
//   none is needed elsewhere in this mode. This is the "mipmapped lightmap" case: no separate
//   dispatch (DenoiseOperation's own combine-only fast path) is worth it just for a trivial
//   per-pixel multiply when this pass already computes mean0/1/2 anyway. rawVariance doesn't exist
//   in this mode (binding 3 isn't declared at all) - nothing downstream of the denoiser-disabled
//   path ever reads variance.
#include "LitboxCommon.wgsl"

@group(1) @binding(0) var irradianceA: texture_2d<f32>;
@group(1) @binding(1) var irradianceB: texture_2d<f32>;
#ifdef COMBINE_ALBEDO_DENSITY
@group(1) @binding(2) var albedo: texture_2d<f32>;
@group(1) @binding(3) var density: texture_2d<f32>;
#endif

// combinedMip0/1/2: raw mean(A,B) irradiance when the denoiser is enabled, or the final
// mean*albedo*density lit result (bound directly to lightmap's own mip0/1/2) when it's disabled -
// see the file header.
@group(2) @binding(0) var combinedMip0: texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var combinedMip1: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var combinedMip2: texture_storage_2d<rgba16float, write>;
// rawVariance only exists (binding 3) when the denoiser is enabled - in the disabled/combine mode
// nothing downstream ever reads variance, so there's no reason to declare or write it (see the
// file header).
#ifndef COMBINE_ALBEDO_DENSITY
@group(2) @binding(3) var rawVariance: texture_storage_2d<r32float, write>;
#endif

const TILE_SIZE: u32 = 16u;
const THREAD_COUNT: u32 = TILE_SIZE * TILE_SIZE;

var<workgroup> sharedMeanA: array<vec3<f32>, THREAD_COUNT>;
var<workgroup> sharedMeanB: array<vec3<f32>, THREAD_COUNT>;
var<workgroup> sharedVariance: array<f32, THREAD_COUNT>;
// E[X]/E[X^2] of each pixel's own logLuminance((a+b)/2) - reduced through the same box-filter
// pyramid as everything else, purely so mip2 can recover Var(X) = E[X^2] - E[X]^2 over the full
// 4x4 (16 mip0 pixel) footprint at the end - see mip2's varianceOfMeans2 below for why this
// (spatial variance of each pixel's OWN mean) is a different, complementary quantity from
// sharedVariance's mean-of-(temporal A-vs-B)-variances.
var<workgroup> sharedLogMean: array<f32, THREAD_COUNT>;
var<workgroup> sharedLogMeanSq: array<f32, THREAD_COUNT>;
#ifdef COMBINE_ALBEDO_DENSITY
// Reduced through the same box-filter pyramid as sharedMeanA/B (survive1/survive2 below), rather
// than sampled from a separately-generated albedo/density mip chain - none is needed elsewhere
// when the denoiser is disabled, so this is the only place these mips are ever produced.
var<workgroup> sharedAlbedo: array<vec3<f32>, THREAD_COUNT>;
var<workgroup> sharedDensity: array<f32, THREAD_COUNT>;
#endif

// Mirrors filter_variance.wgsl's identically-valued LUMINANCE_FLOOR (and
// DEFAULT_DENOISER_TUNABLES.darknessNoiseFloor's role) - this shader has no uniforms of its own
// (same reasoning as filter_variance.wgsl's own copy), so it's a local constant rather than a
// wired-through tunable.
const LUMINANCE_FLOOR: f32 = 0.002;

// Log-luminance-domain squared difference, not a mean-normalized ratio (this function's previous
// version divided by mean^2 - see this project's denoiser plan for the full argument against that).
// This pipeline's noise isn't uniform shot noise: a laser's forward-scattered halo is fed by a
// handful of unevenly-weighted scattering events landing differently in A vs B, a fundamentally
// noisier regime (closer to a compound-Poisson process) than the broad ambient light's many small,
// evenly-distributed contributions - no fixed power of mean can rank the two correctly against each
// other, since the same mean brightness can hide very different noise depending on how concentrated
// its energy is. Comparing in log-luminance space instead (same perceptual-ratio reasoning as
// LitboxCommon.wgsl's logLuminance, already used for exactly this reason by denoise.wgsl's
// radianceWeight and build_denoiser_quadtree.wgsl's detail trigger) makes a fixed difference mean
// the same thing regardless of a pixel's absolute brightness, and LUMINANCE_FLOOR compresses the
// comparison near black so a sparsely-sampled dark pixel doesn't explode the way a raw ratio's
// near-zero-denominator blowup does.
fn relativeVariance(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let logDiff = logLuminance(luminance(a), LUMINANCE_FLOOR) - logLuminance(luminance(b), LUMINANCE_FLOOR);
    return logDiff * logDiff;
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let mip0Size = textureDimensions(combinedMip0);
    // Out-of-bounds threads (edge tiles when mip0's resolution isn't a multiple of 16) still
    // participate in every reduction stage below (workgroupBarrier requires uniform participation
    // - no thread may return early) - clamping to the nearest real texel here means they
    // contribute a duplicated edge value rather than garbage, a standard, minor edge softness
    // shared with any tiled reduction.
    let clampedCoord = min(id.xy, mip0Size - vec2<u32>(1u, 1u));

    let a = textureLoad(irradianceA, vec2<i32>(clampedCoord), 0).rgb;
    let b = textureLoad(irradianceB, vec2<i32>(clampedCoord), 0).rgb;
    let mean0 = (a + b) * 0.5;
    let variance0 = relativeVariance(a, b);
    let logMean0 = logLuminance(luminance(mean0), LUMINANCE_FLOOR);

    sharedMeanA[localIndex] = a;
    sharedMeanB[localIndex] = b;
    sharedVariance[localIndex] = variance0;
    sharedLogMean[localIndex] = logMean0;
    sharedLogMeanSq[localIndex] = logMean0 * logMean0;

#ifdef COMBINE_ALBEDO_DENSITY
    let albedo0 = textureLoad(albedo, vec2<i32>(clampedCoord), 0).rgb;
    let density0 = textureLoad(density, vec2<i32>(clampedCoord), 0).r / DENSITY_SCALE;
    sharedAlbedo[localIndex] = albedo0;
    sharedDensity[localIndex] = density0;
#endif

    if (id.x < mip0Size.x && id.y < mip0Size.y) {
#ifdef COMBINE_ALBEDO_DENSITY
        textureStore(combinedMip0, vec2<i32>(id.xy), vec4<f32>(mean0 * albedo0 * density0, 1.0));
#else
        textureStore(combinedMip0, vec2<i32>(id.xy), vec4<f32>(mean0, 1.0));
#endif
    }

    // --- mip1: 8x8 survivors, each averaging its own 2x2 group of mip0 values ---
    workgroupBarrier();
    let survive1 = (localId.x % 2u) == 0u && (localId.y % 2u) == 0u;
    var meanA1 = vec3<f32>(0.0);
    var meanB1 = vec3<f32>(0.0);
    var variance1 = 0.0;
    var logMean1 = 0.0;
    var logMeanSq1 = 0.0;
#ifdef COMBINE_ALBEDO_DENSITY
    var albedo1 = vec3<f32>(0.0);
    var density1 = 0.0;
#endif
    if (survive1) {
        let i00 = localIndex;
        let i10 = localIndex + 1u;
        let i01 = localIndex + TILE_SIZE;
        let i11 = localIndex + TILE_SIZE + 1u;
        meanA1 = (sharedMeanA[i00] + sharedMeanA[i10] + sharedMeanA[i01] + sharedMeanA[i11]) * 0.25;
        meanB1 = (sharedMeanB[i00] + sharedMeanB[i10] + sharedMeanB[i01] + sharedMeanB[i11]) * 0.25;
        // Pure propagation (mean of the 4 per-pixel temporal variances) - unchanged from before;
        // see mip2 below for where this gets combined with the spatial variance-of-means.
        variance1 = (sharedVariance[i00] + sharedVariance[i10] + sharedVariance[i01] + sharedVariance[i11]) * 0.25;
        // E[X]/E[X^2] of logMean over this 2x2 block - just propagated further, not yet turned into
        // a variance (that only happens once, at mip2, over the full 4x4 footprint - see below).
        logMean1 = (sharedLogMean[i00] + sharedLogMean[i10] + sharedLogMean[i01] + sharedLogMean[i11]) * 0.25;
        logMeanSq1 = (sharedLogMeanSq[i00] + sharedLogMeanSq[i10] + sharedLogMeanSq[i01] + sharedLogMeanSq[i11]) * 0.25;
#ifdef COMBINE_ALBEDO_DENSITY
        albedo1 = (sharedAlbedo[i00] + sharedAlbedo[i10] + sharedAlbedo[i01] + sharedAlbedo[i11]) * 0.25;
        density1 = (sharedDensity[i00] + sharedDensity[i10] + sharedDensity[i01] + sharedDensity[i11]) * 0.25;
#endif
    }
    workgroupBarrier();
    if (survive1) {
        sharedMeanA[localIndex] = meanA1;
        sharedMeanB[localIndex] = meanB1;
        sharedVariance[localIndex] = variance1;
        sharedLogMean[localIndex] = logMean1;
        sharedLogMeanSq[localIndex] = logMeanSq1;
#ifdef COMBINE_ALBEDO_DENSITY
        sharedAlbedo[localIndex] = albedo1;
        sharedDensity[localIndex] = density1;
#endif

        let mip1Size = textureDimensions(combinedMip1);
        let mip1Coord = id.xy / 2u;
        if (mip1Coord.x < mip1Size.x && mip1Coord.y < mip1Size.y) {
#ifdef COMBINE_ALBEDO_DENSITY
            let mean1 = (meanA1 + meanB1) * 0.5;
            textureStore(combinedMip1, vec2<i32>(mip1Coord), vec4<f32>(mean1 * albedo1 * density1, 1.0));
#else
            textureStore(combinedMip1, vec2<i32>(mip1Coord), vec4<f32>((meanA1 + meanB1) * 0.5, 1.0));
#endif
        }
    }

    // --- mip2 (quarter resolution): 4x4 survivors, each averaging a 2x2 group of mip1 values -
    // this is also where rawVariance is emitted (when it exists), matching
    // TracerPostProcessing.compute's _out_variance[id.xy/4] target resolution. This is the last
    // level this fused pass produces - see the file header for why (storage-texture-per-stage
    // limit, not a tile-size limit). ---
    workgroupBarrier();
    let survive2 = (localId.x % 4u) == 0u && (localId.y % 4u) == 0u;
    var meanA2 = vec3<f32>(0.0);
    var meanB2 = vec3<f32>(0.0);
    var variance2 = 0.0;
#ifdef COMBINE_ALBEDO_DENSITY
    var albedo2 = vec3<f32>(0.0);
    var density2 = 0.0;
#endif
    if (survive2) {
        let i00 = localIndex;
        let i10 = localIndex + 2u;
        let i01 = localIndex + 2u * TILE_SIZE;
        let i11 = localIndex + 2u * TILE_SIZE + 2u;
        meanA2 = (sharedMeanA[i00] + sharedMeanA[i10] + sharedMeanA[i01] + sharedMeanA[i11]) * 0.25;
        meanB2 = (sharedMeanB[i00] + sharedMeanB[i10] + sharedMeanB[i01] + sharedMeanB[i11]) * 0.25;
        // mean-of-variances: average of the 16 mip0 pixels' own (temporal, A-vs-B) variance -
        // catches a localized single-pixel spike, but degenerates under sparse sampling (both halves
        // can simultaneously read zero at a given pixel, misreading as agreement rather than "no
        // data yet").
        let meanOfVariances2 = (sharedVariance[i00] + sharedVariance[i10] + sharedVariance[i01] + sharedVariance[i11]) * 0.25;
        // variance-of-means: Var(X) = E[X^2] - E[X]^2 of the 16 mip0 pixels' own logLuminance(mean)
        // - a SPATIAL variance across the block, not a temporal A-vs-B comparison. Robust to the
        // above degeneracy: a lone photon landing in just one pixel's a_i (not necessarily matched
        // by a hit in that same pixel's b_i) still moves that pixel's own mean away from its
        // neighbors', showing up here even where the temporal comparison at that exact pixel read
        // as (spurious) agreement. max(), not a blend - same reasoning as decideBlurSize's
        // centerVariance/darknessShortfall combination: either signal being right is sufficient
        // justification to call this block noisy, and averaging would let a confident reading from
        // one get diluted by the other's blind spot. max(0.0, ...) guards float roundoff taking
        // E[X^2]-E[X]^2 very slightly negative for a near-perfectly-uniform block.
        let logMean2 = (sharedLogMean[i00] + sharedLogMean[i10] + sharedLogMean[i01] + sharedLogMean[i11]) * 0.25;
        let logMeanSq2 = (sharedLogMeanSq[i00] + sharedLogMeanSq[i10] + sharedLogMeanSq[i01] + sharedLogMeanSq[i11]) * 0.25;
        let varianceOfMeans2 = max(0.0, logMeanSq2 - logMean2 * logMean2);
        variance2 = max(meanOfVariances2, varianceOfMeans2);
#ifdef COMBINE_ALBEDO_DENSITY
        albedo2 = (sharedAlbedo[i00] + sharedAlbedo[i10] + sharedAlbedo[i01] + sharedAlbedo[i11]) * 0.25;
        density2 = (sharedDensity[i00] + sharedDensity[i10] + sharedDensity[i01] + sharedDensity[i11]) * 0.25;
#endif
    }
    workgroupBarrier();
    if (survive2) {
        let mip2Size = textureDimensions(combinedMip2);
        let mip2Coord = id.xy / 4u;
        if (mip2Coord.x < mip2Size.x && mip2Coord.y < mip2Size.y) {
#ifdef COMBINE_ALBEDO_DENSITY
            let mean2 = (meanA2 + meanB2) * 0.5;
            textureStore(combinedMip2, vec2<i32>(mip2Coord), vec4<f32>(mean2 * albedo2 * density2, 1.0));
#else
            textureStore(combinedMip2, vec2<i32>(mip2Coord), vec4<f32>((meanA2 + meanB2) * 0.5, 1.0));
            textureStore(rawVariance, vec2<i32>(mip2Coord), vec4<f32>(variance2, 0.0, 0.0, 0.0));
#endif
        }
    }
}
