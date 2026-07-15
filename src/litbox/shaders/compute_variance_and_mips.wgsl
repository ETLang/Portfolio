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
// combinedIrradiance (mip0..mip2) is the actual mean(A,B) signal used everywhere downstream -
// each level is a genuine box-filtered pyramid of A and of B *independently*, only combined into
// a mean at the point of writing each level's output texture - never a combination of
// already-computed variances (seemingly-plausible but not statistically valid - see this
// project's denoiser plan for why that distinction matters).
//
// rawVariance is the confirmed-live Unity relative-variance formula
// (dot(((a-b)^2/(mean^2+eps)),1/3)), computed once per pixel at mip0 from the raw (A,B) pair and
// then *propagated* (box-averaged stage-by-stage, exactly like Unity's kernel does) down to mip2
// (quarter resolution) alongside the mean reduction - this preserves localized-noise-spike
// information a from-scratch variance recompute at mip2 (from independently mip-chained A/B)
// would smooth away. Both signals are legitimate, different evidence ("how noisy is this pixel,
// smoothed for stability" vs. "how noisy is this region overall") - this pass only produces the
// first (matching Unity); the second is a plausible future evidence channel, not built here.
#include "LitboxCommon.wgsl"

@group(1) @binding(0) var irradianceA: texture_2d<f32>;
@group(1) @binding(1) var irradianceB: texture_2d<f32>;

@group(2) @binding(0) var combinedMip0: texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var combinedMip1: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var combinedMip2: texture_storage_2d<rgba16float, write>;
@group(2) @binding(3) var rawVariance: texture_storage_2d<r32float, write>;

const TILE_SIZE: u32 = 16u;
const THREAD_COUNT: u32 = TILE_SIZE * TILE_SIZE;

var<workgroup> sharedMeanA: array<vec3<f32>, THREAD_COUNT>;
var<workgroup> sharedMeanB: array<vec3<f32>, THREAD_COUNT>;
var<workgroup> sharedVariance: array<f32, THREAD_COUNT>;

fn relativeVariance(a: vec3<f32>, b: vec3<f32>, mean: vec3<f32>) -> f32 {
    let diff = a - b;
    return dot((diff * diff) / (mean * mean + 1e-5), vec3<f32>(1.0 / 3.0));
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
    let variance0 = relativeVariance(a, b, mean0);

    sharedMeanA[localIndex] = a;
    sharedMeanB[localIndex] = b;
    sharedVariance[localIndex] = variance0;

    if (id.x < mip0Size.x && id.y < mip0Size.y) {
        textureStore(combinedMip0, vec2<i32>(id.xy), vec4<f32>(mean0, 1.0));
    }

    // --- mip1: 8x8 survivors, each averaging its own 2x2 group of mip0 values ---
    workgroupBarrier();
    let survive1 = (localId.x % 2u) == 0u && (localId.y % 2u) == 0u;
    var meanA1 = vec3<f32>(0.0);
    var meanB1 = vec3<f32>(0.0);
    var variance1 = 0.0;
    if (survive1) {
        let i00 = localIndex;
        let i10 = localIndex + 1u;
        let i01 = localIndex + TILE_SIZE;
        let i11 = localIndex + TILE_SIZE + 1u;
        meanA1 = (sharedMeanA[i00] + sharedMeanA[i10] + sharedMeanA[i01] + sharedMeanA[i11]) * 0.25;
        meanB1 = (sharedMeanB[i00] + sharedMeanB[i10] + sharedMeanB[i01] + sharedMeanB[i11]) * 0.25;
        variance1 = (sharedVariance[i00] + sharedVariance[i10] + sharedVariance[i01] + sharedVariance[i11]) * 0.25;
    }
    workgroupBarrier();
    if (survive1) {
        sharedMeanA[localIndex] = meanA1;
        sharedMeanB[localIndex] = meanB1;
        sharedVariance[localIndex] = variance1;

        let mip1Size = textureDimensions(combinedMip1);
        let mip1Coord = id.xy / 2u;
        if (mip1Coord.x < mip1Size.x && mip1Coord.y < mip1Size.y) {
            textureStore(combinedMip1, vec2<i32>(mip1Coord), vec4<f32>((meanA1 + meanB1) * 0.5, 1.0));
        }
    }

    // --- mip2 (quarter resolution): 4x4 survivors, each averaging a 2x2 group of mip1 values -
    // this is also where rawVariance is emitted, matching TracerPostProcessing.compute's
    // _out_variance[id.xy/4] target resolution. This is the last level this fused pass produces -
    // see the file header for why (storage-texture-per-stage limit, not a tile-size limit). ---
    workgroupBarrier();
    let survive2 = (localId.x % 4u) == 0u && (localId.y % 4u) == 0u;
    var meanA2 = vec3<f32>(0.0);
    var meanB2 = vec3<f32>(0.0);
    var variance2 = 0.0;
    if (survive2) {
        let i00 = localIndex;
        let i10 = localIndex + 2u;
        let i01 = localIndex + 2u * TILE_SIZE;
        let i11 = localIndex + 2u * TILE_SIZE + 2u;
        meanA2 = (sharedMeanA[i00] + sharedMeanA[i10] + sharedMeanA[i01] + sharedMeanA[i11]) * 0.25;
        meanB2 = (sharedMeanB[i00] + sharedMeanB[i10] + sharedMeanB[i01] + sharedMeanB[i11]) * 0.25;
        variance2 = (sharedVariance[i00] + sharedVariance[i10] + sharedVariance[i01] + sharedVariance[i11]) * 0.25;
    }
    workgroupBarrier();
    if (survive2) {
        let mip2Size = textureDimensions(combinedMip2);
        let mip2Coord = id.xy / 4u;
        if (mip2Coord.x < mip2Size.x && mip2Coord.y < mip2Size.y) {
            textureStore(combinedMip2, vec2<i32>(mip2Coord), vec4<f32>((meanA2 + meanB2) * 0.5, 1.0));
            textureStore(rawVariance, vec2<i32>(mip2Coord), vec4<f32>(variance2, 0.0, 0.0, 0.0));
        }
    }
}
