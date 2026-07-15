// Bilateral filter of rawVariance (quarter resolution, from ComputeVarianceAndMipsOperation)
// using G-Buffer/irradiance mip2 evidence - structurally matches Unity's confirmed-live
// FilterVariance kernel (TracerPostProcessing.compute): a 5x5 spatial/albedo/adaptive-luminance
// cross filter, same sigma constants. Thresholds are TBD/tunable - see this project's denoiser
// plan; this is evidence-gathering plumbing, not the final size-argument/guided-blur algorithm.
//
// Unlike Unity's tile-plus-halo groupshared version, this is a plain per-pixel 5x5 gather (each
// invocation does its own textureLoads) - simpler and correct; a shared-memory tiled version is a
// possible later perf optimization once profiling calls for it, not needed for this pass.
#include "LitboxCommon.wgsl"

@group(1) @binding(0) var rawVarianceIn: texture_2d<f32>;
@group(1) @binding(1) var albedoMip2: texture_2d<f32>;
@group(1) @binding(2) var combinedIrradianceMip2: texture_2d<f32>;

@group(2) @binding(0) var filteredVarianceOut: texture_storage_2d<r32float, write>;

const SIGMA_SPATIAL: f32 = 1.2;
const SIGMA_ALBEDO: f32 = 0.05;
const SIGMA_LUMINANCE_TIGHT: f32 = 0.05;
const SIGMA_LUMINANCE_LOOSE: f32 = 2.5;
const K_LUMINANCE: f32 = 2.0;
// gaussianWeight/luminance/LUMINANCE_WEIGHTS live in LitboxCommon.wgsl - denoise.wgsl reuses them
// too (its darkness-evidence and radiance-similarity terms), see this project's denoiser plan.

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(filteredVarianceOut);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    let center = vec2<i32>(id.xy);
    let centerVariance = textureLoad(rawVarianceIn, center, 0).r;
    let centerAlbedo = textureLoad(albedoMip2, center, 0).rgb;
    let centerLuminance = luminance(textureLoad(combinedIrradianceMip2, center, 0).rgb);

    // Unity's adaptive luminance sigma: tighter (more selective) around already-low-variance
    // pixels, looser (more permissive, filters harder) as the center pixel's own variance grows.
    let sigmaAdaptive = mix(SIGMA_LUMINANCE_TIGHT, SIGMA_LUMINANCE_LOOSE, smoothstep(0.0, 1.0 / K_LUMINANCE, centerVariance));

    var totalWeight = 0.0;
    var accumulated = 0.0;
    for (var j = -2; j <= 2; j++) {
        for (var i = -2; i <= 2; i++) {
            let offset = vec2<i32>(i, j);
            let sampleCoord = center + offset;
            if (sampleCoord.x < 0 || sampleCoord.y < 0 || sampleCoord.x >= i32(size.x) || sampleCoord.y >= i32(size.y)) {
                continue;
            }

            let sampleVariance = textureLoad(rawVarianceIn, sampleCoord, 0).r;
            let sampleAlbedo = textureLoad(albedoMip2, sampleCoord, 0).rgb;
            let sampleLuminance = luminance(textureLoad(combinedIrradianceMip2, sampleCoord, 0).rgb);

            let spatialWeight = gaussianWeight(length(vec2<f32>(offset)), SIGMA_SPATIAL);
            let albedoWeight = gaussianWeight(length(centerAlbedo - sampleAlbedo), SIGMA_ALBEDO);
            let luminanceWeight = gaussianWeight(abs(centerLuminance - sampleLuminance), sigmaAdaptive);
            let sampleWeight = spatialWeight * albedoWeight * luminanceWeight;

            accumulated += sampleVariance * sampleWeight;
            totalWeight += sampleWeight;
        }
    }

    let filtered = select(centerVariance, accumulated / totalWeight, totalWeight > 0.0);
    textureStore(filteredVarianceOut, center, vec4<f32>(filtered, 0.0, 0.0, 0.0));
}
