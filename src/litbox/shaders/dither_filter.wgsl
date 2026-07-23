// Small guided post-filter over the final (post-denoise, post-albedo/density-combine) lit image -
// cleans up the per-pixel dither noise that denoise.wgsl's temporal seedJitter trades the
// coherent lattice-alignment Moire for (see this project's denoiser plan / conversation history).
// A 5x5 cross-bilateral filter guided by G-Buffer structural similarity (albedo/normal/density),
// same three terms as denoise.wgsl's decideWeight structuralWeight - reused here because a sample
// only deserves to be averaged in if it's plausibly the same surface, exactly the same reasoning
// that applies there. No radiance-similarity term (unlike decideWeight): the whole point here is
// to average away *residual pixel-to-pixel noise* in the final color itself, so gating on that
// same color's own similarity would suppress the filter's own effect.
//
// Deliberately much cheaper than denoise.wgsl's hierarchical guided blur: fixed small radius, no
// traversal/quadtree, plain per-pixel textureLoads (same simplicity tradeoff as
// filter_variance.wgsl's own 5x5 gather) - this is meant to polish denoise's already-converged
// output, not replace its own blur-size/split logic.
#include "LitboxCommon.wgsl"

@group(1) @binding(0) var lightmapIn: texture_2d<f32>;
@group(1) @binding(1) var albedo: texture_2d<f32>;
@group(1) @binding(2) var normalRoughness: texture_2d<f32>;
@group(1) @binding(3) var density: texture_2d<f32>;

@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

const RADIUS: i32 = 2; // 5x5
const SIGMA_SPATIAL: f32 = 1.5;
// Matches denoise.wgsl's DEFAULT_DENOISER_TUNABLES defaults (albedoSensitivity/normalSensitivity/
// densitySensitivity) - same structural-similarity tolerances, reused rather than re-derived,
// since both filters are asking the same "is this plausibly the same surface" question.
const ALBEDO_SENSITIVITY: f32 = 0.3;
const NORMAL_SENSITIVITY: f32 = 8.0;
const DENSITY_SENSITIVITY: f32 = 1.0;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    let center = vec2<i32>(id.xy);
    let centerColor = textureLoad(lightmapIn, center, 0).rgb;
    let centerAlbedo = textureLoad(albedo, center, 0).rgb;
    let centerNormal = textureLoad(normalRoughness, center, 0).xyz;
    let centerDensityValue = textureLoad(density, center, 0).r / DENSITY_SCALE;
    let centerOpticalDepth = opticalDepth(centerDensityValue);

    var accumulated = vec3<f32>(0.0, 0.0, 0.0);
    var totalWeight = 0.0;
    for (var j = -RADIUS; j <= RADIUS; j++) {
        for (var i = -RADIUS; i <= RADIUS; i++) {
            let offset = vec2<i32>(i, j);
            let sampleCoord = center + offset;
            if (sampleCoord.x < 0 || sampleCoord.y < 0 || sampleCoord.x >= i32(size.x) || sampleCoord.y >= i32(size.y)) {
                continue;
            }

            let sampleColor = textureLoad(lightmapIn, sampleCoord, 0).rgb;
            let sampleAlbedo = textureLoad(albedo, sampleCoord, 0).rgb;
            let sampleNormal = textureLoad(normalRoughness, sampleCoord, 0).xyz;
            let sampleDensityValue = textureLoad(density, sampleCoord, 0).r / DENSITY_SCALE;

            let spatialWeight = gaussianWeight(length(vec2<f32>(offset)), SIGMA_SPATIAL);

            var albedoWeight = 1.0 - saturate(distance(centerAlbedo, sampleAlbedo) / ALBEDO_SENSITIVITY);
            albedoWeight *= albedoWeight;

            let normalDot = saturate(dot(normalize(centerNormal), normalize(sampleNormal)));
            let normalWeight = pow(normalDot, NORMAL_SENSITIVITY);

            let opticalDepthDiff = abs(centerOpticalDepth - opticalDepth(sampleDensityValue));
            var densityWeight = 1.0 - saturate(opticalDepthDiff / DENSITY_SENSITIVITY);
            densityWeight *= densityWeight;

            let weight = spatialWeight * albedoWeight * normalWeight * densityWeight;
            accumulated += sampleColor * weight;
            totalWeight += weight;
        }
    }

    let filtered = select(centerColor, accumulated / totalWeight, totalWeight > 0.0);
    textureStore(output, center, vec4<f32>(filtered, 1.0));
}
