// Normal-based edge detector - the sole point where the baked denoiser quadtree (see
// build_denoiser_quadtree.wgsl and this project's denoiser plan) touches the normal texture;
// every quadtree level above propagates this via max-reduction instead of re-deriving it.
// Structural port of the Unity reference's ComputeVolatilityLevel0.compute (a complete, working
// kernel - unlike Denoiser3.compute's disabled traversal).
//
// 4-neighborhood read without an array (see this project's CLAUDE.md - never index a
// function-local array literal by a runtime value); out-of-bounds textureLoads are well-defined
// in WGSL (return 0), which the dot(n,n) < 0.1 background check already treats as "no neighbor
// data" - no explicit edge clamping needed.
#include "LitboxCommon.wgsl"

@group(1) @binding(0) var normalRoughnessIn: texture_2d<f32>;

@group(2) @binding(0) var volatilityOut: texture_storage_2d<r32float, write>;

fn deviation(centerNormalized: vec3<f32>, neighborRaw: vec3<f32>) -> f32 {
    if (dot(neighborRaw, neighborRaw) < 0.1) {
        return 0.0;
    }
    return 1.0 - saturate(dot(centerNormalized, normalize(neighborRaw)));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(volatilityOut);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let normalCRaw = textureLoad(normalRoughnessIn, coords, 0).xyz;
    if (dot(normalCRaw, normalCRaw) < 0.1) {
        textureStore(volatilityOut, coords, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    let normalC = normalize(normalCRaw);

    let up = textureLoad(normalRoughnessIn, coords + vec2<i32>(0, 1), 0).xyz;
    let down = textureLoad(normalRoughnessIn, coords + vec2<i32>(0, -1), 0).xyz;
    let left = textureLoad(normalRoughnessIn, coords + vec2<i32>(-1, 0), 0).xyz;
    let right = textureLoad(normalRoughnessIn, coords + vec2<i32>(1, 0), 0).xyz;

    var volatility = deviation(normalC, up);
    volatility = max(volatility, deviation(normalC, down));
    volatility = max(volatility, deviation(normalC, left));
    volatility = max(volatility, deviation(normalC, right));

    textureStore(volatilityOut, coords, vec4<f32>(volatility, 0.0, 0.0, 0.0));
}
