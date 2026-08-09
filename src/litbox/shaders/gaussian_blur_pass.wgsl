// Single-axis separable Gaussian blur pass - the constant-resolution real-blur building block for
// LightmapBlurCascade (lightmap_blur_cascade.ts). Used both for the persisted cascade levels and
// for the calibrated bridging pass that hands off into the (separate, untouched)
// GaussianPyramidDownsampleOperation decimation tail - see lightmap_blur_cascade.ts's doc comment
// for the full design and why a real, densely-sampled blur is used here instead of widening the
// decimation kernel's stride.
//
// radius/sigma are runtime uniform values, not compile-time constants, so the tap loop below is a
// genuine `for` loop over a runtime-bounded index - not a dynamically-indexed array literal, so
// CLAUDE.md's WGSL array-indexing gotcha doesn't apply (matches filter_variance.wgsl's identical
// loop-based-not-table-based pattern). Two dispatches (direction = (1,0) then (0,1), horizontal
// then vertical) make a full 2D blur - see LightmapBlurCascade for how the two are chained through
// a shared scratch texture.
#include "LitboxCommon.wgsl"

struct Uniforms {
    direction: vec2<f32>, // texel step direction for this pass: (1,0) horizontal, (0,1) vertical
    radius: i32,          // tap radius in texels - loop runs [-radius, radius]
    sigma: f32,           // Gaussian sigma for this pass, in texels
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var sourceTex: texture_2d<f32>;

@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

fn clampedLoad(sourceSize: vec2<i32>, coord: vec2<i32>) -> vec4<f32> {
    return textureLoad(sourceTex, clamp(coord, vec2<i32>(0), sourceSize - vec2<i32>(1)), 0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let outputSize = textureDimensions(output);
    if (id.x >= outputSize.x || id.y >= outputSize.y) {
        return;
    }

    let sourceSize = vec2<i32>(textureDimensions(sourceTex));
    let center = vec2<i32>(id.xy);
    let step = vec2<i32>(uniforms.direction);

    var sum = vec4<f32>(0.0);
    var weightSum = 0.0;
    for (var i = -uniforms.radius; i <= uniforms.radius; i++) {
        let weight = gaussianWeight(f32(i), uniforms.sigma);
        sum += weight * clampedLoad(sourceSize, center + step * i);
        weightSum += weight;
    }

    textureStore(output, center, sum / max(weightSum, 1e-5));
}
