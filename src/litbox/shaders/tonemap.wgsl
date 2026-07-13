// Final pass: HDR frame buffer -> swapchain. Fullscreen quad (this pass, unlike the simulation
// composite, genuinely is screen-aligned). Applies a UE5-style filmic tonemap:
// smoothstep(blackPoint, whitePoint, log10(x) + exposure).

#include "LitboxCommon.wgsl"

struct TonemapUniform {
    exposure: f32,
    // 0.0/1.0 rather than a bool - uniform buffer members can't be bool in WGSL.
    enabled: f32,
}

struct ToneMappingShape {
    exposure: f32,
    whitePoint: vec3<f32>,
    blackPoint: vec3<f32>,
}

fn toneMapDefaultShape() -> ToneMappingShape {
    var shape: ToneMappingShape;
    shape.exposure = 0.0;
    shape.whitePoint = vec3<f32>(2.0);
    shape.blackPoint = vec3<f32>(-4.0);
    return shape;
}

// WGSL has no log10 builtin; derive it from log2.
fn log10(x: vec3<f32>) -> vec3<f32> {
    return log2(x) / log2(10.0);
}

// Analogous to UE5's standard tone mapping. Good general-purpose curve, but it makes
// things kinda feel like UE5...
fn toneMapUE5(x: vec3<f32>, shape: ToneMappingShape) -> vec3<f32> {
    return smoothstep(shape.blackPoint, shape.whitePoint, log10(x) + shape.exposure);
}

@group(0) @binding(0) var<uniform> tonemapUniform: TonemapUniform;
@group(0) @binding(1) var hdrTex: texture_2d<f32>;
@group(0) @binding(2) var hdrSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertex_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    let pos = fullscreenQuadPosition(vertexIndex);
    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = clipSpaceToUv(pos);
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdrTex, hdrSampler, in.uv).rgb;
    if (tonemapUniform.enabled < 0.5) {
        return vec4<f32>(hdr, 1.0);
    }
    var shape = toneMapDefaultShape();
    shape.exposure = tonemapUniform.exposure;
    let mapped = toneMapUE5(hdr, shape);
    return vec4<f32>(mapped, 1.0);
}
