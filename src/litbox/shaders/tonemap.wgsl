// Final pass: HDR frame buffer -> swapchain. Fullscreen quad (this pass, unlike the simulation
// composite, genuinely is screen-aligned). Applies a UE5-style filmic tonemap:
// smoothstep(blackPoint, whitePoint, log10(x) + exposure).

#include "LitboxCommon.wgsl"

struct TonemapUniform {
    exposure: f32,
    // 0.0/1.0 rather than a bool - uniform buffer members can't be bool in WGSL.
    enabled: f32,
    // WGSL implicitly pads the following vec3<f32>s to 16-byte alignment - see tonemap.ts'
    // updateUniforms for the matching Float32Array layout.
    whitePointLog: vec3<f32>,
    blackPointLog: vec3<f32>,
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

// Alpha for compositing this pass over a preceding Background-bypass sprite pass, rather than
// overwriting it outright (see litbox_scene_renderer.ts's render()). Deliberately *not* the HDR
// buffer's own alpha: the simulation's additive composite covers almost the entire camera frustum
// with a single world-space quad regardless of how dark it is there (see
// simulation_composite.wgsl), so "did anything draw here" is ~always true and can't distinguish
// empty night sky from a lit ship - it would never let a Background sprite show through anywhere.
// The display-referred color's own brightness is the right signal instead: a pixel that tonemaps
// to (near) black has nothing to show *regardless of source*, so it's correct to treat it as
// transparent and let the background through; a pixel with real brightness represents real scene
// content and should occlude it. max(r,g,b) (not luminance's weighted sum) so a fully-saturated
// but dim-average color (e.g. pure blue) still counts as "something's there."
fn compositeAlpha(displayColor: vec3<f32>) -> f32 {
    return saturate(max(max(displayColor.r, displayColor.g), displayColor.b));
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdrTex, hdrSampler, in.uv).rgb;
    if (tonemapUniform.enabled < 0.5) {
        let ldr = linearToSrgb(hdr);
        return vec4<f32>(ldr, compositeAlpha(ldr));
    }
    var shape = toneMapDefaultShape();
    shape.exposure = tonemapUniform.exposure;
    shape.whitePoint = tonemapUniform.whitePointLog;
    shape.blackPoint = tonemapUniform.blackPointLog;
    let mapped = toneMapUE5(hdr, shape);
    // mapped is linear (the smoothstep curve, not a display transform on its own) - gamma-encode
    // before writing, since the swapchain's presentation format is never an "-srgb" variant (see
    // linearToSrgb's doc comment in LitboxCommon.wgsl).
    let ldr = linearToSrgb(mapped);
    return vec4<f32>(ldr, compositeAlpha(ldr));
}
