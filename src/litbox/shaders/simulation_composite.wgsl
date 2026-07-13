// Additively composites the simulation's HDR lightmap into the frame buffer as a
// world-space quad (not a fullscreen triangle - the simulation region isn't necessarily
// screen-aligned). Samples the lightmap's base mip only; no exposure is applied here -
// exposure is a final tonemapping concern (see tonemap.wgsl).

#include "LitboxCommon.wgsl"

struct CameraUniform {
    viewProjection: mat4x4<f32>,
    simInverseWorldTransform: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct QuadUniform {
    worldTransform: mat4x4<f32>,
}
@group(1) @binding(0) var<uniform> quad: QuadUniform;
@group(1) @binding(1) var lightmapTex: texture_2d<f32>;
@group(1) @binding(2) var lightmapSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertex_main(@location(0) localPos: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    let world = quad.worldTransform * vec4<f32>(localPos, 0.0, 1.0);
    out.position = camera.viewProjection * world;
    // localPos is Y-up (local +Y = up, matching this project's world space and the G-Buffer
    // camera's own unflipped projection - see RaytracedResources.refreshViewProjection), but
    // WebGPU's texture V axis is fixed Y-down (V=0 = row 0 = NDC y=+1) - so local +Y (top) must
    // land on the lightmap's row 0, i.e. v=0. This single expression *is* that mapping, not a
    // naive uv patched afterward.
    out.uv = vec2<f32>(localPos.x + 0.5, 0.5 - localPos.y);
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(lightmapTex, lightmapSampler, in.uv, 0.0);
}
