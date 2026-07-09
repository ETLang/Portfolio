// Additively composites the simulation's HDR lightmap into the frame buffer as a
// world-space quad (not a fullscreen triangle - the simulation region isn't necessarily
// screen-aligned). Samples the lightmap's base mip only; no exposure is applied here -
// exposure is a final tonemapping concern (see tonemap.wgsl).

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
    out.uv = localPos + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(lightmapTex, lightmapSampler, in.uv, 0.0);
}
