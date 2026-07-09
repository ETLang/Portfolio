// Final pass: HDR frame buffer -> swapchain. Fullscreen triangle (this pass, unlike the
// simulation composite, genuinely is screen-aligned). Applies exposure; the precise
// tonemap operator is a placeholder (simple exposure-scaled clamp) - refined later.

struct TonemapUniform {
    exposure: f32,
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
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let pos = positions[vertexIndex];

    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdrTex, hdrSampler, in.uv).rgb * exp2(tonemapUniform.exposure);
    let mapped = clamp(hdr, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(mapped, 1.0);
}
