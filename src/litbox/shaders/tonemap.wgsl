// Final pass: HDR frame buffer -> swapchain. Fullscreen triangle (this pass, unlike the
// simulation composite, genuinely is screen-aligned). Applies a UE5-style filmic tonemap:
// smoothstep(blackPoint, whitePoint, log10(x) + exposure).

struct TonemapUniform {
    exposure: f32,
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
    // Deliberately not array-indexed: some mobile GPU drivers (confirmed on a Pixel 10
    // Pro, both Chrome and Brave) silently corrupt geometry when a fullscreen-quad's
    // positions come from a WGSL array indexed by vertex_index. Branching instead of
    // indexing works around it.
    var pos: vec2<f32>;
    if (vertexIndex == 0u) {
        pos = vec2<f32>(-1.0, -1.0);
    } else if (vertexIndex == 1u) {
        pos = vec2<f32>(1.0, -1.0);
    } else if (vertexIndex == 2u) {
        pos = vec2<f32>(-1.0, 1.0);
    } else if (vertexIndex == 3u) {
        pos = vec2<f32>(-1.0, 1.0);
    } else if (vertexIndex == 4u) {
        pos = vec2<f32>(1.0, -1.0);
    } else {
        pos = vec2<f32>(1.0, 1.0);
    }

    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdrTex, hdrSampler, in.uv).rgb;
    var shape = toneMapDefaultShape();
    shape.exposure = tonemapUniform.exposure;
    let mapped = toneMapUE5(hdr, shape);
    return vec4<f32>(mapped, 1.0);
}
