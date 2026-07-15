// Density's mip-chain generation - a render-pass sibling of mip_downsample.wgsl (see this
// project's denoiser plan). Density is stored rg16float, which isn't a valid WGSL
// storage-texture texel format, so a compute pass can't textureStore into it the way
// MipDownsampleOperation does for Albedo/NormalRoughness/the combined irradiance - a
// render-attachment write has no such restriction, hence this one dedicated render pipeline.
//
// Same box-filter technique as mip_downsample.wgsl: sampling the source mip at this fragment's
// own UV (which, for a fullscreen quad over a destination exactly half the source's resolution,
// lands precisely on the shared corner of the corresponding source 2x2 block) with a linear
// sampler gives the box-filter average in one fetch.
#include "LitboxCommon.wgsl"

@group(0) @binding(0) var sourceTex: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;

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
    return textureSample(sourceTex, linearSampler, in.uv);
}
