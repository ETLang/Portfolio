// Generic single-mip-level box-filter downsample - reused for every mip chain in this project's
// denoiser evidence-gathering pipeline (the G-Buffer's Albedo/NormalRoughness, and the combined
// HDR irradiance/lightmap past mip4 - see this project's denoiser plan). One dispatch downsamples
// exactly one source mip level into the next; callers loop this for a whole chain.
//
// Sampling technique: a single textureSampleLevel tap with a linear sampler, at the destination
// texel's own UV - since the destination is exactly half the source's resolution in each
// dimension, that UV lands exactly on the shared corner of the corresponding source 2x2 block,
// giving the box-filter average in one fetch instead of four separate textureLoads. Valid since
// rgba8unorm/rgba16float/rg16float are all filterable by default in WebGPU (only 32-bit float
// formats need the float32-filterable feature).
//
// OUTPUT_FORMAT is a compile-time substitution (see MipDownsampleOperation.updateSwitches) -
// texture_storage_2d's texel format must be known at shader-compile time in WGSL, so it can't
// vary at runtime the way a regular sampled-texture binding can.
//
// Density (rg16float) can't go through this operation - see DensityMipBlitResources - rg16float
// isn't a valid WGSL storage-texture texel format, so a compute pass can't textureStore into it.
#include "LitboxCommon.wgsl"

@group(1) @binding(0) var sourceTex: texture_2d<f32>;
@group(1) @binding(1) var linearSampler: sampler;

@group(2) @binding(0) var output: texture_storage_2d<OUTPUT_FORMAT, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let value = textureSampleLevel(sourceTex, linearSampler, uv, 0.0);
    textureStore(output, vec2<i32>(id.xy), value);
}
