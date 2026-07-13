// Converts the photon-receptor buffer's raw per-channel irradiance into the final HDR color by
// multiplying by the G-Buffer's albedo and density - single-frame port of Unity's ConvertToHDR
// kernel (ForwardMonteCarlo.compute). Not ported: the Unity original's multi-frame convergence/
// overflow-guard machinery (g_accumulated_output_hdr, g_needs_accumulation, g_batch_count_inv) -
// this is a realtime, single-frame pipeline, not an offline accumulator.
//
// photons holds 3 consecutive atomic<u32> entries per pixel (R, G, B), indexed
// (y * width + x) * 3 + channel. WGSL only allows atomic types inside a storage buffer declared
// read_write (never read-only, and never on a texture, unlike Unity's RWTexture2D<uint> original)
// - see this project's CLAUDE.md WGSL guidance for why a texture can't be used here.
//
// albedo/density are read with textureLoad (a direct texel fetch), not textureSample - the photon
// buffer and G-Buffer share resolution 1:1, exactly matching how Unity's ConvertToHDR indexes
// g_albedo/g_density directly by id.xy with no filtering.
#include "LitboxCommon.wgsl"

struct Uniforms {
    // Converts the raw fixed-point atomic accumulator back into a sane float range - see
    // ConvertPhotonIrradianceToHdrOperation for the exact formula (matches Unity's g_hdr_scale).
    hdrScale: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> photons: array<atomic<u32>>;
@group(1) @binding(1) var albedo: texture_2d<f32>;
@group(1) @binding(2) var density: texture_2d<f32>;

@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    let base = (id.y * size.x + id.x) * 3u;
    let irradiance = vec3<f32>(
        f32(atomicLoad(&photons[base])),
        f32(atomicLoad(&photons[base + 1u])),
        f32(atomicLoad(&photons[base + 2u])),
    ) * uniforms.hdrScale;

    let coords = vec2<i32>(id.xy);
    let albedoSample = textureLoad(albedo, coords, 0).rgb;
    let densitySample = textureLoad(density, coords, 0).r / DENSITY_SCALE;

    textureStore(output, coords, vec4<f32>(irradiance * albedoSample * densitySample, 1.0));
}
