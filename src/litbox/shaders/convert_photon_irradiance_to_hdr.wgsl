// Converts the photon-receptor buffer's raw per-channel irradiance into HDR color, one texture
// per half of this frame's two-way variance-estimation split (see this project's denoiser plan) -
// single-frame port of Unity's ConvertToHDR kernel (ForwardMonteCarlo.compute). Not ported: the
// Unity original's multi-frame convergence/overflow-guard machinery (g_accumulated_output_hdr,
// g_needs_accumulation, g_batch_count_inv) - this is a realtime, single-frame pipeline, not an
// offline accumulator.
//
// photons holds 6 consecutive atomic<u32> entries per pixel - two interleaved 3-wide (R,G,B)
// halves, indexed (y * width + x) * 6 + half*3 + channel - see forward_monte_carlo.wgsl's
// writePhotonIndexed, the writer using this same layout. WGSL only allows atomic types inside a
// storage buffer declared read_write (never read-only, and never on a texture, unlike Unity's
// RWTexture2D<uint> original) - see this project's CLAUDE.md WGSL guidance for why a texture
// can't be used here.
//
// albedo/density are read with textureLoad (a direct texel fetch), not textureSample - the photon
// buffer and G-Buffer share resolution 1:1, exactly matching how Unity's ConvertToHDR indexes
// g_albedo/g_density directly by id.xy with no filtering.
//
// combineAlbedoDensity is a compile-time switch (see ConvertPhotonIrradianceToHdrOperation.
// updateSwitches): the multiply now normally happens later, in the denoiser stage, after variance
// computation and denoising - not here - per this project's denoiser plan (albedo/density must be
// combined only once the final signal is settled). Kept toggleable here too, purely for debugging
// (comparing pre-denoise raw-lit output against the deferred path).
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

@group(2) @binding(0) var outputA: texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var outputB: texture_storage_2d<rgba16float, write>;

fn convertHalf(coords: vec2<i32>, base: u32, albedoSample: vec3<f32>, densitySample: f32) -> vec4<f32> {
    let irradiance = vec3<f32>(
        f32(atomicLoad(&photons[base])),
        f32(atomicLoad(&photons[base + 1u])),
        f32(atomicLoad(&photons[base + 2u])),
    ) * uniforms.hdrScale;

#ifdef COMBINE_ALBEDO_DENSITY
    return vec4<f32>(irradiance * albedoSample * densitySample, 1.0);
#else
    return vec4<f32>(irradiance, 1.0);
#endif
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(outputA);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let albedoSample = textureLoad(albedo, coords, 0).rgb;
    let densitySample = textureLoad(density, coords, 0).r / DENSITY_SCALE;

    let base = (id.y * size.x + id.x) * 6u;
    textureStore(outputA, coords, convertHalf(coords, base, albedoSample, densitySample));
    textureStore(outputB, coords, convertHalf(coords, base + 3u, albedoSample, densitySample));
}
