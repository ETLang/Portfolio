#include "Random.wgsl"

// Density is stored scaled by this factor so that precision is not lost on low-density areas in
// the FP16 GBuffer. Things that read density from the G Buffer need to divide the raw value by
// this scale.
const DENSITY_SCALE: f32 = 8192.0;

// Deliberately not array-indexed: some mobile GPU drivers (confirmed on a Pixel 10 Pro, both
// Chrome and Brave) silently corrupt geometry when a fullscreen quad's positions come from a
// WGSL array indexed by vertex_index. Branching instead of indexing works around it - see this
// project's CLAUDE.md WGSL guidance. For passes that draw a fullscreen quad as 2 triangles via
// draw(6) (tonemap.wgsl, debug_view_blit.wgsl) - NOT for passes that draw real mesh vertices
// (raytraced_gbuffer.wgsl, sprite.wgsl, simulation_composite.wgsl).
fn fullscreenQuadPosition(vertexIndex: u32) -> vec2<f32> {
    if (vertexIndex == 0u) {
        return vec2<f32>(-1.0, -1.0);
    } else if (vertexIndex == 1u) {
        return vec2<f32>(1.0, -1.0);
    } else if (vertexIndex == 2u) {
        return vec2<f32>(-1.0, 1.0);
    } else if (vertexIndex == 3u) {
        return vec2<f32>(-1.0, 1.0);
    } else if (vertexIndex == 4u) {
        return vec2<f32>(1.0, -1.0);
    } else {
        return vec2<f32>(1.0, 1.0);
    }
}

// Maps a clip space position to a top-left-origin [0,1] UV (V flipped to match this
// project's top-down texture upload/sampling convention - see texture_cache.ts).
fn clipSpaceToUv(pos: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
}

// LUTs (lookup tables - see lut.ts/lut_resources.ts) are procedural, static textures sampled with
// a texel-center remap: LUT-space u/v/w in [0,1] must land on the first/last texel's *center*,
// not the texture edge. `texelCount` must always come from a #define supplied by the consuming
// shader's own preprocessShader() call (e.g. `TEARDROP_SCATTERING_LUT_TEXEL_COUNT`, sourced from
// lut.ts's TEARDROP_SCATTERING_LUT_SAMPLES/BRDF_LUT_RESOLUTION) - never hardcode a LUT's
// resolution directly into a .wgsl file, since it's TS's job to be the single source of truth.
fn lutUv(u: f32, texelCount: f32) -> f32 {
    return 0.5 / texelCount + u * (1.0 - 1.0 / texelCount);
}

fn sampleLut1D(t: texture_2d<f32>, s: sampler, u: f32, texelCount: f32) -> vec4<f32> {
    return textureSampleLevel(t, s, vec2<f32>(lutUv(u, texelCount), 0.5), 0.0);
}

fn sampleLut2D(t: texture_2d<f32>, s: sampler, uv: vec2<f32>, texelCounts: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(t, s, vec2<f32>(lutUv(uv.x, texelCounts.x), lutUv(uv.y, texelCounts.y)), 0.0);
}

fn sampleLut3D(t: texture_3d<f32>, s: sampler, uvw: vec3<f32>, texelCounts: vec3<f32>) -> vec4<f32> {
    return textureSampleLevel(t, s, vec3<f32>(
        lutUv(uvw.x, texelCounts.x), lutUv(uvw.y, texelCounts.y), lutUv(uvw.z, texelCounts.z)), 0.0);
}
