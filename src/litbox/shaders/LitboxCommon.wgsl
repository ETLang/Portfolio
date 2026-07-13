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
