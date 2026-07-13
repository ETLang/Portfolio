// Debug-only pass: blits any single registered debug view's source texture to the swapchain,
// applying a per-mode display transform (see debug_view.ts's DEBUG_VIEW_MODE) - not part of the
// normal render path, only used when LitboxSceneRenderer.debugView is set. This shader itself
// knows nothing about *which* subsystem a view comes from (G-Buffer, or anything registered
// later, e.g. the lightmap once the simulation pass is real) - it just interprets whatever `mode`
// the caller passes alongside the source texture. Modes 0-3 below happen to all currently be
// contributed by RaytracedResources' G-Buffer targets (see raytraced_gbuffer.wgsl), but nothing
// here is G-Buffer-specific - a future debug view can reuse an existing mode (e.g. mode 0's plain
// passthrough works for any displayable RGBA source) or extend this switch with a new one.
//
//   - mode 0 (passthrough): already displayable RGBA as-is.
//   - mode 1 (density): the source stores (1-transmittance)*DENSITY_SCALE (see
//     raytraced_gbuffer.wgsl's file header for why it's density, not raw transmittance), so
//     undoing the scale directly recovers a density-like value that's ~0 for thin/empty objects
//     and grows toward 1 for dense ones - displayed dense=white, thin=black.
//   - mode 2 (normal remap): worldNormal*heightScale has negative components and isn't unit
//     length; the standard *0.5+0.5 remap makes negative components visible instead of clipping
//     to black.
//   - mode 3 (alpha as luminance): displays the source's alpha channel as grayscale luminance,
//     not as an actual alpha/transparency channel - e.g. RaytracedResources' NormalRoughness
//     target's particleAlignment, which lives in alpha.
//
// Uses textureSampleLevel with a non-filtering sampler (not textureSample with a filtering one)
// for exact per-texel inspection (nearest-neighbor, no interpolation) - appropriate for a debug
// view regardless of which source's format is bound.

#include "LitboxCommon.wgsl"

struct DebugViewUniform {
    mode: u32,
    // Divisor applied before clamping to [0,1] - only mode 1 (density) currently consumes this;
    // other modes ignore it. Tunable live (see LitboxSceneRenderer.debugViewScale) since the
    // "typical" value range depends entirely on the active scene, which isn't knowable in advance
    // from this shader alone.
    scale: f32,
}
@group(0) @binding(0) var<uniform> debugUniform: DebugViewUniform;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var srcSampler: sampler;

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
    let c = textureSampleLevel(srcTex, srcSampler, in.uv, 0.0);

    if (debugUniform.mode == 0u) {
        return vec4<f32>(c.rgb, 1.0);
    } else if (debugUniform.mode == 1u) {
        let density = c.r / DENSITY_SCALE;
        let normalized = clamp(density / debugUniform.scale, 0.0, 1.0);
        return vec4<f32>(vec3<f32>(normalized), 1.0);
    } else if (debugUniform.mode == 2u) {
        let remapped = clamp(c.rgb * 0.5 + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(remapped, 1.0);
    } else {
        return vec4<f32>(vec3<f32>(c.a), 1.0);
    }
}
