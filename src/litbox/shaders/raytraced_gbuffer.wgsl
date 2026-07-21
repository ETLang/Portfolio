// Ported from the Unity reference shader RTObjectMat.shader, rasterizing every raytraced scene
// object into 3 G-Buffer render targets that a future raytracing/path-tracing pass will sample:
//
//   albedo      = premultiplied(albedoMap.rgb * albedo.rgb, albedoMap.a * albedo.a)
//   density     = (1 - pow(1 - substrateDensity * albedoMap.a, 100 / targetHeightPixels)) * DENSITY_SCALE
//   normalRoughness = (worldNormal * heightScale, particleAlignment)
//
// substrateDensity = pow(10, logDensity) and particleAlignment = 1 - roughness are precomputed
// CPU-side (see RaytracedResources) to match the Unity reference's own CPU-computed
// _substrateDensity/_particleAlignment uniforms. The Unity source also samples a `_NormalTex`
// (sdfNormalMap) but never uses the result (confirmed dead code) - not ported here.
//
// The Density target stores *density* (1 - transmittance), not transmittance itself, scaled by
// DENSITY_SCALE (see LitboxCommon.wgsl for the full rationale and where else it's used) - not a
// literal Unity port.
#include "LitboxCommon.wgsl"
//
// The 3 render targets have different blend semantics (see RaytracedResources for the exact
// GPUColorTargetState configs and why): AlbedoAlpha blends "over" (order-dependent), Density
// blends additively (order-independent - this is an approximation of true multi-object
// combination, exact when only one object touches a given pixel and slightly overestimates
// combined density when several genuinely overlap; treated as acceptable given this is a flat 2D
// scene where "on top of" is already an artistic approximation, not literal depth), and
// NormalRoughness is drawn with no blending at all (an unconditional overwrite - last-drawn-object
// wins). Shape silhouettes (rect/ellipse/unspecified) come entirely from which mesh region
// RaytracedResources draws (see primitive_mesh.ts) - a fragment is never emitted outside an
// object's own footprint in the first place, so no discard-based masking is needed here.
//
// Every per-instance value is looked up through storage-buffer indirection
// (raytracedIndices[instance_index] -> transforms/raytracedProperties/atlasTransforms), never a
// function-local array literal indexed by a runtime value - see this project's CLAUDE.md WGSL
// guidance (confirmed silent geometry corruption on some mobile GPU drivers for the latter).

struct GBufferCamera {
    // scale(2,2,1) * inverse(simulation owner's world transform) - maps world space into the
    // simulation's own [-0.5,0.5]^2 local rect, then into WebGPU NDC [-1,1]^2. See
    // RaytracedResources.refreshViewProjection.
    viewProjection: mat4x4<f32>,
    // Height (in texels) of the G-Buffer render targets, for the screen/target-height-normalized
    // transmissibility exponent below - matches RTObjectMat.shader's `_ScreenParams.y`.
    targetHeightPixels: f32,
}
@group(0) @binding(0) var<uniform> camera: GBufferCamera;

// One entry per drawn raytraced object, in ascending sortOrder draw order - see
// RaytracedResources' class doc. Points into the 3 shared arrays below.
struct RaytracedIndex {
    transformIndex: u32,
    propertiesIndex: u32,
    atlasIndex: u32,
    _pad: u32,
}
struct RaytracedProperties {
    albedo: vec4<f32>,
    substrateDensity: f32,
    particleAlignment: f32,
    heightScale: f32,
    primitiveShapeId: u32,
}
// Maps this object's base [0,1] UV into its texture's sub-rectangle within a shared atlas -
// identical layout/semantics to SpriteAtlasTransform in sprite.wgsl.
struct RaytracedAtlasTransform {
    row0: vec4<f32>,
    row1: vec4<f32>,
}

@group(1) @binding(0) var<storage, read> raytracedIndices: array<RaytracedIndex>;
@group(1) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>; // shared TransformResources buffer
@group(1) @binding(2) var<storage, read> raytracedProperties: array<RaytracedProperties>;
@group(1) @binding(3) var<storage, read> atlasTransforms: array<RaytracedAtlasTransform>;
@group(1) @binding(4) var mainSampler: sampler;

@group(2) @binding(0) var mainTex: texture_2d<f32>; // albedoMap only - see file header.

// Transforms a local-space normal into world space, correctly under non-uniform scale, without a
// general matrix inverse. Every world transform in this project (see scene_graph.ts) is built as
// translate * rotateZ(theta) * scale(sx, sy, 1), so worldTransform's upper-left 3x3 is exactly
// R(theta) * diag(sx, sy, 1) - its columns are col0 = sx*(cos th, sin th, 0), col1 =
// sy*(-sin th, cos th, 0), col2 = (0,0,1) (Z is never scaled or rotated out of axis, so col2 is
// always exactly (0,0,1)). The mathematically correct normal transform is the inverse-transpose of
// that 3x3 (mirroring Unity's UnityObjectToWorldNormal), which works out to
// R(theta) * diag(1/sx, 1/sy, 1) applied to the local normal - equivalently, in closed form:
//   col0*(vx/sx^2) + col1*(vy/sy^2) + col2*vz
// with sx^2/sy^2 recovered via dot(col.xy, col.xy) rather than a separate length()+division. For
// an axis-aligned local normal with vz=1, vx=vy=0 (the flat-quad case) this reduces to exactly
// col2 = (0,0,1), matching the old naive `(worldTransform * vec4(n,0)).xyz` exactly - it's only
// non-axis-aligned local normals (RTEllipse's rim, see primitive_mesh.ts) under non-uniform scale
// where this differs from (and corrects) that naive approach: the naive transform multiplies by
// scale instead of dividing by it, which only preserves direction for axis-aligned vectors.
fn computeWorldNormal(worldTransform: mat4x4<f32>, localNormal: vec3<f32>) -> vec3<f32> {
    let col0 = worldTransform[0].xyz;
    let col1 = worldTransform[1].xyz;
    let col2 = worldTransform[2].xyz;
    let sx2 = dot(col0.xy, col0.xy);
    let sy2 = dot(col1.xy, col1.xy);
    let worldDir = col0 * (localNormal.x / sx2) + col1 * (localNormal.y / sy2) + col2 * localNormal.z;
    return normalize(worldDir);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // uv passed through atlasTransform, used only for sampling mainTex.
    @location(0) atlasUv: vec2<f32>,
    // World-space face normal, heightScale NOT yet applied (applied in the fragment stage,
    // alongside particleAlignment, so this buffer's VERTEX-stage-only data stays untouched by
    // per-object property changes that don't move the object). Varies per-vertex (not per-object)
    // for rect/ellipse - see computeWorldNormal and primitive_mesh.ts.
    @location(1) worldNormalUnscaled: vec3<f32>,
    @location(2) @interpolate(flat) propertiesIndex: u32,
}

@vertex
fn vertex_main(
    @builtin(instance_index) instanceIndex: u32,
    @location(0) localPos: vec2<f32>,
    @location(1) localNormal: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    let idx = raytracedIndices[instanceIndex];
    let worldTransform = transforms[idx.transformIndex];
    let atlasTransform = atlasTransforms[idx.atlasIndex];

    let world = worldTransform * vec4<f32>(localPos, 0.0, 1.0);
    out.position = camera.viewProjection * world;
    let uv = localPos + vec2<f32>(0.5, 0.5);
    let atlasUvHomogeneous = vec3<f32>(uv, 1.0);
    out.atlasUv = vec2<f32>(dot(atlasUvHomogeneous, atlasTransform.row0.xyz), dot(atlasUvHomogeneous, atlasTransform.row1.xyz));
    // Matches sprite.wgsl's V-flip - see its comment for why (atlas packer's bottom-left-origin V
    // vs. this project's top-down texture upload/sampling convention).
    out.atlasUv.y = 1.0 - out.atlasUv.y;

    out.worldNormalUnscaled = computeWorldNormal(worldTransform, localNormal);
    out.propertiesIndex = idx.propertiesIndex;
    return out;
}

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    // Scaled density (1-transmittance)*DENSITY_SCALE - see file header. R=G duplicate the
    // scalar value (matches AlbedoAlpha/NormalRoughness's RGBA shape for a uniform MRT layout);
    // B and A are unused (always 0, additive-blend-neutral).
    @location(1) density: vec4<f32>,
    @location(2) normalRoughness: vec4<f32>,
}

@fragment
fn fragment_main(in: VertexOutput) -> GBufferOutput {
    let props = raytracedProperties[in.propertiesIndex];

    let c = textureSample(mainTex, mainSampler, in.atlasUv);

    let imageDensity = props.substrateDensity * c.a;
    // Clamped defensively (not a literal Unity port): substrateDensity*c.a can exceed 1 with
    // real data, which would otherwise drive the base of pow() negative - WGSL's pow() with a
    // negative base and non-integer exponent is undefined. Clamping degrades to t=0 instead of
    // propagating NaN into every consumer of this target.
    let imageTransmissibility = clamp(1.0 - imageDensity, 0.0, 1.0);
    let t = pow(imageTransmissibility, 100.0 / camera.targetHeightPixels);
    let scaledDensity = (1.0 - t) * DENSITY_SCALE;

    // Y-flipped to match simulation-target-pixel-space (see forward_monte_carlo.ts's
    // computeWorldToTargetPixels yFlip), not left as raw world space: this is a direction, not a
    // position, so it isn't corrected by the position-space yFlip baked into the G-buffer's own
    // viewProjection/lightToTarget transforms. World +Y (up) is pixel-space -Y (since pixel/UV
    // space is top-down), so forward_monte_carlo.wgsl's scatterMaterially - which dots and
    // reflects this normal against photon directions that live entirely in pixel space - would
    // otherwise silently reflect/transmit photons across the wrong side of every non-horizontal
    // surface. Unity's original never needed this flip because its pixel space was already
    // Y-up, matching world space with no flip at all.
    let normal = in.worldNormalUnscaled * props.heightScale;
    var out: GBufferOutput;
    out.albedo = vec4<f32>(c.rgb * props.albedo.rgb, 1.0) * c.a * props.albedo.a;
    out.density = vec4<f32>(scaledDensity, scaledDensity, 0.0, 0.0);
    out.normalRoughness = vec4<f32>(normal.x, -normal.y, normal.z, props.particleAlignment);
    return out;
}
