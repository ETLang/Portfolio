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
// DENSITY_SCALE - not a literal Unity port. Rationale (confirmed empirically against
// float16 on real content, not just theory): transmittance clusters near 1 for typical
// (low-density) objects, which float16 resolves poorly right at its rounding boundary (a gap as
// large as 1e-6 below 1.0 can round to exactly 1.0). Reframing as density relocates that same
// information away from a boundary and into a region where floating point's relative precision
// actually pays off; DENSITY_SCALE then pushes it further away from float16's degraded subnormal
// range (values below ~6.1e-5) so it stays in the well-resolved normal range even for very thin
// objects. Must match RaytracedResources' DENSITY_SCALE exactly (also mirrored in
// debug_view_blit.wgsl for display) - chosen conservatively (not the theoretical max) to leave
// overflow headroom for the additive multi-object blend below.
const DENSITY_SCALE: f32 = 8192.0;
//
// The 3 render targets have different blend semantics (see RaytracedResources for the exact
// GPUColorTargetState configs and why): AlbedoAlpha blends "over" (order-dependent), Density
// blends additively (order-independent - this is an approximation of true multi-object
// combination, exact when only one object touches a given pixel and slightly overestimates
// combined density when several genuinely overlap; treated as acceptable given this is a flat 2D
// scene where "on top of" is already an artistic approximation, not literal depth), and
// NormalRoughness is drawn with no blending at all (an unconditional overwrite - last-drawn-object
// wins). Because that last one is an unconditional overwrite, an ellipse-masked fragment outside
// its shape must `discard` rather than write a transparent value - writing alpha=0 would still
// stomp whatever a lower object wrote to NormalRoughness at that pixel.
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

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // Base [0,1] quad-local UV, used for ellipse shape masking - NOT atlas-transformed.
    @location(0) uv: vec2<f32>,
    // uv passed through atlasTransform, used only for sampling mainTex.
    @location(1) atlasUv: vec2<f32>,
    // World-space face normal, heightScale NOT yet applied (applied in the fragment stage,
    // alongside particleAlignment, so this buffer's VERTEX-stage-only data stays untouched by
    // per-object property changes that don't move the object).
    @location(2) worldNormalUnscaled: vec3<f32>,
    @location(3) @interpolate(flat) propertiesIndex: u32,
}

@vertex
fn vertex_main(@builtin(instance_index) instanceIndex: u32, @location(0) localPos: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    let idx = raytracedIndices[instanceIndex];
    let worldTransform = transforms[idx.transformIndex];
    let atlasTransform = atlasTransforms[idx.atlasIndex];

    let world = worldTransform * vec4<f32>(localPos, 0.0, 1.0);
    out.position = camera.viewProjection * world;
    out.uv = localPos + vec2<f32>(0.5, 0.5);
    let atlasUvHomogeneous = vec3<f32>(out.uv, 1.0);
    out.atlasUv = vec2<f32>(dot(atlasUvHomogeneous, atlasTransform.row0.xyz), dot(atlasUvHomogeneous, atlasTransform.row1.xyz));
    // Matches sprite.wgsl's V-flip - see its comment for why (atlas packer's bottom-left-origin V
    // vs. this project's top-down texture upload/sampling convention).
    out.atlasUv.y = 1.0 - out.atlasUv.y;

    // The quad always faces local +Z (see quad_mesh.ts's confirmed CCW winding). Transforming
    // that constant local normal by the raw world matrix (w=0, so translation is ignored) is
    // mathematically exact for every object in this scene graph, not an approximation: every
    // world transform (scene_graph.ts) is built as translate * rotateZ * scale(sx, sy, 1) -
    // rotation is always Z-axis-only and Z-scale is always locked to 1, so neither operation
    // distorts or rotates the Z axis. No inverse-transpose correction is needed here.
    out.worldNormalUnscaled = (worldTransform * vec4<f32>(0.0, 0.0, 1.0, 0.0)).xyz;
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

    // Ellipse geometry mask. Must `discard`, not write alpha=0 - see file header comment on why
    // NormalRoughness's unconditional-overwrite semantics require this.
    if (props.primitiveShapeId == 2u) {
        let centered = (in.uv - vec2<f32>(0.5, 0.5)) * 2.0;
        if (length(centered) > 1.0) {
            discard;
        }
    }

    let c = textureSample(mainTex, mainSampler, in.atlasUv);

    let imageDensity = props.substrateDensity * c.a;
    // Clamped defensively (not a literal Unity port): substrateDensity*c.a can exceed 1 with
    // real data, which would otherwise drive the base of pow() negative - WGSL's pow() with a
    // negative base and non-integer exponent is undefined. Clamping degrades to t=0 instead of
    // propagating NaN into every consumer of this target.
    let imageTransmissibility = clamp(1.0 - imageDensity, 0.0, 1.0);
    let t = pow(imageTransmissibility, 100.0 / camera.targetHeightPixels);
    let scaledDensity = (1.0 - t) * DENSITY_SCALE;

    var out: GBufferOutput;
    out.albedo = vec4<f32>(c.rgb * props.albedo.rgb, 1.0) * c.a * props.albedo.a;
    out.density = vec4<f32>(scaledDensity, scaledDensity, 0.0, 0.0);
    out.normalRoughness = vec4<f32>(in.worldNormalUnscaled * props.heightScale, props.particleAlignment);
    return out;
}
