// Ported from the Unity reference shader PortfolioSpriteShader.shader.
//
//   baseColor = tex2D(_MainTex, uv)
//   light = tex2Dlod(_LightMap, lightUV, _LightDetail) * _LightMod + _Ambience
//   col = baseColor * light * _Color
//   col += _Emissive
//   col *= opacity                     (no Unity-material equivalent; final fade multiplier)
//
// _LightDetail (mip LOD) = simBlur. _LightMod = simContribution. _Ambience = ambient.
// _Color = colorMod. _Emissive = emissive. _Metallic is declared but unused in the
// reference - not ported.
//
// The output alpha is clamped to [0, 1]: the ported formula sums baseColor.a * light.a +
// emissive.a, which can exceed 1 with real exported data. An alpha above 1 flows straight
// into the SrcAlpha/OneMinusSrcAlpha blend (this HDR target isn't clamped like an 8-bit
// one), driving OneMinusSrcAlpha negative - overlapping sprites subtractively cancel
// instead of blending. Alpha is a coverage value for blending purposes; clamping it here
// doesn't change the RGB math above.
//
// lightUV is *adapted*, not ported verbatim: the reference derives it from screen-space
// NDC position (valid only because their camera happens to be aligned with the simulation's
// render target). Since the simulation isn't necessarily screen-aligned here, lightUV is
// instead derived from this sprite's world position transformed into the simulation's own
// local UV space via camera.simInverseWorldTransform.

struct CameraUniform {
    viewProjection: mat4x4<f32>,
    simInverseWorldTransform: mat4x4<f32>,
    debugMode: f32,
}
@group(0) @binding(0) var<uniform> camera: CameraUniform;

// Transform and properties are separate uniforms (not one combined struct) so a
// dynamically-marked ancestor's transform cascade can rewrite just the small transform
// uniform without touching the (usually static) properties uniform, and vice versa for
// a sprite whose own color/opacity/etc. is marked dynamic independent of its transform.
struct SpriteTransform {
    worldTransform: mat4x4<f32>,
}
struct SpriteProperties {
    ambient: vec4<f32>,
    emissive: vec4<f32>,
    simContribution: vec4<f32>,
    colorMod: vec4<f32>,
    opacity: f32,
    simBlur: f32,
    primitiveShapeId: u32,
}
// Maps this sprite's base [0,1] UV into its texture's sub-rectangle within a shared atlas:
// atlasUv = vec2(dot(vec3(uv, 1.0), row0.xyz), dot(vec3(uv, 1.0), row1.xyz)). A texture that
// isn't atlassed gets an identity transform (row0 = (1,0,0), row1 = (0,1,0)) from
// TextureCache, so this is applied unconditionally.
struct SpriteAtlasTransform {
    row0: vec4<f32>,
    row1: vec4<f32>,
}
@group(1) @binding(0) var<uniform> spriteTransform: SpriteTransform;
@group(1) @binding(1) var<uniform> spriteProperties: SpriteProperties;
@group(1) @binding(2) var mainTex: texture_2d<f32>;
@group(1) @binding(3) var mainSampler: sampler;
@group(1) @binding(4) var<uniform> atlasTransform: SpriteAtlasTransform;

@group(2) @binding(0) var lightmapTex: texture_2d<f32>;
@group(2) @binding(1) var lightmapSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // Base [0,1] quad-local UV, used for shape masking - NOT atlas-transformed, since the
    // shape mask is defined in quad space regardless of which sub-rectangle of an atlas
    // this sprite's texture occupies.
    @location(0) uv: vec2<f32>,
    @location(1) worldPos: vec4<f32>,
    // uv passed through atlasTransform, used only for sampling mainTex.
    @location(2) atlasUv: vec2<f32>,
}

@vertex
fn vertex_main(@location(0) localPos: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    let world = spriteTransform.worldTransform * vec4<f32>(localPos, 0.0, 1.0);
    out.worldPos = world;
    out.position = camera.viewProjection * world;
    out.uv = localPos + vec2<f32>(0.5, 0.5);
    let atlasUvHomogeneous = vec3<f32>(out.uv, 1.0);
    out.atlasUv = vec2<f32>(dot(atlasUvHomogeneous, atlasTransform.row0.xyz), dot(atlasUvHomogeneous, atlasTransform.row1.xyz));
    // The exported uvTransform is expressed in the atlas packer's own (Unity/OpenGL-style)
    // bottom-left-origin V, but our texture upload keeps the source PNG's row order as-is
    // (row 0 = top, matching WebGPU's top-left-origin sampling) - flip V to compensate.
    out.atlasUv.y = 1.0 - out.atlasUv.y;
    return out;
}

// The Unity reference has no shape-masking concept at all - its quad is always fully
// opaque (alpha comes from _MainTex, which defaults to an opaque white texture). Only
// ellipse (2) needs an actual mask here, to approximate a circular sprite without a
// separate mesh; unspecified (0) and rect (1) both render the full quad, matching the
// reference's behavior.
fn shapeAlpha(shapeId: u32, uv: vec2<f32>) -> f32 {
    if (shapeId == 2u) {
        let centered = (uv - vec2<f32>(0.5, 0.5)) * 2.0;
        return select(0.0, 1.0, length(centered) <= 1.0);
    }
    return 1.0;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (camera.debugMode > 0.5) {
        // Diagnostic: ignore opacity/shape/image entirely so geometry, transforms, and
        // camera placement can be verified independent of per-sprite data (e.g. opacity 0).
        var debugColor = vec3<f32>(0.2, 0.4, 1.0);
        if (spriteProperties.primitiveShapeId == 1u) {
            debugColor = vec3<f32>(1.0, 0.2, 0.2);
        } else if (spriteProperties.primitiveShapeId == 2u) {
            debugColor = vec3<f32>(0.2, 1.0, 0.2);
        }
        return vec4<f32>(debugColor, 1.0);
    }

    let baseColor = textureSample(mainTex, mainSampler, in.atlasUv);

    let simLocal = camera.simInverseWorldTransform * in.worldPos;
    let lightUV = simLocal.xy + vec2<f32>(0.5, 0.5);
    let lightSample = textureSampleLevel(lightmapTex, lightmapSampler, lightUV, spriteProperties.simBlur);

    let light = lightSample * spriteProperties.simContribution + spriteProperties.ambient;
    var color = baseColor * light * spriteProperties.colorMod;
    color = color + spriteProperties.emissive;
    color = color * spriteProperties.opacity;

    let alpha = clamp(shapeAlpha(spriteProperties.primitiveShapeId, in.uv) * color.a, 0.0, 1.0);
    return vec4<f32>(color.rgb, alpha);
}
