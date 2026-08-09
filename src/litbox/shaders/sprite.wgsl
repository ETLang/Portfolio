// Ported from the Unity reference shader PortfolioSpriteShader.shader.
//
//   baseColor = tex2D(_MainTex, uv)
//   light = tex2Dlod(_LightMap, lightUV, _LightDetail) * _LightMod + _Ambience
//   col = baseColor * light * _Color
//   col.rgb += _Emissive.rgb
//   col *= opacity                     (no Unity-material equivalent; final fade multiplier)
//
// _LightDetail (mip LOD) = simBlur. _LightMod = simContribution. _Ambience = ambient.
// _Color = colorMod. _Emissive = emissive. _Metallic is declared but unused in the
// reference - not ported.
//
// simBlur is authored (and CPU-side offset-corrected - see SpriteResources' writePropertiesData)
// as a discrete blur-size *level*, then split at upload time into simBlurBucket/simBlurLod - see
// this project's plan: "Fix lightmap sprite-blur pixelation on high-contrast content" and
// LightmapBlurCascade (lightmap_blur_cascade.ts). Level 0 is the sharp, unblurred simulation
// result (lightmapSharp, its own binding); levels 1..MAX_CASCADE_LEVELS are real, full-resolution
// Gaussian blurs of the previous level (lightmapCascade1/2/3/4, never decimated, so there's no
// resolution loss for high-contrast content like laser beams to alias against); levels past
// MAX_CASCADE_LEVELS fall back to lightmapMipChain, a real decimated mip chain continuing past the
// cascade. simBlurBucket selects which of those textures to sample (always at LOD 0 for the sharp/
// cascade bindings, since those are single-mip; simBlurLod only matters for the lightmapMipChain
// bucket) - a sprite picks one blur size and never blends between adjacent ones, matching the old
// LightmapBlurPyramid design's same "no blending" guarantee.
//
// lightUV is *adapted*, not ported verbatim: the reference derives it from screen-space
// NDC position (valid only because their camera happens to be aligned with the simulation's
// render target). Since the simulation isn't necessarily screen-aligned here, lightUV is
// instead derived from this sprite's world position transformed into the simulation's own
// local UV space via camera.simInverseWorldTransform.
//
// Every sprite's per-instance data lives in shared storage-buffer arrays (transform,
// properties, atlas UV transform), indexed indirectly through spriteIndices[instance_index] -
// never through a function-local array literal. See the project's uniform-array packing plan
// and this project's CLAUDE.md WGSL guidance: dynamically indexing a local array literal has
// silently corrupted output on some mobile GPU drivers with zero validation error, but
// storage-buffer reads keyed by a runtime index are a completely different, battle-tested
// driver code path.

#include "LitboxCommon.wgsl"

struct CameraUniform {
    viewProjection: mat4x4<f32>,
    simInverseWorldTransform: mat4x4<f32>,
    debugMode: f32,
}
@group(0) @binding(0) var<uniform> camera: CameraUniform;

// One entry per drawn sprite, in strict (layer, sortOrder) draw order - see SpriteResources'
// class doc. Points into the 3 shared arrays below, which are packed independently (static
// entries at the front, dynamic entries at the back - see PackedUniformArray) and so are *not*
// generally in the same order as spriteIndices itself.
struct SpriteIndex {
    transformIndex: u32,
    propertiesIndex: u32,
    atlasIndex: u32,
    _pad: u32,
}
// Shared across sprite/light/raytraced object kinds - see TransformResources. One entry per
// SceneObject that owns at least one such component, indexed by that owner's transform slot.
struct SpriteProperties {
    ambient: vec4<f32>,
    emissive: vec4<f32>,
    simContribution: vec4<f32>,
    colorMod: vec4<f32>,
    opacity: f32,
    // LOD within whichever texture simBlurBucket selects - always 0 for the sharp/cascade
    // bindings (all single-mip), only meaningful for the lightmapMipChain bucket.
    simBlurLod: f32,
    primitiveShapeId: u32,
    // Which lightmap texture to sample: 0 = lightmapSharp; 1..MAX_CASCADE_LEVELS = lightmapCascade1
    // ..MAX_CASCADE_LEVELS, all at LOD 0; MAX_CASCADE_LEVELS+1 (sentinel) = lightmapMipChain at
    // simBlurLod - see this file's header comment.
    simBlurBucket: u32,
}
// Maps this sprite's base [0,1] UV into its texture's sub-rectangle within a shared atlas:
// atlasUv = vec2(dot(vec3(uv, 1.0), row0.xyz), dot(vec3(uv, 1.0), row1.xyz)). A texture that
// isn't atlassed gets an identity transform (row0 = (1,0,0), row1 = (0,1,0)) from
// TextureCache, so this is applied unconditionally.
struct SpriteAtlasTransform {
    row0: vec4<f32>,
    row1: vec4<f32>,
}

@group(1) @binding(0) var<storage, read> spriteIndices: array<SpriteIndex>;
@group(1) @binding(1) var<storage, read> transforms: array<mat4x4<f32>>;
@group(1) @binding(2) var<storage, read> spriteProperties: array<SpriteProperties>;
@group(1) @binding(3) var<storage, read> atlasTransforms: array<SpriteAtlasTransform>;
@group(1) @binding(4) var mainSampler: sampler;

// The one thing that varies per draw call/batch - everything else in this shader is bound
// once per frame regardless of how many textures are in play.
@group(2) @binding(0) var mainTex: texture_2d<f32>;

// Binding 0: the sharp, unblurred lightmap (level 0). Bindings 1..4: LightmapBlurCascade's
// full-resolution blurred levels 1..MAX_CASCADE_LEVELS. Binding 5: the decimated mip chain past
// the cascade. Binding 6: one shared sampler - see SpriteResources' lightmapBindGroupLayout
// comment. MAX_CASCADE_LEVELS is hardcoded to 4 here since WGSL can't import a TS constant across
// the language boundary - must match sprite_resources.ts's MAX_CASCADE_LEVELS exactly.
@group(3) @binding(0) var lightmapSharp: texture_2d<f32>;
@group(3) @binding(1) var lightmapCascade1: texture_2d<f32>;
@group(3) @binding(2) var lightmapCascade2: texture_2d<f32>;
@group(3) @binding(3) var lightmapCascade3: texture_2d<f32>;
@group(3) @binding(4) var lightmapCascade4: texture_2d<f32>;
@group(3) @binding(5) var lightmapMipChain: texture_2d<f32>;
@group(3) @binding(6) var lightmapSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // Base [0,1] quad-local UV, used for shape masking - NOT atlas-transformed, since the
    // shape mask is defined in quad space regardless of which sub-rectangle of an atlas
    // this sprite's texture occupies.
    @location(0) uv: vec2<f32>,
    @location(1) worldPos: vec4<f32>,
    // uv passed through atlasTransform, used only for sampling mainTex.
    @location(2) atlasUv: vec2<f32>,
    // Flat (unlike position/uv/worldPos/atlasUv): an integer index selecting this instance's
    // properties, not an interpolated per-fragment value - every fragment of a given sprite's
    // quad reads the same properties entry.
    @location(3) @interpolate(flat) propertiesIndex: u32,
}

@vertex
fn vertex_main(@builtin(instance_index) instanceIndex: u32, @location(0) localPos: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    let idx = spriteIndices[instanceIndex];
    let worldTransform = transforms[idx.transformIndex];
    let atlasTransform = atlasTransforms[idx.atlasIndex];

    let world = worldTransform * vec4<f32>(localPos, 0.0, 1.0);
    out.worldPos = world;
    out.position = camera.viewProjection * world;
    out.uv = localPos + vec2<f32>(0.5, 0.5);
    let atlasUvHomogeneous = vec3<f32>(out.uv, 1.0);
    out.atlasUv = vec2<f32>(dot(atlasUvHomogeneous, atlasTransform.row0.xyz), dot(atlasUvHomogeneous, atlasTransform.row1.xyz));
    // The exported uvTransform is expressed in the atlas packer's own (Unity/OpenGL-style)
    // bottom-left-origin V, but our texture upload keeps the source PNG's row order as-is
    // (row 0 = top, matching WebGPU's top-left-origin sampling) - flip V to compensate.
    out.atlasUv.y = 1.0 - out.atlasUv.y;
    out.propertiesIndex = idx.propertiesIndex;
    return out;
}

// The Unity reference has no shape-masking concept at all - its quad is always fully
// opaque (alpha comes from _MainTex, which defaults to an opaque white texture). Only
// ellipse (2) needs an actual mask here, to approximate a circular sprite without a
// separate mesh; unspecified (0) and rect (1) both render the full quad, matching the
// reference's behavior.
//
// Antialiased (fwidth-based smoothstep), not a boolean cutoff: a hard-edged circle mask leaves
// visible rasterization gaps where two ellipse sprites' edges pass close to each other (a pixel
// can fall just outside both hard boundaries even though it's visually inside their union) -
// showed up as stray dark pixels where two overlapping cloud puffs' edges nearly meet.
fn shapeCoverage(shapeId: u32, uv: vec2<f32>) -> f32 {
    // fwidth computed unconditionally, outside the shapeId branch - WGSL requires derivatives to
    // be called from uniform control flow, and shapeId varies per-instance (non-uniform across a
    // draw call's fragments).
    let dist = length(uv - vec2<f32>(0.5, 0.5));
    let edgeWidth = max(fwidth(dist), 1e-5);
    let ellipseCoverage = 1.0 - smoothstep(0.5 - edgeWidth, 0.5 + edgeWidth, dist);
    return select(1.0, ellipseCoverage, shapeId == 2u);
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let props = spriteProperties[in.propertiesIndex];

    if (camera.debugMode > 0.5) {
        // Diagnostic: ignore opacity/shape/image entirely so geometry, transforms, and
        // camera placement can be verified independent of per-sprite data (e.g. opacity 0).
        var debugColor = vec3<f32>(0.2, 0.4, 1.0);
        if (props.primitiveShapeId == 1u) {
            debugColor = vec3<f32>(1.0, 0.2, 0.2);
        } else if (props.primitiveShapeId == 2u) {
            debugColor = vec3<f32>(0.2, 1.0, 0.2);
        }
        return vec4<f32>(debugColor, 1.0);
    }

    // mainTex is stored premultiplied (see TextureCache) to avoid dark edge fringing under
    // bilinear filtering - un-premultiply back to straight color for this shader's own math.
    let sampled = textureSample(mainTex, mainSampler, in.atlasUv);
    let baseColor = vec4<f32>(select(vec3<f32>(0.0), sampled.rgb / sampled.a, sampled.a > 0.0001), sampled.a);

    let simLocal = camera.simInverseWorldTransform * in.worldPos;
    var lightUV = simLocal.xy + vec2<f32>(0.5, 0.5);
    lightUV.y = 1.0f - lightUV.y;
    // simBlurBucket is flat (per-primitive-constant, see VertexOutput.propertiesIndex) and every
    // branch below uses textureSampleLevel's explicit LOD (no derivatives), so this non-uniform
    // branch across sprites in the same draw call is legal WGSL - see this project's plan.
    var lightSample: vec4<f32>;
    if (props.simBlurBucket == 0u) {
        lightSample = textureSampleLevel(lightmapSharp, lightmapSampler, lightUV, 0.0);
    } else if (props.simBlurBucket == 1u) {
        lightSample = textureSampleLevel(lightmapCascade1, lightmapSampler, lightUV, 0.0);
    } else if (props.simBlurBucket == 2u) {
        lightSample = textureSampleLevel(lightmapCascade2, lightmapSampler, lightUV, 0.0);
    } else if (props.simBlurBucket == 3u) {
        lightSample = textureSampleLevel(lightmapCascade3, lightmapSampler, lightUV, 0.0);
    } else if (props.simBlurBucket == 4u) {
        lightSample = textureSampleLevel(lightmapCascade4, lightmapSampler, lightUV, 0.0);
    } else {
        lightSample = textureSampleLevel(lightmapMipChain, lightmapSampler, lightUV, props.simBlurLod);
    }

    let light = lightSample * props.simContribution + props.ambient;
    var color = baseColor * light * props.colorMod;
    color = vec4<f32>(color.rgb + props.emissive.rgb, color.a);
    color = color * props.opacity;

    let coverage = shapeCoverage(props.primitiveShapeId, in.uv);
    let alpha = clamp(color.a, 0.0, 1.0) * coverage;
    return vec4<f32>(color.rgb, alpha);
}
