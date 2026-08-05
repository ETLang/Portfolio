// Baked min/max-range quadtree over the G-Buffer (+ volatility + an irradiance range check) -
// backs denoise.wgsl's ShouldSplit(). See this project's denoiser plan for the full argument.
// Structurally ported from the Unity reference's ComputeDenoiserQuadtreeLevel0.compute/
// ComputeDenoiserQuadtree.compute - the two complete, internally-consistent kernels in that
// project (unlike Denoiser3.compute's disabled traversal) - extended with the irradiance range
// term, which those kernels don't have.
//
// Sizing/indexing: this operation's 4 output textures (albedoMin/Max, densityMinMaxVolatility,
// quadtreeMustSplit) are allocated at HALF the G-Buffer's resolution, with their OWN 0-indexed mip
// chain - level i of these textures answers "should a G-Buffer/irradiance-space mip (i+1) node
// split into its mip-i children" (see denoise.wgsl's shouldSplit(), which does the +1 translation
// at the one place it's needed). LEVEL0 (compile-time switch, see
// BuildDenoiserQuadtreeOperation.updateSwitches) reads the raw G-Buffer's mip0 (a genuine 2x2
// texel block) plus volatility (mip0 only, see compute_volatility.wgsl) to produce this
// operation's own mip 0. The #else (iterative) variant reads the PREVIOUS level's own
// min/max/volatility/mustSplit outputs (also a 2x2 block) and ORs children's mustSplit into its
// own - volatility is never re-derived from the normal texture past level 0, only propagated
// upward via max-reduction (mirrors the Unity reference exactly).
//
// irradianceLumaMin/Max ride along in albedoMinOut.a/albedoMaxOut.a (otherwise-unused - the
// albedo range check only ever reads/writes .rgb) rather than their own dedicated storage
// textures: WebGPU only *guarantees* 4 storage textures per compute stage
// (maxStorageTexturesPerShaderStage) - this shader is already at that limit with albedoMin/Max,
// densityMinMaxVolatility, and quadtreeMustSplit, so 2 more dedicated outputs would only run on
// hardware that happens to expose a higher limit (confirmed: this adapter allows 8, but the
// WebGPU-guaranteed minimum is 4 - see compute_variance_and_mips.wgsl's file header for the same
// constraint already being respected elsewhere in this pipeline).
//
// Two permanently-separate operation instances back this shader (one LEVEL0=true, one
// LEVEL0=false, see BuildDenoiserQuadtreeOperation) rather than one instance whose switch flips
// every frame - a switch change is a full pipeline recompile (see ComputeOperation), and this
// runs every frame, so oscillating it per-frame would recompile twice a frame for nothing. Same
// reasoning as SimulationResources' separate mipDownsampleAlbedo/mipDownsample instances.
#include "LitboxCommon.wgsl"

struct BuildQuadtreeUniforms {
    albedoLuminanceThreshold: f32,
    albedoChromaThreshold: f32,
    logDensityThreshold: f32,
    volatilityThreshold: f32,
    detailThreshold: f32,
    varianceGateScale: f32,
    // Floors the logLuminance() singularity at 0 for the irradiance range check below - shared with
    // DenoiseUniforms' identically-named field (denoise.wgsl), same role it plays there for the
    // guided blur's radiance-similarity term.
    darknessNoiseFloor: f32,
}
@group(0) @binding(0) var<uniform> uniforms: BuildQuadtreeUniforms;

#ifdef LEVEL0
@group(1) @binding(0) var albedoIn: texture_2d<f32>;
@group(1) @binding(1) var densityIn: texture_2d<f32>;
@group(1) @binding(2) var volatilityIn: texture_2d<f32>;
@group(1) @binding(3) var combinedIrradianceIn: texture_2d<f32>;
@group(1) @binding(4) var filteredVarianceIn: texture_2d<f32>;
#else
@group(1) @binding(0) var prevAlbedoMin: texture_2d<f32>;
@group(1) @binding(1) var prevAlbedoMax: texture_2d<f32>;
@group(1) @binding(2) var prevDensityMinMaxVolatility: texture_2d<f32>;
@group(1) @binding(3) var prevQuadtreeMustSplit: texture_2d<f32>;
@group(1) @binding(4) var filteredVarianceIn: texture_2d<f32>;
#endif

@group(2) @binding(0) var albedoMinOut: texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var albedoMaxOut: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var densityMinMaxVolatilityOut: texture_storage_2d<rgba16float, write>;
@group(2) @binding(3) var quadtreeMustSplitOut: texture_storage_2d<r32float, write>;

// Luma/chroma decorrelation for the albedo range check - reuses this project's own luminance()
// (Rec.709) for the luma channel, for consistency with the luma metric denoise.wgsl already uses
// everywhere else. (The Unity reference's "RGBtoYCoCg" was actually mislabeled BT.601/YCbCr
// coefficients, not real YCoCg - not worth replicating exactly, see this project's denoiser plan.)
fn albedoLumaChroma(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(luminance(rgb), rgb.r - rgb.b, rgb.g - 0.5 * (rgb.r + rgb.b));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let outSize = textureDimensions(quadtreeMustSplitOut);
    if (id.x >= outSize.x || id.y >= outSize.y) {
        return;
    }

    let srcCoords = vec2<i32>(id.xy) * 2;

#ifdef LEVEL0
    let albedoA = albedoLumaChroma(textureLoad(albedoIn, srcCoords, 0).rgb);
    let albedoB = albedoLumaChroma(textureLoad(albedoIn, srcCoords + vec2<i32>(1, 0), 0).rgb);
    let albedoC = albedoLumaChroma(textureLoad(albedoIn, srcCoords + vec2<i32>(0, 1), 0).rgb);
    let albedoD = albedoLumaChroma(textureLoad(albedoIn, srcCoords + vec2<i32>(1, 1), 0).rgb);
    let albedoMin = min(min(albedoA, albedoB), min(albedoC, albedoD));
    let albedoMax = max(max(albedoA, albedoB), max(albedoC, albedoD));

    let densityA = opticalDepth(textureLoad(densityIn, srcCoords, 0).r / DENSITY_SCALE);
    let densityB = opticalDepth(textureLoad(densityIn, srcCoords + vec2<i32>(1, 0), 0).r / DENSITY_SCALE);
    let densityC = opticalDepth(textureLoad(densityIn, srcCoords + vec2<i32>(0, 1), 0).r / DENSITY_SCALE);
    let densityD = opticalDepth(textureLoad(densityIn, srcCoords + vec2<i32>(1, 1), 0).r / DENSITY_SCALE);
    let densityMin = min(min(densityA, densityB), min(densityC, densityD));
    let densityMax = max(max(densityA, densityB), max(densityC, densityD));

    let volatilityA = textureLoad(volatilityIn, srcCoords, 0).r;
    let volatilityB = textureLoad(volatilityIn, srcCoords + vec2<i32>(1, 0), 0).r;
    let volatilityC = textureLoad(volatilityIn, srcCoords + vec2<i32>(0, 1), 0).r;
    let volatilityD = textureLoad(volatilityIn, srcCoords + vec2<i32>(1, 1), 0).r;
    let volatilityMax = max(max(volatilityA, volatilityB), max(volatilityC, volatilityD));

    // Irradiance range check (this project's denoiser plan): a min/max spread over the SAME 2x2
    // block every other channel here already reduces over, applied to raw irradiance luminance
    // instead of albedo/density. Catches features with no G-Buffer signature at all (e.g. a laser
    // beam through uniform haze), which the range checks above can never see.
    //
    // Replaces an earlier, structurally broken attempt at this (a Laplacian-pyramid-style
    // comparison of a coarse mip against a bilinear sample of the next-finer mip, at the SAME uv):
    // every mip level in this pipeline is *defined* as the box-filtered average of the level below
    // it, and bilinear-sampling the finer level at the exact uv corresponding to a coarser texel's
    // own center reconstructs precisely the same average used to produce that coarser texel -
    // coarse and "fine" were therefore always ~equal by construction, so that version's `detail`
    // was ~0 everywhere regardless of any real feature (confirmed: even with the variance gate
    // below forced to zero, it never fired at the laser beam's brightest pixel). A genuine min/max
    // range over the block's own individual texels has no such degeneracy - a real feature confined
    // to part of the block (the beam covering one corner, say) shows up as a wide min/max spread,
    // exactly like a material or density edge does.
    let irradianceA = luminance(textureLoad(combinedIrradianceIn, srcCoords, 0).rgb);
    let irradianceB = luminance(textureLoad(combinedIrradianceIn, srcCoords + vec2<i32>(1, 0), 0).rgb);
    let irradianceC = luminance(textureLoad(combinedIrradianceIn, srcCoords + vec2<i32>(0, 1), 0).rgb);
    let irradianceD = luminance(textureLoad(combinedIrradianceIn, srcCoords + vec2<i32>(1, 1), 0).rgb);
    let irradianceLumaMin = min(min(irradianceA, irradianceB), min(irradianceC, irradianceD));
    let irradianceLumaMax = max(max(irradianceA, irradianceB), max(irradianceC, irradianceD));
#else
    let albedoMinRawA = textureLoad(prevAlbedoMin, srcCoords, 0);
    let albedoMinRawB = textureLoad(prevAlbedoMin, srcCoords + vec2<i32>(1, 0), 0);
    let albedoMinRawC = textureLoad(prevAlbedoMin, srcCoords + vec2<i32>(0, 1), 0);
    let albedoMinRawD = textureLoad(prevAlbedoMin, srcCoords + vec2<i32>(1, 1), 0);
    let albedoMin = min(min(albedoMinRawA.rgb, albedoMinRawB.rgb), min(albedoMinRawC.rgb, albedoMinRawD.rgb));

    let albedoMaxRawA = textureLoad(prevAlbedoMax, srcCoords, 0);
    let albedoMaxRawB = textureLoad(prevAlbedoMax, srcCoords + vec2<i32>(1, 0), 0);
    let albedoMaxRawC = textureLoad(prevAlbedoMax, srcCoords + vec2<i32>(0, 1), 0);
    let albedoMaxRawD = textureLoad(prevAlbedoMax, srcCoords + vec2<i32>(1, 1), 0);
    let albedoMax = max(max(albedoMaxRawA.rgb, albedoMaxRawB.rgb), max(albedoMaxRawC.rgb, albedoMaxRawD.rgb));

    let dmvA = textureLoad(prevDensityMinMaxVolatility, srcCoords, 0);
    let dmvB = textureLoad(prevDensityMinMaxVolatility, srcCoords + vec2<i32>(1, 0), 0);
    let dmvC = textureLoad(prevDensityMinMaxVolatility, srcCoords + vec2<i32>(0, 1), 0);
    let dmvD = textureLoad(prevDensityMinMaxVolatility, srcCoords + vec2<i32>(1, 1), 0);
    let densityMin = min(min(dmvA.x, dmvB.x), min(dmvC.x, dmvD.x));
    let densityMax = max(max(dmvA.y, dmvB.y), max(dmvC.y, dmvD.y));
    let volatilityMax = max(max(dmvA.z, dmvB.z), max(dmvC.z, dmvD.z));

    let splitA = textureLoad(prevQuadtreeMustSplit, srcCoords, 0).r != 0.0;
    let splitB = textureLoad(prevQuadtreeMustSplit, srcCoords + vec2<i32>(1, 0), 0).r != 0.0;
    let splitC = textureLoad(prevQuadtreeMustSplit, srcCoords + vec2<i32>(0, 1), 0).r != 0.0;
    let splitD = textureLoad(prevQuadtreeMustSplit, srcCoords + vec2<i32>(1, 1), 0).r != 0.0;
    let childSplit = splitA || splitB || splitC || splitD;

    // Irradiance range propagated the same way every other range channel is: min/max-reduce the
    // previous level's own min/max outputs over this level's 2x2 block - never re-derived from raw
    // combinedIrradiance past level 0, exactly like volatility. Carried in albedoMin/Max's
    // otherwise-unused alpha channel (see this file's header) rather than a dedicated texture.
    let irradianceLumaMin = min(min(albedoMinRawA.a, albedoMinRawB.a), min(albedoMinRawC.a, albedoMinRawD.a));
    let irradianceLumaMax = max(max(albedoMaxRawA.a, albedoMaxRawB.a), max(albedoMaxRawC.a, albedoMaxRawD.a));
#endif

    let albedoLumaDeviation = albedoMax.x - albedoMin.x;
    let albedoChromaDeviation = length(albedoMax.yz - albedoMin.yz);
    let densityDeviation = densityMax - densityMin;

    // Log-domain (see LitboxCommon.wgsl's logLuminance): a fixed detailThreshold can only mean the
    // same thing across scenes whose absolute brightness scales are similar if the comparison is a
    // ratio, not a raw magnitude difference.
    let detail = logLuminance(irradianceLumaMax, uniforms.darknessNoiseFloor) - logLuminance(irradianceLumaMin, uniforms.darknessNoiseFloor);

    let outUv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(outSize);
    let varianceSize = vec2<i32>(textureDimensions(filteredVarianceIn));
    let varianceCoords = clamp(vec2<i32>(outUv * vec2<f32>(varianceSize)), vec2<i32>(0), varianceSize - vec2<i32>(1));
    let nearbyVariance = textureLoad(filteredVarianceIn, varianceCoords, 0).r;
    let effectiveDetailThreshold = uniforms.detailThreshold * (1.0 + nearbyVariance * uniforms.varianceGateScale);

    let localSplit = albedoLumaDeviation > uniforms.albedoLuminanceThreshold ||
        albedoChromaDeviation > uniforms.albedoChromaThreshold ||
        densityDeviation > uniforms.logDensityThreshold ||
        volatilityMax > uniforms.volatilityThreshold ||
        detail > effectiveDetailThreshold;

#ifdef LEVEL0
    let mustSplit = localSplit;
#else
    let mustSplit = localSplit || childSplit;
#endif

    let outCoords = vec2<i32>(id.xy);
    textureStore(albedoMinOut, outCoords, vec4<f32>(albedoMin, irradianceLumaMin));
    textureStore(albedoMaxOut, outCoords, vec4<f32>(albedoMax, irradianceLumaMax));
    textureStore(densityMinMaxVolatilityOut, outCoords, vec4<f32>(densityMin, densityMax, volatilityMax, 0.0));
    textureStore(quadtreeMustSplitOut, outCoords, vec4<f32>(select(0.0, 1.0, mustSplit), 0.0, 0.0, 0.0));
}
