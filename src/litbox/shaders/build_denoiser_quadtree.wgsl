// Baked min/max-range quadtree over the G-Buffer (+ volatility + an irradiance-detail trigger) -
// backs denoise.wgsl's ShouldSplit(). See this project's denoiser plan for the full argument.
// Structurally ported from the Unity reference's ComputeDenoiserQuadtreeLevel0.compute/
// ComputeDenoiserQuadtree.compute - the two complete, internally-consistent kernels in that
// project (unlike Denoiser3.compute's disabled traversal) - extended with the irradiance-detail
// OR term, which those kernels don't have.
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
    // G-Buffer/irradiance-space mip this dispatch is building evidence for (this operation's own
    // output mip + 1) - used for the irradiance-detail trigger's mip pair (this mip vs. this mip
    // minus one). Updated every dispatch (see BuildDenoiserQuadtreeOperation.updateUniforms),
    // unlike the threshold fields above which stay constant across the whole chain.
    currentGBufferMip: f32,
}
@group(0) @binding(0) var<uniform> uniforms: BuildQuadtreeUniforms;

#ifdef LEVEL0
@group(1) @binding(0) var albedoIn: texture_2d<f32>;
@group(1) @binding(1) var densityIn: texture_2d<f32>;
@group(1) @binding(2) var volatilityIn: texture_2d<f32>;
@group(1) @binding(3) var combinedIrradianceIn: texture_2d<f32>;
@group(1) @binding(4) var filteredVarianceIn: texture_2d<f32>;
@group(1) @binding(5) var linearSampler: sampler;
#else
@group(1) @binding(0) var prevAlbedoMin: texture_2d<f32>;
@group(1) @binding(1) var prevAlbedoMax: texture_2d<f32>;
@group(1) @binding(2) var prevDensityMinMaxVolatility: texture_2d<f32>;
@group(1) @binding(3) var prevQuadtreeMustSplit: texture_2d<f32>;
@group(1) @binding(4) var combinedIrradianceIn: texture_2d<f32>;
@group(1) @binding(5) var filteredVarianceIn: texture_2d<f32>;
@group(1) @binding(6) var linearSampler: sampler;
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
#else
    let albedoMinA = textureLoad(prevAlbedoMin, srcCoords, 0).rgb;
    let albedoMinB = textureLoad(prevAlbedoMin, srcCoords + vec2<i32>(1, 0), 0).rgb;
    let albedoMinC = textureLoad(prevAlbedoMin, srcCoords + vec2<i32>(0, 1), 0).rgb;
    let albedoMinD = textureLoad(prevAlbedoMin, srcCoords + vec2<i32>(1, 1), 0).rgb;
    let albedoMin = min(min(albedoMinA, albedoMinB), min(albedoMinC, albedoMinD));

    let albedoMaxA = textureLoad(prevAlbedoMax, srcCoords, 0).rgb;
    let albedoMaxB = textureLoad(prevAlbedoMax, srcCoords + vec2<i32>(1, 0), 0).rgb;
    let albedoMaxC = textureLoad(prevAlbedoMax, srcCoords + vec2<i32>(0, 1), 0).rgb;
    let albedoMaxD = textureLoad(prevAlbedoMax, srcCoords + vec2<i32>(1, 1), 0).rgb;
    let albedoMax = max(max(albedoMaxA, albedoMaxB), max(albedoMaxC, albedoMaxD));

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
#endif

    let albedoLumaDeviation = albedoMax.x - albedoMin.x;
    let albedoChromaDeviation = length(albedoMax.yz - albedoMin.yz);
    let densityDeviation = densityMax - densityMin;

    // Irradiance-detail trigger: a Laplacian-pyramid-style comparison of combinedIrradiance
    // against itself one level finer, at the SAME uv - see this project's denoiser plan. Catches
    // features with no G-Buffer signature at all (e.g. a laser beam through uniform haze), which
    // the range checks above can never see: combinedIrradiance's mip pyramid is a genuine box
    // filter, so a real feature causes a real difference between adjacent-mip point samples at the
    // same location, while a flat region's adjacent mips agree almost exactly. Gated by nearby
    // variance (a higher effective threshold where variance is high) so MC noise in a genuinely
    // flat, under-sampled region doesn't masquerade as detail - this gating is an open tuning
    // question, not a fully solved one; expect to revisit it empirically.
    let outUv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(outSize);
    let coarseLuminance = luminance(textureSampleLevel(combinedIrradianceIn, linearSampler, outUv, uniforms.currentGBufferMip).rgb);
    let fineLuminance = luminance(textureSampleLevel(combinedIrradianceIn, linearSampler, outUv, uniforms.currentGBufferMip - 1.0).rgb);
    let detail = abs(coarseLuminance - fineLuminance) / (fineLuminance + 1e-4);

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
    textureStore(albedoMinOut, outCoords, vec4<f32>(albedoMin, 1.0));
    textureStore(albedoMaxOut, outCoords, vec4<f32>(albedoMax, 1.0));
    textureStore(densityMinMaxVolatilityOut, outCoords, vec4<f32>(densityMin, densityMax, volatilityMax, 0.0));
    textureStore(quadtreeMustSplitOut, outCoords, vec4<f32>(select(0.0, 1.0, mustSplit), 0.0, 0.0, 0.0));
}
