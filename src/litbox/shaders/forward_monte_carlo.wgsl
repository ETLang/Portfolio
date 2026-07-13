// Port of Unity's ForwardMonteCarlo.compute + SimulationCommon.cginc (single-pass forward photon
// tracer - see this project's plan/CLAUDE.md for what's deliberately not ported: accumulation
// mode, the quadtree jump optimization, field lights, ScatterMie, the importance-map-guided
// scatter experiments, and Simulate_DefaultLight).
//
// One compute pipeline per light kind, selected at shader-compile time by exactly one of
// LIGHT_KIND_POINT/_SPOT/_LASER/_DIRECTIONAL/_AMBIENT being #define'd (see
// ForwardMonteCarloOperation, which owns one instance per kind) - each block below supplies the
// single emitLight() this compiled instance needs; unlike Unity's separate kernel entry points
// this project selects the kernel body via the WGSL preprocessor instead.
//
// Dispatched once per light instance (not deduplicated by kind - see SimulationResources), with
// @workgroup_size(64,1,1) matching Unity's NUMTHREADS_1D so ray counts (always rounded up to a
// multiple of 64 by the caller) divide evenly into whole workgroups.
#include "LitboxCommon.wgsl"

struct Uniforms {
    // world -> simulation-target-pixel-space transform, already combined with this light's own
    // world transform (SimulationResources) - column-vector convention (M * v), no transpose
    // needed unlike the HLSL original (mul(v, M)).
    lightToTarget: mat4x4<f32>,
    lightEnergy: vec3<f32>,
    bounces: u32,
    // This light's offset into the shared g_rand buffer - see ComputedDataManager.acquireRandomSeedBuffer.
    // Deliberate improvement over Unity's literal behavior: Unity always indexes g_rand[id.x] from
    // 0 for every light, so two lights dispatched the same frame stomp/advance the same RNG state.
    // Giving each light a disjoint slice removes that correlation.
    seedBase: u32,
    directionalLightDirection: vec2<f32>,
    // (pinch^2, atan(pinch^2)) - spot kind only, computed CPU-side exactly like Unity's g_lightPinch.
    lightPinch: vec2<f32>,
    integrationInterval: f32,
    integrationIntervalSquared: f32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> g_rand: array<vec4<u32>>;
@group(1) @binding(1) var albedo: texture_2d<f32>;
@group(1) @binding(2) var density: texture_2d<f32>;
@group(1) @binding(3) var normalRoughness: texture_2d<f32>;
// Operation-internal samplers (never caller-configurable - see CLAUDE.md's ComputeOperation
// guidance): pointSampler mirrors Unity's sampler_point_clamp (used for albedo), linearSampler
// mirrors sampler_linear_clamp/samplerg_density/samplerg_normalAlignment/the LUT samplers (Unity
// uses plain bilinear for all of these - one shared linear sampler covers them all here).
@group(1) @binding(4) var pointSampler: sampler;
@group(1) @binding(5) var linearSampler: sampler;
@group(1) @binding(6) var teardropScatteringLut: texture_2d<f32>;
@group(1) @binding(7) var brdfLut: texture_3d<f32>;

// photons: 3 consecutive atomic<u32> entries per pixel (R,G,B) - same layout
// convert_photon_irradiance_to_hdr.wgsl reads. writeCounter: a 2-element manual uint64 (index 0 =
// low 32 bits, wrapping; index 1 = overflow/carry count) - a lifetime (never per-frame-cleared)
// photon-writes counter, ported from Unity's g_write_counter/GetCurrentWriteCountAsync for a
// portfolio-page "MWrites/s" display - see SimulationResources.getWriteCount.
@group(2) @binding(0) var<storage, read_write> photons: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> writeCounter: array<atomic<u32>>;

struct Ray {
    origin: vec2<f32>,
    direction: vec2<f32>,
    energy: vec3<f32>,
}

struct IntegrationContext {
    photon: Ray,
    uHitCurrent: f32,
    uHitNext: f32,
    uEscape: f32,
    testUV: vec2<f32>,
    // .x = this step's transmissibility, .y = the "minimum transmissibility" Unity's quadtree-lod
    // path would have varied - both read from density's R/G channels, which
    // raytraced_gbuffer.wgsl writes as duplicates of the same scalar (see its GBufferOutput doc
    // comment), so .x and .y are always equal here. Kept as a vec2 (rather than collapsing to one
    // scalar) to mirror Test()'s two read sites faithfully.
    transmissibilityNext: vec2<f32>,
}

// Was `ForwardMonteCarlo : BaseContext, IMonteCarloMethod` in Unity - WGSL has no struct
// inheritance/interfaces, so this is flattened to a plain struct plus free functions taking a
// ptr<function, Integrator>, the same style Random.wgsl already uses for Random's methods.
// hitIntensity (Unity) is dropped: set in Init, never read anywhere afterward - dead state.
struct Integrator {
    rand: Random,
    searchingPhase: bool,
    transmitPotential: f32,
    quantumScale: f32,
    currentSample: f32,
    uSampleTarget: f32,
    transmissibility: f32,
    uSampleRandomOffset: f32,
    testedTransmissibility: f32,
    testedU: f32,
}

fn integratorInit(seed: vec4<u32>) -> Integrator {
    var integrator: Integrator;
    integrator.rand = randomInit(seed);
    integrator.searchingPhase = true;
    integrator.transmissibility = 1.0;
    integrator.transmitPotential = 1.0;
    integrator.quantumScale = 1.0;
    integrator.currentSample = 0.0;
    integrator.uSampleTarget = 0.0;
    integrator.uSampleRandomOffset = randomNext(&integrator.rand);
    return integrator;
}

fn integratorBeginTraversal(integrator: ptr<function, Integrator>) {
    (*integrator).currentSample = 0.0;
    (*integrator).uSampleTarget = randomNext(&(*integrator).rand) * uniforms.integrationInterval;
    (*integrator).transmissibility = 1.0;
}

fn integratorTest(integrator: ptr<function, Integrator>, ctx: ptr<function, IntegrationContext>) -> bool {
    (*integrator).testedU = (*ctx).uHitNext;
    (*integrator).testedTransmissibility = (*ctx).transmissibilityNext.x;

    if ((*integrator).searchingPhase) {
        return (*integrator).testedU > (*ctx).uEscape;
    } else {
        let minimumTransmissibility = (*ctx).transmissibilityNext.y;
        return (*ctx).uHitNext > (*ctx).uEscape
            || minimumTransmissibility * (*integrator).transmissibility < (*integrator).transmitPotential;
    }
}

fn integratorPropagate(integrator: ptr<function, Integrator>, ctx: ptr<function, IntegrationContext>) -> bool {
    (*integrator).transmissibility *= (*integrator).testedTransmissibility;

    while ((*integrator).searchingPhase && (*integrator).testedU > (*integrator).uSampleTarget) {
        (*integrator).currentSample += 1.0;
        (*integrator).uSampleRandomOffset = randomNext(&(*integrator).rand);

        // Note preserved from Unity: ctx.testUV is a slightly different location than
        // uSampleTarget points to - doesn't seem to impact quality, simpler math this way.
        writeSample(integrator, (*ctx).photon.energy, (*ctx).photon.origin + (*ctx).photon.direction * (*integrator).uSampleTarget);
        (*integrator).uSampleTarget = ((*integrator).currentSample + (*integrator).uSampleRandomOffset) * uniforms.integrationInterval;
    }

    return true;
}

fn integratorEndTraversal(integrator: ptr<function, Integrator>, ctx: ptr<function, IntegrationContext>) -> bool {
    if (!(*integrator).searchingPhase) {
        (*ctx).uHitCurrent = (*integrator).testedU
            + log2((*integrator).transmitPotential / (*integrator).transmissibility)
            / (log2((*integrator).testedTransmissibility) - 1e-5);
    } else {
        (*ctx).uHitCurrent = 0.0;
    }
    return true;
}

fn integratorBounce(integrator: ptr<function, Integrator>, ctx: ptr<function, IntegrationContext>, albedoSample: vec3<f32>) -> bool {
    if (!(*integrator).searchingPhase) {
        let importantDirection = scatterMaterially(&(*integrator).rand, &(*ctx).photon.origin, (*ctx).testUV, (*ctx).photon.direction);

        (*ctx).photon.energy *= albedoSample * (*integrator).quantumScale * importantDirection.z;
        (*ctx).photon.direction = importantDirection.xy;
        (*ctx).photon.origin += (*ctx).photon.direction;

        // Every scatterMaterially return path currently yields w=0 (see its "BUG:"-flagged
        // transmit branch, ported verbatim below), so in practice this is always true and the
        // else branch is unreachable today - kept structurally faithful to Unity in case that
        // upstream bug workaround (1-1) ever changes.
        if (importantDirection.w < 0.5) {
            (*integrator).searchingPhase = true;
        } else {
            (*integrator).transmissibility /= (*integrator).testedTransmissibility;
        }
    } else {
        let r = randomNext(&(*integrator).rand);
        let u = (*integrator).transmissibility + (r * r) * (1.0 - (*integrator).transmissibility);
        (*integrator).transmitPotential = u;
        (*integrator).quantumScale = (1.0 - (*integrator).transmissibility) * 2.0 * r;
        (*integrator).searchingPhase = false;
    }
    return (*integrator).searchingPhase;
}

fn cross2D(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

// Unity's ImportanceSamplingTarget defaults to (0.5,0.5) in UV and its setter is never actually
// invoked anywhere in this codebase's Unity source beyond that default - hardcoded here as the
// simulation target's exact center (in pixel space) rather than threading an always-constant
// uniform through every dispatch.
fn scatterImportanceLobed(rand: ptr<function, Random>, origin: vec2<f32>) -> vec3<f32> {
    let targetSize = vec2<f32>(textureDimensions(albedo));
    let importanceSamplingTarget = targetSize * 0.5;

    var importantDirection = importanceSamplingTarget - origin;
    let lsq = dot(importantDirection, importantDirection);
    importantDirection = importantDirection / -sqrt(lsq);
    let perp = vec2<f32>(-importantDirection.y, importantDirection.x);

    let sampleValue = sampleLut1D(teardropScatteringLut, linearSampler, randomNext(rand), TEARDROP_SCATTERING_LUT_TEXEL_COUNT).xyz;
    return vec3<f32>(importantDirection * sampleValue.x + perp * sampleValue.y, sampleValue.z);
}

fn hermiteWeights(u: f32) -> vec4<f32> {
    let uu = u * u;
    let uuu = uu * u;
    return vec4<f32>(
        2.0 * uuu - 3.0 * uu + 1.0,
        uuu - 2.0 * uu + u,
        -2.0 * uuu + 3.0 * uu,
        uuu - uu,
    );
}

// Ported from StandardBRDF (SimulationCommon.cginc). Unity's lut_window_g_bdrfLUT/
// lut_slice_window_g_bdrfLUT (an auto-generated texel-center-remap uniform) is replaced by this
// project's existing lutUv() helper - same math. The U axis then gets a custom two-tap Hermite
// reconstruction (not a plain hardware bilinear/trilinear sample) blending a sampled value and a
// sampled tangent - genuinely bespoke to this LUT's encoding, so it can't be replaced by
// sampleLut3D wholesale. Unity's CubicWeights/`weights` local is dropped: it's computed but its
// result is never read anywhere in StandardBRDF (only hermiteWeights' result is used) - dead code.
fn standardBrdf(rand: ptr<function, Random>, normal: vec2<f32>, reflected: vec2<f32>, roughness: f32) -> vec3<f32> {
    let uvw = vec3<f32>(randomNext(rand), (cross2D(normal, reflected) + 1.0) / 2.0, roughness);
    let tangent = vec2<f32>(-normal.y, normal.x);

    let brdfWidth = BRDF_LUT_TEXEL_COUNT_X;
    let rescaledUvw = vec3<f32>(
        lutUv(uvw.x, BRDF_LUT_TEXEL_COUNT_X),
        lutUv(uvw.y, BRDF_LUT_TEXEL_COUNT_Y),
        lutUv(uvw.z, BRDF_LUT_TEXEL_COUNT_Z),
    );

    let uInPixelSpace = rescaledUvw.x * brdfWidth - 0.5;
    let bilinearParam = fract(uInPixelSpace);
    let uP1 = uInPixelSpace - bilinearParam;
    let uP2 = uP1 + 1.0;

    var uvw1 = rescaledUvw;
    uvw1.x = (uP1 + 0.5) / brdfWidth;
    var uvw2 = rescaledUvw;
    uvw2.x = (uP2 + 0.5) / brdfWidth;

    let scattered1 = textureSampleLevel(brdfLut, linearSampler, uvw1, 0.0);
    let scattered2 = textureSampleLevel(brdfLut, linearSampler, uvw2, 0.0);

    let tangent1 = vec4<f32>(-scattered1.y, scattered1.x, 0.0, 0.0) * scattered1.z;
    let tangent2 = vec4<f32>(-scattered2.y, scattered2.x, 0.0, 0.0) * scattered2.z;

    let hw = hermiteWeights(bilinearParam);
    let scattered = scattered1 * hw.x + tangent1 * hw.y + scattered2 * hw.z + tangent2 * hw.w;

    return vec3<f32>(normalize(scattered.x * normal + scattered.y * tangent), scattered.w * scattered.w);
}

// Ported from ScatterMaterially (SimulationCommon.cginc, the single-argument-uv overload - the
// other Unity overload that derives origin_uv from origin isn't needed, every call site here
// already has both).
fn scatterMaterially(rand: ptr<function, Random>, origin: ptr<function, vec2<f32>>, originUv: vec2<f32>, incoming: vec2<f32>) -> vec4<f32> {
    let eps = 1e-5;
    let normalAlignmentSample = textureSampleLevel(normalRoughness, linearSampler, originUv, 0.0);
    let normal = normalAlignmentSample.xyz;
    let alignment0 = normalAlignmentSample.w;

    if (dot(normal.xy, normal.xy) < eps) {
        // No normal information: scatter uniformly (importance-lobed).
        return vec4<f32>(scatterImportanceLobed(rand, *origin), 0.0);
    } else if (dot(normal.xy, incoming) > 0.0) {
        // Normal points the same general direction as incoming - transmit.
        // Source comment preserved verbatim (Unity SimulationCommon.cginc):
        // "BUG: Setting transmit to 1 causes an infinite loop when photons are emitted within the
        // normal field" - transmit is deliberately hardcoded to 1-1=0 to avoid that, not a typo.
        return vec4<f32>(incoming, 1.0, 1.0 - 1.0);
    } else {
        let len = length(normal.xy);
        let normal2D = normal.xy / len;
        let reflected = reflect(incoming, normal2D);

        let alignment = clamp(alignment0 / len, 0.0, 1.0);
        *origin -= incoming * 2.5;
        if (alignment > 0.999) {
            return vec4<f32>(reflected, 1.0, 0.0);
        } else if (alignment == 0.0) {
            let dir = randomNextDirection(rand);
            return vec4<f32>(select(-dir, dir, dot(dir, normal2D) > 0.0), 1.0, 0.0);
        } else {
            let scattered = standardBrdf(rand, normal2D, reflected, 1.0 - alignment);
            return vec4<f32>(scattered, 0.0);
        }
    }
}

// Bounds-guarded unlike Unity's RWTexture2D<uint> original, whose out-of-range writes are
// silently discarded by D3D itself with no equivalent guarantee for a WGSL storage buffer array -
// this reproduces that discard-at-border behavior (load-bearing for the bilinear splat's
// neighbor taps at the image edge, not just a hypothetical case). The write-counter increment
// happens first regardless of whether the guarded write below actually lands, matching Unity's
// unconditional-before-the-atomic-adds ordering.
fn writePhotonIndexed(pixel: vec2<i32>, energy: vec3<f32>, suppressPhoton: bool) {
    if (!suppressPhoton) {
        atomicAdd(&sharedWriteCounter, 1u);
    }

    let size = vec2<i32>(textureDimensions(albedo));
    if (pixel.x < 0 || pixel.y < 0 || pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }

    let base = (u32(pixel.y) * u32(size.x) + u32(pixel.x)) * 3u;
    atomicAdd(&photons[base], u32(energy.r));
    atomicAdd(&photons[base + 1u], u32(energy.g));
    atomicAdd(&photons[base + 2u], u32(energy.b));
}

fn writePhotonBilinear(location: vec2<f32>, energy: vec3<f32>) {
    let pixel = location - vec2<f32>(0.5, 0.5);
    let pixelFloor = vec2<i32>(floor(pixel));
    let pixelFrac = pixel - vec2<f32>(pixelFloor);

    writePhotonIndexed(pixelFloor, energy * (1.0 - pixelFrac.x) * (1.0 - pixelFrac.y), false);
    writePhotonIndexed(pixelFloor + vec2<i32>(1, 0), energy * pixelFrac.x * (1.0 - pixelFrac.y), true);
    writePhotonIndexed(pixelFloor + vec2<i32>(0, 1), energy * (1.0 - pixelFrac.x) * pixelFrac.y, true);
    writePhotonIndexed(pixelFloor + vec2<i32>(1, 1), energy * pixelFrac.x * pixelFrac.y, true);
}

fn writePhotonNearest(location: vec2<f32>, energy: vec3<f32>) {
    writePhotonIndexed(vec2<i32>(round(location - vec2<f32>(0.5, 0.5))), energy, false);
}

// Unity's BILINEAR_PHOTON_DISTRIBUTION - a compile-time switch (see ForwardMonteCarloOperation.
// updateSwitches), not a runtime uniform: it changes which of writePhotonBilinear (4
// atomicAdd-triplets/sample, smooth splat) vs. writePhotonNearest (1, blocky) gets compiled in,
// since the atomic-write count itself is what's being tuned - a measured ~1.5x photons/s win on
// the Pixel 10 Pro's PowerVR GPU (scattered global atomics are its weak point - see CLAUDE.md/
// mobile-perf-tuning notes) at the cost of visible splat blockiness on larger screens. Defaults on
// (this #ifdef is satisfied whenever the caller's ShaderDefines includes it) to match Unity's
// original default and preserve current desktop quality; expected to flip off on mobile once a
// tile-friendly realtime denoise pass exists to compensate for the lost smoothing.
#ifdef BILINEAR_PHOTON_DISTRIBUTION
fn writeSample(integrator: ptr<function, Integrator>, energy: vec3<f32>, location: vec2<f32>) {
    let outScatterDensity = uniforms.integrationIntervalSquared * (*integrator).transmissibility;
    writePhotonBilinear(location, energy * outScatterDensity);
}
#else
fn writeSample(integrator: ptr<function, Integrator>, energy: vec3<f32>, location: vec2<f32>) {
    let outScatterDensity = uniforms.integrationIntervalSquared * (*integrator).transmissibility;
    writePhotonNearest(location, energy * outScatterDensity);
}
#endif

// Ported from Integrate() (SimulationCommon.cginc) - the ray-march loop shared by every light
// kind. lod is always 0 in the Unity source (the quadtree jump optimization is dead - see file
// header), so this always takes a fixed 1-pixel step at mip 0, never the adaptive-lod path.
fn integrate(photon: ptr<function, Ray>, bounces: u32, integrator: ptr<function, Integrator>) {
    var ctx: IntegrationContext;
    ctx.photon = *photon;

    let targetSize = vec2<f32>(textureDimensions(albedo));
    let pixelSize = vec2<f32>(1.0, 1.0) / targetSize;

    var bounce = 0u;
    while (bounce < bounces) {
        if (ctx.photon.direction.x == 0.0) { ctx.photon.direction.x = 1e-8; }
        if (ctx.photon.direction.y == 0.0) { ctx.photon.direction.y = 1e-8; }

        let uvOrigin = ctx.photon.origin / targetSize;
        let uvDirection = ctx.photon.direction / targetSize;
        let boundaryBox = (vec4<f32>(-pixelSize, vec2<f32>(1.0, 1.0) + pixelSize) - uvOrigin.xyxy) / uvDirection.xyxy;

        ctx.uEscape = min(max(boundaryBox.x, boundaryBox.z), max(boundaryBox.y, boundaryBox.w));
        ctx.uHitCurrent = 0.0;
        ctx.testUV = uvOrigin;

        integratorBeginTraversal(integrator);
        var continueRunning = true;
        // MAX_INTEGRATION_STEPS is a compile-time switch (ForwardMonteCarloOperation.
        // updateSwitches), not a runtime uniform - a step cap on this one search-or-refine phase
        // before it gives up (Unity's original hardcoded this at 2000; see
        // simulation.ts's computeMaxIntegrationSteps for the current one-domain-diagonal
        // derivation - this loop's uEscape-bounded overshoot condition can't legitimately need more
        // steps than that, since uEscape is a ray-vs-box exit distance). Lower caps cut per-thread
        // worst-case work directly, which matters most for divergent SIMT execution: every thread
        // in a workgroup pays for whichever thread takes the most steps - see CLAUDE.md/
        // mobile-perf-tuning notes.
        for (var steps = 0; steps < MAX_INTEGRATION_STEPS; steps++) {
            ctx.transmissibilityNext = vec2<f32>(1.0, 1.0) - textureSampleLevel(density, linearSampler, ctx.testUV, 0.0).xy / DENSITY_SCALE;
            ctx.uHitNext = ctx.uHitCurrent + 1.0;
            let overshoot = integratorTest(integrator, &ctx);

            if (!overshoot) {
                ctx.uHitCurrent = ctx.uHitNext;
                ctx.testUV = uvOrigin + uvDirection * ctx.uHitCurrent;
                if (!integratorPropagate(integrator, &ctx)) {
                    continueRunning = false;
                    break;
                }
            } else {
                continueRunning = integratorEndTraversal(integrator, &ctx);
                break;
            }
        }

        if (!continueRunning) { break; }

        ctx.photon.origin += ctx.photon.direction * ctx.uHitCurrent;
        let albedoSample = textureSampleLevel(albedo, pointSampler, ctx.testUV, 0.0).rgb;
        if (integratorBounce(integrator, &ctx, albedoSample)) {
            bounce++;
        }
    }

    *photon = ctx.photon;
}

#ifdef LIGHT_KIND_POINT
// Unity's RTPointLight.WorldTransform applies an extra 0.5 scale (its local sampling is a
// unit-radius disc, but the object's transform is calibrated to a diameter=scale convention like
// every other sprite/raytraced quad in this engine) - ported as scaling the sampled local
// position by 0.5 here instead, so the transform lookup stays uniform across all 5 light kinds.
fn emitLight(rand: ptr<function, Random>) -> Ray {
    let pos = randomNextCircle(rand) * 0.5;
    let origin = (uniforms.lightToTarget * vec4<f32>(pos, 0.0, 1.0)).xy;
    let importantDirection = scatterImportanceLobed(rand, origin);

    var emitted: Ray;
    emitted.origin = origin;
    emitted.direction = importantDirection.xy;
    emitted.energy = uniforms.lightEnergy * importantDirection.z;
    return emitted;
}
#endif

#ifdef LIGHT_KIND_SPOT
fn emitLight(rand: ptr<function, Random>) -> Ray {
    var pinched = 2.0 * randomNext(rand) - 1.0;
    pinched = tan(pinched * uniforms.lightPinch.y) / uniforms.lightPinch.x;

    var emitted: Ray;
    emitted.origin = (uniforms.lightToTarget * vec4<f32>(randomNext(rand) - 0.5, randomNext(rand) - 0.5, 0.0, 1.0)).xy;
    emitted.direction = normalize((uniforms.lightToTarget * vec4<f32>(pinched, -1.0, 0.0, 0.0)).xy);
    emitted.energy = uniforms.lightEnergy;
    return emitted;
}
#endif

#ifdef LIGHT_KIND_LASER
fn emitLight(rand: ptr<function, Random>) -> Ray {
    var emitted: Ray;
    emitted.origin = (uniforms.lightToTarget * vec4<f32>(randomNext(rand) - 0.5, randomNext(rand), 0.0, 1.0)).xy;
    emitted.direction = normalize((uniforms.lightToTarget * vec4<f32>(0.0, -1.0, 0.0, 0.0)).xy);
    emitted.energy = uniforms.lightEnergy;
    return emitted;
}
#endif

#ifdef LIGHT_KIND_DIRECTIONAL
fn emitLight(rand: ptr<function, Random>) -> Ray {
    let targetSize = vec2<f32>(textureDimensions(albedo));
    var perp = uniforms.directionalLightDirection.yx;
    perp.y *= -1.0;

    var emitted: Ray;
    emitted.direction = uniforms.directionalLightDirection;
    emitted.origin = (vec2<f32>(0.5, 0.5) - uniforms.directionalLightDirection + perp * (randomNext(rand) * 1.415 - 0.7075)) * targetSize;
    emitted.energy = uniforms.lightEnergy;
    return emitted;
}
#endif

#ifdef LIGHT_KIND_AMBIENT
fn emitLight(rand: ptr<function, Random>) -> Ray {
    let targetSize = vec2<f32>(textureDimensions(albedo));
    let nOrigin = randomNext2(rand);

    var emitted: Ray;
    emitted.origin = nOrigin * targetSize;
    emitted.direction = normalize(randomNextDirection(rand) - (nOrigin * 2.0 - vec2<f32>(1.0, 1.0)) / 1.44);
    emitted.energy = uniforms.lightEnergy;
    return emitted;
}
#endif

// Write-counter bookkeeping (see writeCounter above) is per-workgroup, not per-thread: a "last
// thread standing" pattern (sharedLiveThreads counts down from workgroup size) flushes the
// workgroup's accumulated sample count into the global counter exactly once, with the same manual
// uint64 carry Unity's g_write_counter[0]/[1] hack uses.
var<workgroup> sharedWriteCounter: atomic<u32>;
var<workgroup> sharedLiveThreads: atomic<i32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>, @builtin(local_invocation_index) localIndex: u32) {
    if (localIndex == 0u) {
        atomicStore(&sharedWriteCounter, 0u);
        atomicStore(&sharedLiveThreads, 64);
    }
    workgroupBarrier();

    let seedIndex = uniforms.seedBase + globalId.x;
    var integrator = integratorInit(g_rand[seedIndex]);
    var photon = emitLight(&integrator.rand);

    integrate(&photon, uniforms.bounces, &integrator);

    g_rand[seedIndex] = integrator.rand.state;

    let liveBeforeDecrement = atomicSub(&sharedLiveThreads, 1);
    if (liveBeforeDecrement == 1) {
        let writeCount = atomicLoad(&sharedWriteCounter);
        let previous = atomicAdd(&writeCounter[0], writeCount);
        if (writeCount > (0xFFFFFFFFu - previous)) {
            atomicAdd(&writeCounter[1], 1u);
        }
    }
}
