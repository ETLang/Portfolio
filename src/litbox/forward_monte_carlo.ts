import { mat4, vec4 } from 'gl-matrix';
import type { LightKind } from './scene.ts';
import type { LutResources } from './lut_resources.ts';
import { ComputeOperation } from './compute_operation.ts';
import shaderCode from './shaders/forward_monte_carlo.wgsl?raw';
import { preprocessShader, type ShaderDefines } from './shaders/shader_preprocessor.ts';

// erasableSyntaxOnly forbids `enum` - see light_resources.ts's LIGHT_KIND for the same pattern.
const LIGHT_KIND_DEFINE: Record<LightKind, string> = {
    point: 'LIGHT_KIND_POINT',
    spot: 'LIGHT_KIND_SPOT',
    laser: 'LIGHT_KIND_LASER',
    directional: 'LIGHT_KIND_DIRECTIONAL',
    ambient: 'LIGHT_KIND_AMBIENT',
};

// Must match FrameUniforms in forward_monte_carlo.wgsl exactly: 4 consecutive f32 (each naturally
// 4-byte aligned, no padding needed) - total 16 bytes.
const FRAME_UNIFORMS_SIZE_BYTES = 16;

// Must match the LightJob struct layout in forward_monte_carlo.wgsl exactly (WGSL's default
// storage-address-space struct layout rules: mat4x4 at 0 (64 bytes), vec3 at 64 (padded to 16),
// bounces/seedBase/halfIndex/rayOffset (u32) packed into that vec3's trailing padding plus the
// following 4 bytes at 76/80/84/88, directionalLightDirection (vec2, 8-byte aligned) at 96,
// lightPinch at 104, directionalLightSegmentStart/Vector (vec2 each, 8-byte aligned) at 112/120 -
// total 128 bytes, already a multiple of the struct's overall 16-byte alignment (from the leading
// mat4x4) so no trailing padding. `array<LightJob>`'s stride is this same 128 with no attributes
// needed, since WGSL's default array stride is already roundUp(align, size) = roundUp(16, 128).
const LIGHT_JOB_STRIDE_BYTES = 128;

export interface ForwardMonteCarloSwitches {
    /**
     * Unity's BILINEAR_PHOTON_DISTRIBUTION: smooth 4-tap bilinear photon splat (true, the default)
     * vs. a single-tap nearest write (false) - see forward_monte_carlo.wgsl's writeSample for the
     * actual tradeoff (visual smoothness vs. 4x fewer scattered global atomicAdds/sample, which
     * measurably matters on mobile GPUs weak at scattered atomics - see CLAUDE.md).
     */
    bilinearPhotonDistribution: boolean;
    /**
     * Per-bounce ray-march step cap - see forward_monte_carlo.wgsl's integrate() loop. Unity
     * hardcoded this at 2000; lowering it bounds each thread's worst-case work per bounce, which
     * matters most for divergent SIMT execution (every thread in a workgroup pays for whichever
     * thread takes the most steps) - see CLAUDE.md/mobile-perf-tuning notes.
     */
    maxIntegrationSteps: number;
}

/** Historical Unity-ported value, used only for the constructor's placeholder pre-updateSwitches shader compile - see updateSwitches, always called before the first real dispatch. */
const DEFAULT_MAX_INTEGRATION_STEPS = 2000;

/** Frame-constant across every light this frame - see forward_monte_carlo.wgsl's FrameUniforms doc comment for why these live in a separate binding from ForwardMonteCarloJob. */
export interface ForwardMonteCarloFrameUniforms {
    integrationInterval: number;
    integrationIntervalSquared: number;
    /** SimulationTunables.surfaceBias - self-intersection pushback after a bounce (forward_monte_carlo.wgsl's scatterMaterially). */
    surfaceBias: number;
    /** SimulationTunables.ambientScatterSoftness - ambient emitLight's direction-perturbation divisor. */
    ambientScatterSoftness: number;
}

/** One (light instance, A/B half) pair - see forward_monte_carlo.wgsl's LightJob doc comment. Many of these are batched into a single updateJobs() call/dispatch per light kind. */
export interface ForwardMonteCarloJob {
    /** world -> simulation-target-pixel-space transform, already combined with this light's own world transform - see SimulationResources. */
    lightToTarget: mat4;
    lightEnergy: readonly [number, number, number];
    bounces: number;
    /** This job's offset into the shared random-seed buffer - see ComputedDataManager.acquireRandomSeedBuffer. */
    seedBase: number;
    /** 0 or 1 - which half of the two-way variance-estimation split this job writes into - see forward_monte_carlo.wgsl's LightJob.halfIndex. */
    halfIndex: number;
    directionalLightDirection: readonly [number, number];
    /** Directional kind only - one endpoint of the sampling segment, see computeDirectionalLightSegment. */
    directionalLightSegmentStart: readonly [number, number];
    /** Directional kind only - the segment's full extent (end - start), see computeDirectionalLightSegment. */
    directionalLightSegmentVector: readonly [number, number];
    /** (pinch^2, atan(pinch^2)) - spot kind only. */
    lightPinch: readonly [number, number];
    /** This job's ray budget for this frame (always a multiple of 64) - determines its workgroup count and its `rayOffset` within the batch, see updateJobs. */
    rays: number;
}

/**
 * Port of Unity's ForwardMonteCarlo.compute's Simulate_<LightKind> kernels + SimulationCommon.cginc's
 * shared ray-march integrator - see forward_monte_carlo.wgsl for the actual math. One instance per
 * light kind (the kind is a compile-time #define baked in at construction, never changed
 * afterward - see baseDefines below), dispatched once per *kind* per frame by SimulationResources -
 * every active light of that kind (already split into its own A/B half, same as before) is batched
 * into that single dispatch via updateJobs, rather than each instance getting its own dispatch. See
 * forward_monte_carlo.wgsl's file header for why (many tiny same-kind dispatches all writing the
 * same atomic photons/writeCounter buffers forced the GPU to serialize them one at a time).
 * Unlike lightKind, BILINEAR_PHOTON_DISTRIBUTION *can* change at runtime, via updateSwitches - see
 * its doc comment and forward_monte_carlo.wgsl's writeSample.
 *
 * Samplers are created here, not caller-configurable, per this project's ComputeOperation
 * convention (CLAUDE.md) - pointSampler/linearSampler mirror Unity's sampler_point_clamp/
 * sampler_linear_clamp (used for albedo, and for density/normalRoughness/both LUTs respectively -
 * Unity uses plain bilinear for all of those, so one shared linear sampler covers them all).
 */
export class ForwardMonteCarloOperation extends ComputeOperation {
    /**
     * Fixed-size, allocated once and reused every frame via writeBuffer - safe because (unlike the
     * old per-dispatch-pool design this replaced) this operation is dispatched at most once per
     * frame now, the same assumption every other ComputeOperation subclass already relies on.
     */
    private frameUniformsBuffer: GPUBuffer;
    /** Grow-only, mirrors ComputedDataManager.acquireRandomSeedBuffer's pattern - see ensureJobsCapacity. */
    private jobsBuffer: GPUBuffer | null = null;
    private jobsCapacityBytes = 0;
    /** Grow-only, same pattern as jobsBuffer - see ensureWorkgroupToJobCapacity. */
    private workgroupToJobBuffer: GPUBuffer | null = null;
    private workgroupToJobCapacityBytes = 0;
    private pointSampler: GPUSampler;
    private linearSampler: GPUSampler;
    /**
     * Last switches actually applied, so updateSwitches can skip re-running preprocessShader
     * (a full pass over this shader's source plus its #include chain) when called again with the
     * same values - the common case, since SimulationResources.tracePhotons calls it once per
     * light *kind* per frame with frame-constant values.
     */
    private lastSwitches: ForwardMonteCarloSwitches | null = null;

    /**
     * `lightKind`'s LIGHT_KIND_* define and the LUT texel-count defines are constant for this
     * instance's lifetime (never revisited by updateSwitches) - kept here so updateSwitches can
     * re-run preprocessShader with them plus whatever BILINEAR_PHOTON_DISTRIBUTION state was just
     * requested, without the caller having to re-supply lightKind/lutResources every switch change.
     */
    private readonly baseDefines: ShaderDefines;

    constructor(device: GPUDevice, lightKind: LightKind, lutResources: LutResources) {
        const [brdfWidth, brdfHeight, brdfDepth] = lutResources.getBrdfResolution();
        const baseDefines: ShaderDefines = {
            [LIGHT_KIND_DEFINE[lightKind]]: true,
            TEARDROP_SCATTERING_LUT_TEXEL_COUNT: `${lutResources.getTeardropScatteringSampleCount()}.0`,
            BRDF_LUT_TEXEL_COUNT_X: `${brdfWidth}.0`,
            BRDF_LUT_TEXEL_COUNT_Y: `${brdfHeight}.0`,
            BRDF_LUT_TEXEL_COUNT_Z: `${brdfDepth}.0`,
        };
        // Matches Unity's BILINEAR_PHOTON_DISTRIBUTION defaulting on; MAX_INTEGRATION_STEPS gets
        // its historical Unity value here purely so this placeholder compile is valid WGSL - see
        // updateSwitches, always called (with the caller's real values) before the first dispatch.
        super(device, preprocessShader(shaderCode, {
            ...baseDefines,
            BILINEAR_PHOTON_DISTRIBUTION: true,
            MAX_INTEGRATION_STEPS: `${DEFAULT_MAX_INTEGRATION_STEPS}`,
        }), 'main');
        this.baseDefines = baseDefines;

        this.pointSampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest', mipmapFilter: 'nearest' });
        this.linearSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'nearest' });

        this.frameUniformsBuffer = device.createBuffer({
            size: FRAME_UNIFORMS_SIZE_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    /** See CLAUDE.md's updateSwitches/pipelineDirty pattern - a no-op if `switches` matches what's already compiled. */
    public updateSwitches(switches: ForwardMonteCarloSwitches): void {
        if (
            this.lastSwitches
            && this.lastSwitches.bilinearPhotonDistribution === switches.bilinearPhotonDistribution
            && this.lastSwitches.maxIntegrationSteps === switches.maxIntegrationSteps
        ) {
            return;
        }
        this.lastSwitches = switches;

        // steps (forward_monte_carlo.wgsl's integrate()) is i32 - Math.ceil rather than truncating,
        // so the compiled bound never falls short of the caller's computed step budget.
        const maxIntegrationSteps = Math.ceil(switches.maxIntegrationSteps);
        const defines: ShaderDefines = { ...this.baseDefines, MAX_INTEGRATION_STEPS: `${maxIntegrationSteps}` };
        if (switches.bilinearPhotonDistribution) {
            defines.BILINEAR_PHOTON_DISTRIBUTION = true;
        }
        this.setShaderCode(preprocessShader(shaderCode, defines));
    }

    /** Writes the frame-constant scalars - see ForwardMonteCarloFrameUniforms's doc comment. Cheap enough (16 bytes) to call unconditionally once per kind per frame, no dirty-check needed. */
    public updateFrameUniforms(uniforms: ForwardMonteCarloFrameUniforms): void {
        const data = new ArrayBuffer(FRAME_UNIFORMS_SIZE_BYTES);
        const view = new DataView(data);
        view.setFloat32(0, uniforms.integrationInterval, true);
        view.setFloat32(4, uniforms.integrationIntervalSquared, true);
        view.setFloat32(8, uniforms.surfaceBias, true);
        view.setFloat32(12, uniforms.ambientScatterSoftness, true);
        this.device.queue.writeBuffer(this.frameUniformsBuffer, 0, data);
        // frameUniformsBuffer's object identity never changes (allocated once, fixed size, in the
        // constructor) - no setUniforms() call needed here. updateJobs (always called this same
        // frame, in either order relative to this) is what (re)establishes the group 0 bind group,
        // and it always references this same buffer object.
    }

    /**
     * Packs every active light of this kind (each already split into its own A/B half, one
     * ForwardMonteCarloJob per half - see SimulationResources.tracePhotons) into the jobs storage
     * buffer, builds the workgroupToJob mapping that lets forward_monte_carlo.wgsl's main() resolve
     * each workgroup's job in O(1), and sets the dispatch extent to the batch's total ray count.
     * Replaces the old per-dispatch updateUniforms/execute loop: this operation is now dispatched
     * at most once per frame, so jobsBuffer/workgroupToJobBuffer are ordinary grow-only buffers
     * (ComputedDataManager.acquireRandomSeedBuffer's pattern), not a per-dispatch pool - see the
     * class doc comment for why the old pool is no longer needed.
     */
    public updateJobs(jobs: readonly ForwardMonteCarloJob[]): void {
        const { rayOffsets, workgroupToJob: workgroupToJobData, totalRays } = layoutJobBatch(jobs.map((job) => job.rays));

        const jobsData = new ArrayBuffer(jobs.length * LIGHT_JOB_STRIDE_BYTES);
        const jobsView = new DataView(jobsData);
        for (let i = 0; i < jobs.length; i++) {
            writeJob(jobsView, i * LIGHT_JOB_STRIDE_BYTES, jobs[i], rayOffsets[i]);
        }

        this.ensureJobsCapacity(jobsData.byteLength);
        this.device.queue.writeBuffer(this.jobsBuffer!, 0, jobsData);
        this.ensureWorkgroupToJobCapacity(workgroupToJobData.byteLength);
        this.device.queue.writeBuffer(this.workgroupToJobBuffer!, 0, workgroupToJobData);

        // entriesEqual (ComputeOperation.setUniforms) compares buffer object identity, so this is a
        // no-op in the common steady-state case where the light count this frame didn't force
        // either buffer to grow (reusing the exact same 3 objects as last frame).
        this.setUniforms([
            { binding: 0, resource: { buffer: this.frameUniformsBuffer } },
            { binding: 1, resource: { buffer: this.jobsBuffer! } },
            { binding: 2, resource: { buffer: this.workgroupToJobBuffer! } },
        ]);
        this.setDispatchExtent(totalRays, 1, 1);
    }

    private ensureJobsCapacity(byteLength: number): void {
        if (this.jobsBuffer && this.jobsCapacityBytes >= byteLength) {
            return;
        }
        this.jobsBuffer?.destroy();
        this.jobsBuffer = this.device.createBuffer({
            size: byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.jobsCapacityBytes = byteLength;
    }

    private ensureWorkgroupToJobCapacity(byteLength: number): void {
        if (this.workgroupToJobBuffer && this.workgroupToJobCapacityBytes >= byteLength) {
            return;
        }
        this.workgroupToJobBuffer?.destroy();
        this.workgroupToJobBuffer = this.device.createBuffer({
            size: byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.workgroupToJobCapacityBytes = byteLength;
    }

    public updateInputs(
        seedBuffer: GPUBuffer,
        albedo: GPUTextureView,
        density: GPUTextureView,
        normalRoughness: GPUTextureView,
        lutResources: LutResources,
    ): void {
        this.setInputs([
            { binding: 0, resource: { buffer: seedBuffer } },
            { binding: 1, resource: albedo },
            { binding: 2, resource: density },
            { binding: 3, resource: normalRoughness },
            { binding: 4, resource: this.pointSampler },
            { binding: 5, resource: this.linearSampler },
            { binding: 6, resource: lutResources.getTeardropScatteringView() },
            { binding: 7, resource: lutResources.getBrdfView() },
        ]);
    }

    public updateOutputs(photonBuffer: GPUBuffer, writeCounterBuffer: GPUBuffer): void {
        this.setOutputs([
            { binding: 0, resource: { buffer: photonBuffer } },
            { binding: 1, resource: { buffer: writeCounterBuffer } },
        ]);
    }
}

/** Packs one ForwardMonteCarloJob into `view` at `byteOffset`, matching LIGHT_JOB_STRIDE_BYTES's layout exactly. `rayOffset` is this job's start position within the batch's flat dispatch (see LightJob.rayOffset), computed by the caller from the running sum of prior jobs' `rays` - not a field read off `job` itself. */
function writeJob(view: DataView, byteOffset: number, job: ForwardMonteCarloJob, rayOffset: number): void {
    new Float32Array(view.buffer, view.byteOffset + byteOffset, 16).set(job.lightToTarget as Float32Array);
    view.setFloat32(byteOffset + 64, job.lightEnergy[0], true);
    view.setFloat32(byteOffset + 68, job.lightEnergy[1], true);
    view.setFloat32(byteOffset + 72, job.lightEnergy[2], true);
    view.setUint32(byteOffset + 76, job.bounces, true);
    view.setUint32(byteOffset + 80, job.seedBase, true);
    view.setUint32(byteOffset + 84, job.halfIndex, true);
    view.setUint32(byteOffset + 88, rayOffset, true);
    view.setFloat32(byteOffset + 96, job.directionalLightDirection[0], true);
    view.setFloat32(byteOffset + 100, job.directionalLightDirection[1], true);
    view.setFloat32(byteOffset + 104, job.lightPinch[0], true);
    view.setFloat32(byteOffset + 108, job.lightPinch[1], true);
    view.setFloat32(byteOffset + 112, job.directionalLightSegmentStart[0], true);
    view.setFloat32(byteOffset + 116, job.directionalLightSegmentStart[1], true);
    view.setFloat32(byteOffset + 120, job.directionalLightSegmentVector[0], true);
    view.setFloat32(byteOffset + 124, job.directionalLightSegmentVector[1], true);
}

// --- Pure per-light CPU-side math (SimulationResources.run's orchestration) - kept as standalone
// functions, not private methods, so they're unit-testable without a GPU device. Mirrors Unity's
// ForwardMonteCarlo.cs Integrate()/SimulateLight() - see forward_monte_carlo.test.ts and this
// project's plan for the derivation of each formula.

export interface JobBatchLayout {
    /** Parallel to the input `jobRayCounts` - rayOffsets[i] is job i's start offset within the flat batched dispatch (see LightJob.rayOffset's doc comment). */
    rayOffsets: number[];
    /** One entry per workgroup in the flat batched dispatch, giving that workgroup's index into the jobs array - see forward_monte_carlo.wgsl's workgroupToJob binding. */
    workgroupToJob: Uint32Array;
    /** Sum of `jobRayCounts` - the batch's dispatch extent. */
    totalRays: number;
}

/**
 * Pure batching math behind ForwardMonteCarloOperation.updateJobs, split out so it's unit-testable
 * without a GPU device - see forward_monte_carlo.test.ts. `jobRayCounts` must each already be a
 * multiple of 64 (computeRayCount's contract); this function doesn't itself round.
 */
export function layoutJobBatch(jobRayCounts: readonly number[]): JobBatchLayout {
    let totalWorkgroups = 0;
    for (const rays of jobRayCounts) {
        totalWorkgroups += rays / 64;
    }
    const workgroupToJob = new Uint32Array(totalWorkgroups);

    const rayOffsets: number[] = [];
    let totalRays = 0;
    let workgroupCursor = 0;
    for (let i = 0; i < jobRayCounts.length; i++) {
        rayOffsets.push(totalRays);
        const workgroupCount = jobRayCounts[i] / 64;
        workgroupToJob.fill(i, workgroupCursor, workgroupCursor + workgroupCount);
        workgroupCursor += workgroupCount;
        totalRays += jobRayCounts[i];
    }

    return { rayOffsets, workgroupToJob, totalRays };
}

const LUMINANCE_WEIGHTS = [0.2126, 0.7152, 0.0722] as const;

/** Rec.709 luminance of an (already-linear) energy color - ForwardMonteCarlo.cs's Luminance(). */
export function luminance(energyRgb: readonly [number, number, number]): number {
    return energyRgb[0] * LUMINANCE_WEIGHTS[0] + energyRgb[1] * LUMINANCE_WEIGHTS[1] + energyRgb[2] * LUMINANCE_WEIGHTS[2];
}

/**
 * This light's share of `raysPerFrame` (by whatever per-light `weight`/`totalWeight` the caller
 * passes in), rounded up to a multiple of 64 (so ray counts divide evenly into
 * @workgroup_size(64,1,1) workgroups) - matches SimulateLight's C# truncating-division rounding
 * exactly, including its minimum of 64 rays even for a light with zero weight share (a real,
 * faithfully-ported quirk of that rounding trick, not a bug here).
 *
 * Deliberately not luminance-linear: `simulation.ts` weights by sqrt(luminance), not luminance
 * itself. Splitting a light's total emitted energy across N photons makes each photon's energy
 * ∝ 1/N, so that light's own relative variance (grain) works out to ∝ 1/N regardless of its
 * intensity - a pure luminance-linear N (N ∝ I) therefore leaves relative grain ∝ 1/I, exactly
 * the "dim lights look grainy" symptom. sqrt-weighting is the Neyman-style compromise between
 * that (minimizes total absolute variance across lights) and equal-N-per-light (minimizes/
 * equalizes each light's own relative variance, ignoring how much it matters to the final image).
 */
export function computeRayCount(weight: number, totalWeight: number, raysPerFrame: number): number {
    const raysRaw = Math.trunc((weight / totalWeight) * raysPerFrame);
    return (Math.trunc((raysRaw - 1) / 64) + 1) * 64;
}

/** `simulation.photonBounces === -1` is Unity's OverrideBounceCount == null sentinel - use this light's own bounce count. */
export function resolveBounces(photonBounces: number, lightBounces: number): number {
    return photonBounces === -1 ? lightBounces : photonBounces;
}

/**
 * World space -> simulation-target pixel space ([0,width] x [0,height]), replacing Unity's
 * WorldToTargetTransform. `simWorldTransform` maps the simulation's own local rect to world space
 * (sceneGraph.getWorldTransform(simulation.ownerId)); its inverse maps world space back into that
 * local [-0.5,0.5]^2 rect (see RaytracedResources.refreshViewProjection's identical convention),
 * then translate+scale expands that into pixel space.
 */
export function computeWorldToTargetPixels(simWorldTransform: mat4, width: number, height: number): mat4 {
    const simInverse = mat4.create();
    mat4.invert(simInverse, simWorldTransform);

    const yFlip = mat4.create();
    mat4.fromScaling(yFlip, [1, -1, 1]);
    const translate = mat4.create();
    mat4.fromTranslation(translate, [0.5, 0.5, 0]);
    const scale = mat4.create();
    mat4.fromScaling(scale, [width, height, 1]);

    const worldToTargetPixels = mat4.create();
    mat4.multiply(worldToTargetPixels, yFlip, simInverse);
    mat4.multiply(worldToTargetPixels, translate, worldToTargetPixels);
    mat4.multiply(worldToTargetPixels, scale, worldToTargetPixels);
    return worldToTargetPixels;
}

/** Combines the frame-constant pixelsFromWorld with one light's own world transform - Unity's `WorldToTargetTransform * light.WorldTransform`. */
export function computeLightToTarget(pixelsFromWorld: mat4, worldFromLight: mat4): mat4 {
    const lightToTarget = mat4.create();
    mat4.multiply(lightToTarget, pixelsFromWorld, worldFromLight);
    return lightToTarget;
}

/** ComputeDirectionalLightSegment's small safety margin, purely against rounding error: pushing
 *  the segment back by exactly the rect's half-diagonal already clears every corner in theory, but
 *  a grazing ray at that exact distance could clip one under floating-point error. */
const DIRECTIONAL_LIGHT_SEGMENT_MARGIN = 1.02;

export interface DirectionalLightSegment {
    /** Unit ray direction in target-pixel space. */
    direction: readonly [number, number];
    /** One endpoint of the perpendicular sampling segment, in target-pixel space. */
    segmentStart: readonly [number, number];
    /** The segment's full extent (segmentEnd - segmentStart). */
    segmentVector: readonly [number, number];
    /** |segmentVector| - used to scale this light's per-photon energy so total emitted power stays independent of the target's resolution. */
    length: number;
}

/**
 * Directional-kind-only: transforms local "down" (0,-1,0) as a direction (w=0, translation-free)
 * through `lightToTarget` and normalizes, then builds a line segment perpendicular to that
 * direction which is guaranteed to clear the [0,width]x[0,height] target rect from any direction -
 * a random point on it is a valid ray origin "just out of view" of the simulation area. The rect's
 * half-diagonal is the smallest distance that works for every direction (no point in the rect is
 * farther from its center than that), so it's used both as the segment's half-length and as how
 * far back to push it from the rect's center along the incoming direction.
 */
export function computeDirectionalLightSegment(lightToTarget: mat4, width: number, height: number): DirectionalLightSegment {
    const transformed = vec4.create();
    vec4.transformMat4(transformed, [0, -1, 0, 0], lightToTarget);
    const rawLength = Math.hypot(transformed[0], transformed[1]);
    const direction: [number, number] = rawLength > 0 ? [transformed[0] / rawLength, transformed[1] / rawLength] : [0, -1];

    const perp: [number, number] = [-direction[1], direction[0]];
    const halfDiagonal = 0.5 * Math.hypot(width, height) * DIRECTIONAL_LIGHT_SEGMENT_MARGIN;

    const centerX = width * 0.5 - direction[0] * halfDiagonal;
    const centerY = height * 0.5 - direction[1] * halfDiagonal;

    const segmentStart: [number, number] = [centerX - perp[0] * halfDiagonal, centerY - perp[1] * halfDiagonal];
    const segmentVector: [number, number] = [perp[0] * 2 * halfDiagonal, perp[1] * 2 * halfDiagonal];

    return { direction, segmentStart, segmentVector, length: 2 * halfDiagonal };
}

/** Combines the write counter's 2 u32 readback values (see SimulationResources.getWriteCount) into the manual-uint64 they represent - `lo` is the wrapping low 32 bits, `hi` is the overflow/carry count. */
export function combineWriteCount(lo: number, hi: number): bigint {
    return (BigInt(hi) << 32n) | BigInt(lo);
}
