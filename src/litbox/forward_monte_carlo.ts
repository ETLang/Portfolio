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

// Must match the Uniforms struct layout in forward_monte_carlo.wgsl exactly (WGSL's default
// uniform-address-space struct layout rules: mat4x4 at 0 (64 bytes), vec3 at 64 (padded to 16),
// bounces/seedBase (u32) packed into that vec3's trailing padding at 76/80, directionalLightDirection
// (vec2, 8-byte aligned) at 88, lightPinch at 96, then two f32 at 104/108 - total 112 bytes).
const UNIFORMS_SIZE_BYTES = 112;

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

export interface ForwardMonteCarloUniforms {
    /** world -> simulation-target-pixel-space transform, already combined with this light's own world transform - see SimulationResources. */
    lightToTarget: mat4;
    lightEnergy: readonly [number, number, number];
    bounces: number;
    /** This light's offset into the shared random-seed buffer - see ComputedDataManager.acquireRandomSeedBuffer. */
    seedBase: number;
    directionalLightDirection: readonly [number, number];
    /** (pinch^2, atan(pinch^2)) - spot kind only. */
    lightPinch: readonly [number, number];
    integrationInterval: number;
    integrationIntervalSquared: number;
    /** This light's ray budget for this frame - also the dispatch extent, see updateUniforms. */
    rays: number;
}

/**
 * Port of Unity's ForwardMonteCarlo.compute's Simulate_<LightKind> kernels + SimulationCommon.cginc's
 * shared ray-march integrator - see forward_monte_carlo.wgsl for the actual math. One instance per
 * light kind (the kind is a compile-time #define baked in at construction, never changed
 * afterward - see baseDefines below), dispatched once per light *instance* by SimulationResources.
 * Unlike lightKind, BILINEAR_PHOTON_DISTRIBUTION *can* change at runtime, via updateSwitches - see
 * its doc comment and forward_monte_carlo.wgsl's writeSample.
 *
 * Samplers are created here, not caller-configurable, per this project's ComputeOperation
 * convention (CLAUDE.md) - pointSampler/linearSampler mirror Unity's sampler_point_clamp/
 * sampler_linear_clamp (used for albedo, and for density/normalRoughness/both LUTs respectively -
 * Unity uses plain bilinear for all of those, so one shared linear sampler covers them all).
 */
export class ForwardMonteCarloOperation extends ComputeOperation {
    private uniformBuffer: GPUBuffer;
    private pointSampler: GPUSampler;
    private linearSampler: GPUSampler;

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

        this.uniformBuffer = device.createBuffer({
            size: UNIFORMS_SIZE_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.setUniforms([{ binding: 0, resource: { buffer: this.uniformBuffer } }]);

        this.pointSampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest', mipmapFilter: 'nearest' });
        this.linearSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'nearest' });
    }

    /** See CLAUDE.md's updateSwitches/pipelineDirty pattern - a no-op (via setShaderCode) if `switches` matches what's already compiled. */
    public updateSwitches(switches: ForwardMonteCarloSwitches): void {
        // steps (forward_monte_carlo.wgsl's integrate()) is i32 - Math.ceil rather than truncating,
        // so the compiled bound never falls short of the caller's computed step budget.
        const maxIntegrationSteps = Math.ceil(switches.maxIntegrationSteps);
        const defines: ShaderDefines = { ...this.baseDefines, MAX_INTEGRATION_STEPS: `${maxIntegrationSteps}` };
        if (switches.bilinearPhotonDistribution) {
            defines.BILINEAR_PHOTON_DISTRIBUTION = true;
        }
        this.setShaderCode(preprocessShader(shaderCode, defines));
    }

    public updateUniforms(uniforms: ForwardMonteCarloUniforms): void {
        const data = new ArrayBuffer(UNIFORMS_SIZE_BYTES);
        new Float32Array(data, 0, 16).set(uniforms.lightToTarget as Float32Array);
        const view = new DataView(data);
        view.setFloat32(64, uniforms.lightEnergy[0], true);
        view.setFloat32(68, uniforms.lightEnergy[1], true);
        view.setFloat32(72, uniforms.lightEnergy[2], true);
        view.setUint32(76, uniforms.bounces, true);
        view.setUint32(80, uniforms.seedBase, true);
        view.setFloat32(88, uniforms.directionalLightDirection[0], true);
        view.setFloat32(92, uniforms.directionalLightDirection[1], true);
        view.setFloat32(96, uniforms.lightPinch[0], true);
        view.setFloat32(100, uniforms.lightPinch[1], true);
        view.setFloat32(104, uniforms.integrationInterval, true);
        view.setFloat32(108, uniforms.integrationIntervalSquared, true);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, data);

        // Deliberate deviation from the usual ComputeOperation convention (dispatch extent
        // normally derives from updateOutputs' resource size): here the dispatch extent is this
        // light's ray budget, a per-dispatch value unrelated to the output buffer's fixed size -
        // see the class doc comment.
        this.setDispatchExtent(uniforms.rays, 1, 1);
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

// --- Pure per-light CPU-side math (SimulationResources.run's orchestration) - kept as standalone
// functions, not private methods, so they're unit-testable without a GPU device. Mirrors Unity's
// ForwardMonteCarlo.cs Integrate()/SimulateLight() - see forward_monte_carlo.test.ts and this
// project's plan for the derivation of each formula.

const LUMINANCE_WEIGHTS = [0.2126, 0.7152, 0.0722] as const;

/** Rec.709 luminance of an (already-linear) energy color - ForwardMonteCarlo.cs's Luminance(). */
export function luminance(energyRgb: readonly [number, number, number]): number {
    return energyRgb[0] * LUMINANCE_WEIGHTS[0] + energyRgb[1] * LUMINANCE_WEIGHTS[1] + energyRgb[2] * LUMINANCE_WEIGHTS[2];
}

/**
 * This light's luminance-weighted share of `raysPerFrame`, rounded up to a multiple of 64 (so ray
 * counts divide evenly into @workgroup_size(64,1,1) workgroups) - matches SimulateLight's C#
 * truncating-division rounding exactly, including its minimum of 64 rays even for a light with
 * zero luminance share (a real, faithfully-ported quirk of that rounding trick, not a bug here).
 */
export function computeRayCount(luma: number, totalLuma: number, raysPerFrame: number): number {
    const raysRaw = Math.trunc((luma / totalLuma) * raysPerFrame);
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

/** Directional-kind-only: transforms local "down" (0,-1,0) as a direction (w=0, translation-free) through `lightToTarget` and normalizes. */
export function computeDirectionalLightDirection(lightToTarget: mat4): [number, number] {
    const transformed = vec4.create();
    vec4.transformMat4(transformed, [0, -1, 0, 0], lightToTarget);
    const length = Math.hypot(transformed[0], transformed[1]);
    return length > 0 ? [transformed[0] / length, transformed[1] / length] : [0, 0];
}

/** Combines the write counter's 2 u32 readback values (see SimulationResources.getWriteCount) into the manual-uint64 they represent - `lo` is the wrapping low 32 bits, `hi` is the overflow/carry count. */
export function combineWriteCount(lo: number, hi: number): bigint {
    return (BigInt(hi) << 32n) | BigInt(lo);
}
