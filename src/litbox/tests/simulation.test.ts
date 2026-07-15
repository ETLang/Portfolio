import { describe, expect, it } from 'vitest';
import { getSimulationDeviceProfile, deriveEffectiveSimulation, computeMaxIntegrationSteps, DEFAULT_DENOISER_TUNABLES } from '../simulation.ts';
import type { SceneSimulation } from '../scene.ts';

const RAW_SIMULATION: SceneSimulation = {
    ownerId: 7,
    width: 512,
    height: 512,
    raysPerFrame: 100000,
    integrationInterval: 0.01,
    photonBounces: -1,
};

describe('getSimulationDeviceProfile', () => {
    it('is the identity profile on desktop, regardless of GPU friendliness', () => {
        expect(getSimulationDeviceProfile('desktop', true)).toEqual({
            resolutionScale: 1,
            raysPerFrameScale: 1,
            bilinearPhotonDistribution: true,
            maxBlurMip: DEFAULT_DENOISER_TUNABLES.maxBlurMip,
        });
        expect(getSimulationDeviceProfile('desktop', false)).toEqual({
            resolutionScale: 1,
            raysPerFrameScale: 1,
            bilinearPhotonDistribution: true,
            maxBlurMip: DEFAULT_DENOISER_TUNABLES.maxBlurMip,
        });
    });

    it('halves resolution and quarters rays, and disables bilinear, on mobile with an unfriendly GPU (the default assumption)', () => {
        expect(getSimulationDeviceProfile('android', false)).toEqual({
            resolutionScale: 0.5,
            raysPerFrameScale: 0.25,
            bilinearPhotonDistribution: false,
            maxBlurMip: DEFAULT_DENOISER_TUNABLES.maxBlurMip - 2,
        });
        expect(getSimulationDeviceProfile('ios', false)).toEqual({
            resolutionScale: 0.5,
            raysPerFrameScale: 0.25,
            bilinearPhotonDistribution: false,
            maxBlurMip: DEFAULT_DENOISER_TUNABLES.maxBlurMip - 2,
        });
    });

    it('halves resolution and rays (not quartered), and keeps bilinear on, on mobile with a friendly GPU', () => {
        expect(getSimulationDeviceProfile('ios', true)).toEqual({
            resolutionScale: 0.5,
            raysPerFrameScale: 0.5,
            bilinearPhotonDistribution: true,
            maxBlurMip: DEFAULT_DENOISER_TUNABLES.maxBlurMip - 2,
        });
    });

    it('reduces maxBlurMip on mobile by exactly 2 levels regardless of GPU friendliness (a mip-count argument, not a scattered-access one)', () => {
        expect(getSimulationDeviceProfile('android', true).maxBlurMip).toBe(DEFAULT_DENOISER_TUNABLES.maxBlurMip - 2);
    });
});

describe('deriveEffectiveSimulation', () => {
    it('leaves everything unchanged under the identity (desktop) profile', () => {
        const effective = deriveEffectiveSimulation(RAW_SIMULATION, { resolutionScale: 1, raysPerFrameScale: 1, bilinearPhotonDistribution: true, maxBlurMip: DEFAULT_DENOISER_TUNABLES.maxBlurMip });
        expect(effective.width).toBe(512);
        expect(effective.height).toBe(512);
        expect(effective.raysPerFrame).toBe(100000);
        expect(effective.ownerId).toBe(7);
    });

    it('scales width/height/raysPerFrame by the profile, rounding and flooring at 1', () => {
        const effective = deriveEffectiveSimulation(RAW_SIMULATION, { resolutionScale: 0.5, raysPerFrameScale: 0.25, bilinearPhotonDistribution: false, maxBlurMip: DEFAULT_DENOISER_TUNABLES.maxBlurMip - 2 });
        expect(effective.width).toBe(256);
        expect(effective.height).toBe(256);
        expect(effective.raysPerFrame).toBe(25000);
    });

    it('never scales a dimension or ray count down to 0', () => {
        const tiny: SceneSimulation = { ...RAW_SIMULATION, width: 1, height: 1, raysPerFrame: 1 };
        const effective = deriveEffectiveSimulation(tiny, { resolutionScale: 0.5, raysPerFrameScale: 0.25, bilinearPhotonDistribution: false, maxBlurMip: DEFAULT_DENOISER_TUNABLES.maxBlurMip - 2 });
        expect(effective.width).toBeGreaterThanOrEqual(1);
        expect(effective.height).toBeGreaterThanOrEqual(1);
        expect(effective.raysPerFrame).toBeGreaterThanOrEqual(1);
    });

});

describe('computeMaxIntegrationSteps', () => {
    it('is the domain diagonal (search and refine phases each get their own independent step budget, not a combined one)', () => {
        expect(computeMaxIntegrationSteps(512, 512)).toBeCloseTo(Math.hypot(512, 512));
    });

    it('scales down with resolution, so a halved simulation gets roughly half the step budget', () => {
        const full = computeMaxIntegrationSteps(512, 512);
        const halved = computeMaxIntegrationSteps(256, 256);
        expect(halved).toBeCloseTo(full / 2);
    });
});
