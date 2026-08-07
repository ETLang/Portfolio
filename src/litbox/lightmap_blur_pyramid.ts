import { ComputedTexture } from './computed_data_manager.ts';
import { GaussianPyramidDownsampleOperation } from './gaussian_pyramid_downsample.ts';

/**
 * Regenerates the lightmap's mip 1+ chain as a real Gaussian blur pyramid, for sprite.wgsl's
 * blur-size selection (see this project's plan: "Sprite lightmap sampling: mip-LOD blur ->
 * discrete Gaussian blur-size selection"). Deliberately the only thing SimulationResources.run()
 * touches for this - it calls regenerate() once and never loops mip levels or references
 * GaussianPyramidDownsampleOperation itself, so a different internal strategy later (a wider
 * kernel, a separable two-pass blur, a shared-memory multi-level-per-dispatch reduction) is a
 * change contained entirely to this file.
 *
 * Not a ComputeOperation itself - it owns one internally and drives it once per mip level, mip 0
 * (the actual denoised lightmap) down. Each level blurs the *previous* level, not mip 0 directly,
 * so the blur compounds geometrically down the chain (the classic Burt-Adelson Gaussian pyramid
 * "reduce" operator) - see gaussian_pyramid_downsample.wgsl for why that's what makes a small
 * per-level kernel reach a very soft/blobby look after a few levels.
 *
 * Always uses Karis-weighted taps (see gaussian_pyramid_downsample.wgsl's #ifdef KARIS_WEIGHTED
 * doc comment), not just for the first reduce step: a single-texel HDR firefly does get fully
 * absorbed by one Karis-weighted pass, but a laser-beam-style feature (bright and only 1-2 texels
 * wide along its whole length, but long) stays that narrow across many octaves of downsampling,
 * so it can start re-aliasing at whichever level its width first drops below ~1 texel - not
 * predictable in advance, so every level needs the same protection.
 */
export class LightmapBlurPyramid {
    private downsample: GaussianPyramidDownsampleOperation;

    constructor(device: GPUDevice) {
        this.downsample = new GaussianPyramidDownsampleOperation(device);
        this.downsample.updateSwitches({ karisWeighted: true });
    }

    /**
     * Regenerates `lightmap`'s mips [mipRegenStart, lightmap.mipLevelCount) from whatever's
     * already written at mip (mipRegenStart - 1) - mirrors the loop shape
     * SimulationResources.run() used to run directly against MipDownsampleOperation. `width`/
     * `height` are mip 0's dimensions (the effective simulation resolution), matching every other
     * mip-loop in this codebase (e.g. SimulationResources.generateEvidenceMips).
     */
    public regenerate(encoder: GPUCommandEncoder, lightmap: ComputedTexture, width: number, height: number, mipRegenStart: number): void {
        for (let mip = mipRegenStart; mip < lightmap.mipLevelCount; mip++) {
            const mipWidth = Math.max(1, width >> mip);
            const mipHeight = Math.max(1, height >> mip);
            this.downsample.updateInputs(lightmap.getMipView(mip - 1));
            this.downsample.updateOutputs(lightmap.getMipView(mip), mipWidth, mipHeight);
            this.downsample.execute(encoder);
        }
    }
}
