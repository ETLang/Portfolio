import { ComputedTexture } from './computed_data_manager.ts';
import { GaussianBlurPassOperation } from './gaussian_blur_pass.ts';
import { GaussianPyramidDownsampleOperation } from './gaussian_pyramid_downsample.ts';

/**
 * HDR-safe replacement for LightmapBlurPyramid's role (regenerating the lightmap's higher blur
 * levels for sprite.wgsl's blur-size selection) - see this project's plan: "Fix lightmap
 * sprite-blur pixelation on high-contrast content". A new, separate implementation, not a
 * modification of LightmapBlurPyramid/GaussianPyramidDownsampleOperation - both of those still
 * work well and stay cheap for near-LDR scenes, and remain completely untouched; this class is
 * SimulationResources.run()'s new call site instead.
 *
 * `cascadeLevels` are full resolution, never decimated - each is a real, densely-sampled separable
 * Gaussian blur of the previous level (see gaussian_blur_pass.wgsl), so there's no resolution loss
 * for thin/bright/high-contrast content (e.g. laser beams) to alias against, unlike
 * GaussianPyramidDownsampleOperation's decimation pyramid. `cascadeLevels`' LAST entry is never
 * exposed to sprites as a selectable bucket - it's an internal "bridging" level, one more
 * calibrated doubling past the last exposed level, that exists purely to be decimated into
 * `mipChain`'s own mip0 (see "Calibration" below).
 *
 * Calibration: each level's target *effective* blur radius doubles the previous level's - the same
 * invariant GaussianPyramidDownsampleOperation's decimation pyramid gets for free purely from
 * halving resolution each step. Levels here never change resolution, so that doubling has to be
 * computed explicitly instead: level `i`'s target cumulative sigma is `baseSigma * 2^i`, and each
 * pass's own sigma (what's actually fed to GaussianBlurPassOperation) is derived from the
 * compounding relationship `sqrt(target_i^2 - target_{i-1}^2)`. Without the bridging level, the
 * first decimation step - which, like every other step in that pyramid, assumes its own small
 * fixed kernel only needs to contribute one halving-step's worth of extra blur - would receive an
 * already-large, externally-blurred input without the "free" doubling a halving pyramid normally
 * provides, silently under-blurring that level relative to its neighbors. Widening the decimation
 * kernel's own stride to compensate was considered and rejected: a 5-tap kernel sampled at a wide
 * stride is sparse, leaving gaps between taps that reintroduce the exact aliasing this design
 * exists to remove - so the fix lives entirely in this class's own pass-sigma calibration, and
 * GaussianPyramidDownsampleOperation is never modified or given a variable stride.
 */
export class LightmapBlurCascade {
    private blurPass: GaussianBlurPassOperation;
    private downsample: GaussianPyramidDownsampleOperation;

    constructor(device: GPUDevice) {
        this.blurPass = new GaussianBlurPassOperation(device);
        this.downsample = new GaussianPyramidDownsampleOperation(device);
        this.downsample.updateSwitches({ karisWeighted: true });
    }

    /**
     * Regenerates `cascadeLevels` from `sourceMip0View`, then decimates the last (bridging) level
     * into `mipChain`'s own mip chain - a real, halving-resolution mip chain from that point on,
     * via the existing, unmodified GaussianPyramidDownsampleOperation - see this class's doc
     * comment. `width`/`height` are the source/cascade levels' shared full resolution.
     */
    public regenerate(
        encoder: GPUCommandEncoder,
        sourceMip0View: GPUTextureView,
        cascadeLevels: ComputedTexture[],
        scratch: ComputedTexture,
        mipChain: ComputedTexture,
        width: number,
        height: number,
        baseSigma: number,
    ): void {
        // Resets GaussianBlurPassOperation's per-frame uniform-buffer-pool slot cursor - required
        // once per call, before this method's first updateUniforms() below - see
        // GaussianBlurPassOperation.beginFrame's doc comment: without it, every horizontal/
        // vertical pass in the loop below would silently read the LAST dispatch's direction
        // uniform once the frame's command buffer actually ran (confirmed: this produced
        // blur that only ever looked vertical).
        this.blurPass.beginFrame();

        let previousView = sourceMip0View;
        let cumulativeSigma = 0;

        for (let i = 0; i < cascadeLevels.length; i++) {
            const targetSigma = baseSigma * 2 ** i;
            const passSigma = Math.sqrt(Math.max(0, targetSigma * targetSigma - cumulativeSigma * cumulativeSigma));
            this.runSeparablePass(encoder, previousView, scratch, cascadeLevels[i], width, height, passSigma);
            previousView = cascadeLevels[i].view;
            cumulativeSigma = targetSigma;
        }

        // The bridging level (cascadeLevels' last entry) hands off into the existing decimation
        // pyramid exactly like that pyramid's own very first mip0->mip1 step: a normal, single 2x
        // reduction. mipChain's own mip0 must be freshly sized at width/2 x height/2, NOT reusing
        // any absolute mip-index sizing from elsewhere - see this class's calibration note above.
        const bridgingLevel = cascadeLevels[cascadeLevels.length - 1];
        this.downsample.updateInputs(bridgingLevel.view);
        this.downsample.updateOutputs(mipChain.getMipView(0), Math.max(1, width >> 1), Math.max(1, height >> 1));
        this.downsample.execute(encoder);

        for (let mip = 1; mip < mipChain.mipLevelCount; mip++) {
            const mipWidth = Math.max(1, width >> (mip + 1));
            const mipHeight = Math.max(1, height >> (mip + 1));
            this.downsample.updateInputs(mipChain.getMipView(mip - 1));
            this.downsample.updateOutputs(mipChain.getMipView(mip), mipWidth, mipHeight);
            this.downsample.execute(encoder);
        }
    }

    /**
     * One full 2D Gaussian blur pass at `sigma`: horizontal into `scratch`, vertical from
     * `scratch` into `dest` - see gaussian_blur_pass.wgsl. `scratch` is purely a transient
     * horizontal-pass intermediate, reused across every call; it's never anyone's final output.
     * Radius is truncated at 3 sigma, the standard cutoff beyond which a Gaussian's remaining tail
     * is negligible.
     */
    private runSeparablePass(
        encoder: GPUCommandEncoder,
        sourceView: GPUTextureView,
        scratch: ComputedTexture,
        dest: ComputedTexture,
        width: number,
        height: number,
        sigma: number,
    ): void {
        const radius = Math.max(1, Math.ceil(sigma * 3));

        this.blurPass.updateUniforms({ direction: [1, 0], radius, sigma });
        this.blurPass.updateInputs(sourceView);
        this.blurPass.updateOutputs(scratch.view, width, height);
        this.blurPass.execute(encoder);

        this.blurPass.updateUniforms({ direction: [0, 1], radius, sigma });
        this.blurPass.updateInputs(scratch.view);
        this.blurPass.updateOutputs(dest.view, width, height);
        this.blurPass.execute(encoder);
    }
}
