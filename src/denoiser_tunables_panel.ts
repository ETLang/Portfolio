import type { DenoiserTunables } from './litbox/simulation.ts';

/** One row's slider/textbox range - see DenoiserTunables' own doc comment for what each field does; these ranges are generous starting points around the current defaults, not derived from anything. */
interface DenoiserParamMeta {
    key: keyof DenoiserTunables;
    label: string;
    min: number;
    max: number;
    step: number;
}

/** Order matches which DenoiseOperation/BuildDenoiserQuadtreeOperation each field feeds - see DenoiserTunables. */
const DENOISER_PARAMS: DenoiserParamMeta[] = [
    { key: 'varianceScale', label: 'Variance Scale', min: 0, max: 20, step: 0.1 },
    { key: 'darknessNoiseFloor', label: 'Darkness Noise Floor', min: 0, max: 0.02, step: 0.0005 },
    { key: 'maxBlurMip', label: 'Max Blur Mip', min: 0, max: 8, step: 1 },
    { key: 'albedoSensitivity', label: 'Albedo Sensitivity', min: 0, max: 2, step: 0.01 },
    { key: 'densitySensitivity', label: 'Density Sensitivity', min: 0, max: 5, step: 0.05 },
    { key: 'normalSensitivity', label: 'Normal Sensitivity', min: 0, max: 32, step: 0.5 },
    { key: 'sigmaLuminanceTight', label: 'Sigma Luminance (Tight)', min: 0, max: 1, step: 0.01 },
    { key: 'sigmaLuminanceLoose', label: 'Sigma Luminance (Loose)', min: 0, max: 10, step: 0.1 },
    { key: 'kLuminance', label: 'K Luminance', min: 0.1, max: 10, step: 0.1 },
    { key: 'maxSplitDistance', label: 'Max Split Distance', min: 0, max: 8, step: 0.1 },
    { key: 'albedoLuminanceThreshold', label: 'Albedo Luminance Threshold', min: 0, max: 1, step: 0.01 },
    { key: 'albedoChromaThreshold', label: 'Albedo Chroma Threshold', min: 0, max: 1, step: 0.01 },
    { key: 'logDensityThreshold', label: 'Log Density Threshold', min: 0, max: 2, step: 0.01 },
    { key: 'volatilityThreshold', label: 'Volatility Threshold', min: 0, max: 1, step: 0.01 },
    { key: 'detailThreshold', label: 'Detail Threshold', min: 0, max: 2, step: 0.01 },
    { key: 'varianceGateScale', label: 'Variance Gate Scale', min: 0, max: 100, step: 1 },
];

/**
 * Expandable list of slider+textbox pairs, one per denoiser threshold (see this project's denoiser
 * plan) - `current` supplies the live values to render (main.ts re-generates this on every visit to
 * the litbox view, reading from SimulationResources.denoiserTunables, so it never shows stale
 * values after the user has already adjusted something). Both the slider and the textbox for a row
 * share the same `data-param` attribute (the DenoiserTunables key) - main.ts's delegated listener
 * uses that to know which field to write and to keep the paired control in sync, rather than one
 * id-based branch per parameter.
 */
export function getDenoiserTunablesPanel(current: DenoiserTunables): string {
    const rows = DENOISER_PARAMS.map(({ key, label, min, max, step }) => {
        const value = current[key];
        return `
            <div class="denoiser-param">
                <label for="denoiser-${key}-slider">${label}</label>
                <div class="denoiser-param-controls">
                    <input type="range" id="denoiser-${key}-slider" class="denoiser-param-slider" data-param="${key}"
                        min="${min}" max="${max}" step="${step}" value="${value}">
                    <input type="number" class="denoiser-param-number" data-param="${key}"
                        min="${min}" max="${max}" step="${step}" value="${value}">
                </div>
            </div>`;
    }).join('');

    return `
        <details class="denoiser-tunables">
            <summary>Denoiser Parameters</summary>
            <div class="denoiser-tunables-list">${rows}
            </div>
        </details>`;
}
