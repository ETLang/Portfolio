import type { SimulationTunables } from './litbox/simulation.ts';

/** One row's slider/textbox range for a scene-independent-magnitude SimulationTunables field - see SimulationTunables' own doc comment for what each field does; these ranges are generous starting points around the current defaults, not derived from anything. */
interface SimulationParamMeta {
    key: keyof Omit<SimulationTunables, 'raysPerFrame'>;
    label: string;
    min: number;
    max: number;
    step: number;
}

/** Order matches ForwardMonteCarloOperation's uniform fields each feeds - see SimulationTunables. raysPerFrame is deliberately excluded - see getSimulationTunablesPanel's own doc comment. */
const SIMULATION_PARAMS: SimulationParamMeta[] = [
    { key: 'integrationInterval', label: 'Integration Interval', min: 0.005, max: 0.3, step: 0.0005 },
    { key: 'photonBounces', label: 'Photon Bounces (Override)', min: -1, max: 8, step: 1 },
    { key: 'surfaceBias', label: 'Surface Bias', min: 0, max: 10, step: 0.1 },
    { key: 'ambientScatterSoftness', label: 'Ambient Scatter Softness', min: 0.1, max: 5, step: 0.05 },
];

/** Fixed range for the simulation's base (pre-device-scale) resolution rows - see LitboxSceneRenderer.resizeSimulation's own doc comment for why these are handled separately from the SimulationTunables rows above. */
const SIMULATION_SIZE_RANGE = { min: 64, max: 2048, step: 64 };

/** Renders one slider+textbox row, tagged with `dataAttr` (either 'data-sim-param' or 'data-sim-size' - see this file's own doc comment) so main.ts's listeners know which write-back path to route the row to. */
function renderParamRow(key: string, label: string, min: number, max: number, step: number, value: number, dataAttr: string): string {
    return `
        <div class="simulation-param">
            <label for="simulation-${key}-slider">${label}</label>
            <div class="simulation-param-controls">
                <input type="range" id="simulation-${key}-slider" class="simulation-param-slider" ${dataAttr}="${key}"
                    min="${min}" max="${max}" step="${step}" value="${value}">
                <input type="number" class="simulation-param-number" ${dataAttr}="${key}"
                    min="${min}" max="${max}" step="${step}" value="${value}">
            </div>
        </div>`;
}

/**
 * Expandable list of slider+textbox pairs for the raytracing simulation's own tunables (contrast
 * denoiser_tunables_panel.ts, which tunes the post-process blur instead) - `current` supplies the
 * live SimulationTunables values, `rawSize` the active scene's own (pre-device-scale) resolution
 * (see LitboxSceneRenderer.resizeSimulation's own doc comment for why resolution isn't part of
 * SimulationTunables itself). main.ts re-generates this on every visit to the litbox view and
 * after every scene switch/resize, so it never shows stale values.
 *
 * raysPerFrame and the two size rows are rendered separately from SIMULATION_PARAMS, not as
 * static meta entries:
 * - raysPerFrame's natural magnitude varies ~100x scene-to-scene (a fixed absolute range can't
 *   fit both a simple Cornell box and a many-light battle scene), so its slider range is derived
 *   from the current value instead.
 * - The size rows read/write a different underlying value (the scene's raw JSON, not
 *   SimulationTunables) via a distinct `data-sim-size` attribute, so main.ts's listener can route
 *   them to resizeSimulation() (an async rebuild, on 'change' only) instead of a direct field
 *   write (on every 'input').
 */
export function getSimulationTunablesPanel(current: SimulationTunables, rawSize: { width: number; height: number }): string {
    const raysPerFrameMax = Math.max(1024, current.raysPerFrame * 4);
    const rows = [
        renderParamRow('raysPerFrame', 'Rays Per Frame', 0, raysPerFrameMax, 64, current.raysPerFrame, 'data-sim-param'),
        ...SIMULATION_PARAMS.map(({ key, label, min, max, step }) => renderParamRow(key, label, min, max, step, current[key], 'data-sim-param')),
        renderParamRow('width', 'Simulation Width', SIMULATION_SIZE_RANGE.min, SIMULATION_SIZE_RANGE.max, SIMULATION_SIZE_RANGE.step, rawSize.width, 'data-sim-size'),
        renderParamRow('height', 'Simulation Height', SIMULATION_SIZE_RANGE.min, SIMULATION_SIZE_RANGE.max, SIMULATION_SIZE_RANGE.step, rawSize.height, 'data-sim-size'),
    ].join('');

    return `
        <details class="simulation-tunables">
            <summary>Simulation Parameters</summary>
            <div class="simulation-tunables-list">${rows}
            </div>
        </details>`;
}
