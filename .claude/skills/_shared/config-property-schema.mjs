// Single source of truth for set-config-property's whitelist - imported by BOTH
// set-config-property.mjs (fast local pre-validation / a locally-generated error before making a
// network call) and daemon.mjs (the authoritative check - a network client's own validation must
// never be trusted). Two independently-maintained copies of this list is a drift bug waiting to
// happen, so it lives here once.
//
// Deliberately does NOT include DENSITY_SCALE - that's a compile-time WGSL constant duplicated
// manually in LitboxCommon.wgsl and RaytracedResources.DENSITY_SCALE (see CLAUDE.md), not a
// runtime-settable property. It also does NOT include debugView/debugViewScale/debugViewMipLevel/
// debugSolidColor - those are select-debug-view's job, not this skill's, since they're meaningless
// without an active debug view.

// Must match DenoiserTunables in src/litbox/simulation.ts exactly.
export const DENOISER_TUNABLE_KEYS = [
    'varianceScale',
    'darknessNoiseFloor',
    'maxBlurMip',
    'densityBlurFalloff',
    'albedoSensitivity',
    'densitySensitivity',
    'normalSensitivity',
    'sigmaLuminanceTight',
    'sigmaLuminanceLoose',
    'kLuminance',
    'maxSplitDistance',
    'detailMaxSplitDistance',
    'albedoLuminanceThreshold',
    'albedoChromaThreshold',
    'logDensityThreshold',
    'volatilityThreshold',
    'detailThreshold',
    'varianceGateScale',
];

// Must match SimulationTunables in src/litbox/simulation.ts exactly.
export const SIMULATION_TUNABLE_KEYS = [
    'raysPerFrame',
    'integrationInterval',
    'photonBounces',
    'surfaceBias',
    'ambientScatterSoftness',
];

export const STATIC_PROPERTY_NAMES = [
    'exposure', 'tonemap', 'denoiser',
    ...DENOISER_TUNABLE_KEYS.map((k) => `denoiser.${k}`),
    ...SIMULATION_TUNABLE_KEYS.map((k) => `simulation.${k}`),
    'simulation.width', 'simulation.height',
];

/**
 * Parses a --property/--value pair into a typed, validated command the daemon can dispatch, or
 * throws with the full list of valid static names (scene-slider.<label> is deliberately excluded
 * from this list, since valid labels are scene-dependent and only the daemon - holding the live
 * page - can enumerate them; an unrecognized scene-slider.* name is left for the daemon to reject
 * with the active scene's real labels).
 */
export function parsePropertyArg(property, rawValue) {
    if (property.startsWith('scene-slider.')) {
        const label = property.slice('scene-slider.'.length);
        if (!label) throw new Error('scene-slider.<label> requires a label, e.g. --property "scene-slider.Light Intensity"');
        const value = Number(rawValue);
        if (Number.isNaN(value)) throw new Error(`scene-slider values must be numeric, got: ${rawValue}`);
        return { kind: 'scene-slider', label, value };
    }

    if (property === 'exposure') {
        if (rawValue === 'auto' || rawValue === 'null') return { kind: 'exposure', value: null };
        const value = Number(rawValue);
        if (Number.isNaN(value)) throw new Error(`exposure must be numeric or "auto" to clear it, got: ${rawValue}`);
        return { kind: 'exposure', value };
    }

    if (property === 'tonemap' || property === 'denoiser') {
        const value = parseBoolean(rawValue);
        if (value === null) throw new Error(`${property} must be true/false/1/0, got: ${rawValue}`);
        return { kind: property, value };
    }

    if (property.startsWith('denoiser.')) {
        const key = property.slice('denoiser.'.length);
        if (!DENOISER_TUNABLE_KEYS.includes(key)) {
            throw new Error(`Unknown denoiser tunable "${key}". Valid tunables: ${DENOISER_TUNABLE_KEYS.join(', ')}`);
        }
        const value = Number(rawValue);
        if (Number.isNaN(value)) throw new Error(`denoiser.${key} must be numeric, got: ${rawValue}`);
        return { kind: 'denoiser-tunable', key, value };
    }

    if (property === 'simulation.width' || property === 'simulation.height') {
        // Resolution is deliberately NOT one of SIMULATION_TUNABLE_KEYS (see
        // LitboxSceneRenderer.resizeSimulation's own doc comment - it needs an async GPU resource
        // rebuild, not a plain property write) - handled as its own kind, which resolves the
        // OTHER dimension from the simulation's current live resolution so a single-dimension
        // call still produces a valid resizeSimulation(width, height) pair.
        const dimension = property === 'simulation.width' ? 'width' : 'height';
        const value = Number(rawValue);
        if (Number.isNaN(value)) throw new Error(`${property} must be numeric, got: ${rawValue}`);
        return { kind: 'simulation-size', dimension, value };
    }

    if (property.startsWith('simulation.')) {
        const key = property.slice('simulation.'.length);
        if (!SIMULATION_TUNABLE_KEYS.includes(key)) {
            throw new Error(`Unknown simulation tunable "${key}". Valid tunables: ${SIMULATION_TUNABLE_KEYS.join(', ')}`);
        }
        const value = Number(rawValue);
        if (Number.isNaN(value)) throw new Error(`simulation.${key} must be numeric, got: ${rawValue}`);
        return { kind: 'simulation-tunable', key, value };
    }

    throw new Error(`Unknown property "${property}". Valid properties: ${STATIC_PROPERTY_NAMES.join(', ')}, scene-slider.<label>`);
}

function parseBoolean(raw) {
    if (raw === 'true' || raw === '1') return true;
    if (raw === 'false' || raw === '0') return false;
    return null;
}
