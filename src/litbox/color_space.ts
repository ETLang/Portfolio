import type { Color } from './scene.ts';

// Unity's Inspector color picker always stores/serializes Color values in sRGB (gamma) space,
// regardless of the project's active color space - the automatic sRGB->linear conversion only
// happens when a Color value is bound to a shader property declared `Color` (not `Vector`) via
// Material/MaterialPropertyBlock.SetColor, in a project with Linear color space active (this
// project's Unity source: ProjectSettings.asset's `m_ActiveColorSpace: 1`). The exported scene
// JSON carries the raw, unconverted sRGB values (see LitboxDemoSceneExporter.cs - it reads Color
// fields directly, with no `.linear` conversion), by design: keeping colors in sRGB end-to-end
// through the JSON leaves room for a future color-picker UI authored in the same (human-intuitive)
// space Unity's Inspector uses (linearColorToSrgb exists for that direction - converting a
// GPU/linear-space value back for display/editing - even though nothing calls it yet). Correction
// into linear space happens here instead, applied only at the point a color is packed into a
// GPU-resident uniform/storage buffer - see raytraced_resources.ts's
// and sprite_resources.ts's writePropertiesData for the call sites, and their comments for exactly
// which Color-typed fields this applies to (verified per-field against the corresponding Unity
// shader's Properties block - not every Color-typed field in this project's JSON got Unity's
// automatic conversion; light colors notably did not, since they reach Unity's simulation via
// ComputeShader.SetVector rather than a Material Color property, so they must NOT be converted here
// either, to match Unity's actual - if inconsistent - rendered behavior).

/** The standard sRGB electro-optical transfer function (IEC 61966-2-1), converting one gamma-encoded [0,1] channel to linear. */
export function srgbToLinear(c: number): number {
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** The inverse of srgbToLinear (the sRGB opto-electronic transfer function), converting one linear [0,1] channel to gamma-encoded sRGB. */
export function linearToSrgb(c: number): number {
    return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

/** Converts a Color's r/g/b from sRGB to linear space; alpha is not a color quantity and is passed through unchanged. */
export function srgbColorToLinear(color: Color): Color {
    return { r: srgbToLinear(color.r), g: srgbToLinear(color.g), b: srgbToLinear(color.b), a: color.a };
}

/** Converts a Color's r/g/b from linear to sRGB space; alpha is not a color quantity and is passed through unchanged. */
export function linearColorToSrgb(color: Color): Color {
    return { r: linearToSrgb(color.r), g: linearToSrgb(color.g), b: linearToSrgb(color.b), a: color.a };
}
