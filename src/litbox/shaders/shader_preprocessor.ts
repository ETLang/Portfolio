import litboxCommonSource from './LitboxCommon.wgsl?raw';

// WGSL/WebGPU has no native #include, #define, or #ifdef; this is a minimal project-local
// preprocessor emulating a C-style one, resolved here by a single line-oriented pass before the
// source reaches GPUDevice.createShaderModule. Every shader-loading TS file must call
// preprocessShader on its raw ?raw import - see any of tonemap.ts, debug_view.ts,
// raytraced_resources.ts, sprite_resources.ts, simulation.ts, convert_photon_irradiance_to_hdr.ts
// for the pattern.
//
// Supported directives:
//   #include "Name.wgsl"        - inline another known file's contents (see KNOWN_INCLUDES
//                                  below). Each named file is inlined at most once per top-level
//                                  call - WGSL has no include guards, and every .wgsl file here
//                                  is expected to declare top-level consts/fns, so silently
//                                  re-inlining the same file would just be a duplicate-declaration
//                                  error. Unlike real C, this is automatic, not opt-in via guards.
//   #define NAME [value]        - defines NAME, satisfying #ifdef NAME. An optional trailing
//                                  value is substituted (whole-identifier match, single pass, not
//                                  recursively re-expanded - good enough for simple constants, not
//                                  a full macro-expansion engine) into every later active line.
//   #undef NAME
//   #ifdef NAME / #ifndef NAME  - #else and #endif close either form; nesting is supported.
//   ... #else ... #endif          No #elif and no #if <expression> - not needed for this
//                                  project's scope (simple feature-flag-style switches); add real
//                                  expression evaluation if that scope grows.
// #define/#undef/#include only take effect inside an active (not #ifdef'd-out) branch, matching
// C. Any other line starting with # is treated as a typo'd/unsupported directive and throws
// (checked regardless of branch activity, so a typo inside a currently-inactive branch is still
// caught) rather than being silently passed through to the shader compiler, which would just
// report the '#' as invalid WGSL syntax at a callsite far from the actual mistake.
const KNOWN_INCLUDES: Record<string, string> = {
    'LitboxCommon.wgsl': litboxCommonSource,
};

const INCLUDE_RE = /^#include\s+"([^"]+)"\s*$/;
const DEFINE_RE = /^#define\s+(\w+)(?:\s+(.*\S))?\s*$/;
const UNDEF_RE = /^#undef\s+(\w+)\s*$/;
const IFDEF_RE = /^#(ifdef|ifndef)\s+(\w+)\s*$/;
const ELSE_RE = /^#else\s*$/;
const ENDIF_RE = /^#endif\s*$/;
const DIRECTIVE_LINE_RE = /^#\S*/;

/** `true` defines a bare flag (e.g. for #ifdef); a string defines a flag with a substitutable value. */
export type ShaderDefines = Record<string, string | true>;

/** Preprocesses `source`'s #include/#define/#ifdef directives (see the file header for the exact supported subset) into plain WGSL, ready for GPUDevice.createShaderModule. `defines` seeds the macro table before the first line is processed - e.g. `{ DEBUG_MODE: true }` to satisfy an #ifdef DEBUG_MODE the caller wants active. */
export function preprocessShader(source: string, defines: ShaderDefines = {}): string {
    return preprocessFrom(source, KNOWN_INCLUDES, defines);
}

/**
 * Core resolver, parameterized over the includes map so tests can exercise nested/circular/
 * diamond #include cases without depending on real files under shaders/ - see preprocessShader
 * for the entry point real shader-loading code should use.
 */
export function preprocessFrom(source: string, includes: Record<string, string>, defines: ShaderDefines = {}): string {
    const macros = new Map<string, string>();
    for (const [name, value] of Object.entries(defines)) {
        macros.set(name, value === true ? '' : value);
    }
    return preprocessRecursive(source, includes, [], new Set(), macros).join('\n');
}

/**
 * `ancestors` is the current #include chain (for a clear circular-include error); `included` is
 * every file name inlined anywhere so far in this resolve (not just on the current path) - see
 * the file header for why that's an automatic once-per-file dedup rather than opt-in guards.
 * `macros` is shared by reference across the whole recursive tree so #define/#undef persist
 * across #include boundaries in both directions, matching C.
 */
function preprocessRecursive(
    source: string,
    includes: Record<string, string>,
    ancestors: string[],
    included: Set<string>,
    macros: Map<string, string>,
): string[] {
    const output: string[] = [];
    // Each frame covers one #ifdef/#ifndef...#endif chain: `active` is whether *this* branch's
    // content should currently be emitted; `parentActive` is whether the enclosing scope was
    // active when this frame was pushed (an #else can't become active if it wasn't); `taken` is
    // whether any branch in this chain has been active yet (so at most one branch of an #else pair
    // ever emits).
    const conditionalStack: { active: boolean; parentActive: boolean; taken: boolean }[] = [];
    const isActive = () => conditionalStack.every((frame) => frame.active);

    for (const rawLine of source.split('\n')) {
        const line = rawLine.trim();

        const ifdefMatch = IFDEF_RE.exec(line);
        if (ifdefMatch) {
            const parentActive = isActive();
            const [, kind, name] = ifdefMatch;
            const wantDefined = kind === 'ifdef';
            const active = parentActive && macros.has(name) === wantDefined;
            conditionalStack.push({ active, parentActive, taken: active });
            continue;
        }
        if (ELSE_RE.test(line)) {
            const frame = conditionalStack[conditionalStack.length - 1];
            if (!frame) {
                throw new Error('WGSL preprocessor: #else with no matching #ifdef/#ifndef.');
            }
            frame.active = frame.parentActive && !frame.taken;
            frame.taken = frame.taken || frame.active;
            continue;
        }
        if (ENDIF_RE.test(line)) {
            if (conditionalStack.length === 0) {
                throw new Error('WGSL preprocessor: #endif with no matching #ifdef/#ifndef.');
            }
            conditionalStack.pop();
            continue;
        }

        const active = isActive();

        const defineMatch = DEFINE_RE.exec(line);
        if (defineMatch) {
            if (active) {
                const [, name, value] = defineMatch;
                macros.set(name, value ?? '');
            }
            continue;
        }
        const undefMatch = UNDEF_RE.exec(line);
        if (undefMatch) {
            if (active) {
                macros.delete(undefMatch[1]);
            }
            continue;
        }
        const includeMatch = INCLUDE_RE.exec(line);
        if (includeMatch) {
            if (active) {
                const includeName = includeMatch[1];
                if (ancestors.includes(includeName)) {
                    throw new Error(`Circular WGSL #include: ${[...ancestors, includeName].join(' -> ')}`);
                }
                if (!included.has(includeName)) {
                    const includeSource = includes[includeName];
                    if (includeSource === undefined) {
                        throw new Error(`Unknown WGSL #include: "${includeName}"`);
                    }
                    included.add(includeName);
                    output.push(...preprocessRecursive(includeSource, includes, [...ancestors, includeName], included, macros));
                }
            }
            continue;
        }

        if (DIRECTIVE_LINE_RE.test(line)) {
            throw new Error(`WGSL preprocessor: unrecognized directive: "${line}"`);
        }

        if (active) {
            output.push(applyMacros(rawLine, macros));
        }
    }

    if (conditionalStack.length > 0) {
        throw new Error('WGSL preprocessor: unterminated #ifdef/#ifndef (missing #endif).');
    }

    return output;
}

function applyMacros(line: string, macros: Map<string, string>): string {
    let result = line;
    for (const [name, value] of macros) {
        if (value === '') {
            continue; // flag-only define - nothing to substitute
        }
        result = result.replace(new RegExp(`\\b${name}\\b`, 'g'), value);
    }
    return result;
}
