import { describe, expect, it } from 'vitest';
import { preprocessFrom } from '../shaders/shader_preprocessor.ts';

describe('preprocessFrom - #include', () => {
    it('inlines a single include', () => {
        const includes = { 'A.wgsl': 'const A: f32 = 1.0;' };
        const result = preprocessFrom('#include "A.wgsl"\nfn main() {}', includes);
        expect(result).toBe('const A: f32 = 1.0;\nfn main() {}');
    });

    it('resolves an include nested inside another include', () => {
        const includes = {
            'A.wgsl': '#include "B.wgsl"\nconst A: f32 = 1.0;',
            'B.wgsl': 'const B: f32 = 2.0;',
        };
        const result = preprocessFrom('#include "A.wgsl"', includes);
        expect(result).toBe('const B: f32 = 2.0;\nconst A: f32 = 1.0;');
    });

    it('inlines a diamond-shared include only once', () => {
        const includes = {
            'A.wgsl': '#include "C.wgsl"\nconst A: f32 = 1.0;',
            'B.wgsl': '#include "C.wgsl"\nconst B: f32 = 2.0;',
            'C.wgsl': 'const C: f32 = 3.0;',
        };
        const result = preprocessFrom('#include "A.wgsl"\n#include "B.wgsl"', includes);
        expect(result).toBe('const C: f32 = 3.0;\nconst A: f32 = 1.0;\nconst B: f32 = 2.0;');
    });

    it('throws a clear error on a circular include', () => {
        const includes = {
            'A.wgsl': '#include "B.wgsl"',
            'B.wgsl': '#include "A.wgsl"',
        };
        expect(() => preprocessFrom('#include "A.wgsl"', includes)).toThrow('Circular WGSL #include: A.wgsl -> B.wgsl -> A.wgsl');
    });

    it('throws on an unknown include', () => {
        expect(() => preprocessFrom('#include "Missing.wgsl"', {})).toThrow('Unknown WGSL #include: "Missing.wgsl"');
    });
});

describe('preprocessFrom - #define/#undef', () => {
    it('substitutes a value macro on later lines', () => {
        const result = preprocessFrom('#define SCALE 8192.0\nconst x = SCALE;', {});
        expect(result).toBe('const x = 8192.0;');
    });

    it('does not substitute inside a larger identifier (whole-word match only)', () => {
        const result = preprocessFrom('#define SCALE 8192.0\nconst x = SCALE_FACTOR;', {});
        expect(result).toBe('const x = SCALE_FACTOR;');
    });

    it('a bare #define (no value) is a flag only - no text substitution', () => {
        const result = preprocessFrom('#define FEATURE\nconst x = FEATURE;', {});
        expect(result).toBe('const x = FEATURE;');
    });

    it('#undef removes a prior #define', () => {
        const result = preprocessFrom('#define SCALE 2.0\n#undef SCALE\n#ifdef SCALE\nconst x = 1.0;\n#endif', {});
        expect(result).toBe('');
    });

    it('caller-seeded defines are visible from the first line', () => {
        const result = preprocessFrom('#ifdef DEBUG_MODE\nconst debug = true;\n#endif', {}, { DEBUG_MODE: true });
        expect(result).toBe('const debug = true;');
    });

    it('caller-seeded define with a value substitutes like #define', () => {
        const result = preprocessFrom('const x = MAX_LIGHTS;', {}, { MAX_LIGHTS: '8' });
        expect(result).toBe('const x = 8;');
    });
});

describe('preprocessFrom - #ifdef/#ifndef/#else', () => {
    it('#ifdef keeps content when defined, drops it when not', () => {
        expect(preprocessFrom('#define FOO\n#ifdef FOO\nkept\n#endif', {})).toBe('kept');
        expect(preprocessFrom('#ifdef FOO\ndropped\n#endif', {})).toBe('');
    });

    it('#ifndef keeps content when NOT defined', () => {
        expect(preprocessFrom('#ifndef FOO\nkept\n#endif', {})).toBe('kept');
        expect(preprocessFrom('#define FOO\n#ifndef FOO\ndropped\n#endif', {})).toBe('');
    });

    it('#else takes the opposite branch', () => {
        expect(preprocessFrom('#define FOO\n#ifdef FOO\na\n#else\nb\n#endif', {})).toBe('a');
        expect(preprocessFrom('#ifdef FOO\na\n#else\nb\n#endif', {})).toBe('b');
    });

    it('supports nesting', () => {
        const source = '#define OUTER\n#ifdef OUTER\n#ifdef INNER\nboth\n#else\nouter-only\n#endif\n#endif';
        expect(preprocessFrom(source, {})).toBe('outer-only');
    });

    it('an inactive branch does not process nested #define/#include - directives inside it are inert', () => {
        const includes = { 'Missing.wgsl': 'never reached' };
        const source = '#ifdef FOO\n#include "Missing.wgsl"\n#define BAR 1\n#endif\nconst x = BAR;';
        expect(preprocessFrom(source, includes)).toBe('const x = BAR;');
    });

    it('#define/#ifdef state set inside an included file is visible after the #include returns', () => {
        const includes = { 'Setup.wgsl': '#define FOO' };
        const source = '#include "Setup.wgsl"\n#ifdef FOO\nkept\n#endif';
        expect(preprocessFrom(source, includes)).toBe('kept');
    });

    it('throws on an unterminated #ifdef', () => {
        expect(() => preprocessFrom('#ifdef FOO\nx', {})).toThrow('unterminated #ifdef/#ifndef');
    });

    it('throws on a stray #endif', () => {
        expect(() => preprocessFrom('#endif', {})).toThrow('#endif with no matching #ifdef/#ifndef');
    });

    it('throws on a stray #else', () => {
        expect(() => preprocessFrom('#else', {})).toThrow('#else with no matching #ifdef/#ifndef');
    });
});

describe('preprocessFrom - unsupported directives', () => {
    it('throws on an unrecognized directive, even inside an inactive branch', () => {
        expect(() => preprocessFrom('#elif FOO', {})).toThrow('unrecognized directive');
        expect(() => preprocessFrom('#ifdef NEVER_DEFINED\n#elif FOO\n#endif', {})).toThrow('unrecognized directive');
    });
});
