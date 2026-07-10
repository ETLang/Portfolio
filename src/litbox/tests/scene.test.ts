import { describe, expect, it } from 'vitest';
import { parseScene, parseUvTransform } from '../scene.ts';

describe('parseUvTransform', () => {
    it('parses a row-major 2x3 array string into a UvTransform', () => {
        expect(parseUvTransform('[[1, 0, 0], [0, 1, 0]]')).toEqual({ a: 1, b: 0, c: 0, d: 0, e: 1, f: 0 });
        expect(parseUvTransform('[[0.5, 0, 0.25], [0, 0.5, 0.75]]')).toEqual({ a: 0.5, b: 0, c: 0.25, d: 0, e: 0.5, f: 0.75 });
    });

    it('throws on a malformed shape', () => {
        expect(() => parseUvTransform('[[1, 0], [0, 1]]')).toThrow(/malformed uvTransform/);
        expect(() => parseUvTransform('[[1, 0, 0]]')).toThrow(/malformed uvTransform/);
    });
});

describe('parseScene', () => {
    it('defaults textureAtlasKeys to an empty array when absent', () => {
        const scene = parseScene(JSON.stringify({}));
        expect(scene.textureAtlasKeys).toEqual([]);
    });

    it('parses textureAtlasKeys entries, converting uvTransform from its string form', () => {
        const json = JSON.stringify({
            textureAtlasKeys: [
                { textureName: 'Moon', atlasName: 'Atlas1', uvTransform: '[[0.5, 0, 0], [0, 0.5, 0.5]]' },
            ],
        });

        const scene = parseScene(json);

        expect(scene.textureAtlasKeys).toEqual([
            { textureName: 'Moon', atlasName: 'Atlas1', uvTransform: { a: 0.5, b: 0, c: 0, d: 0, e: 0.5, f: 0.5 } },
        ]);
    });
});
