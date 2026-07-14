import { describe, expect, it } from 'vitest';
import {
    ELLIPSE_REGION_VERTICES,
    ELLIPSE_SEGMENTS,
    PRIMITIVE_MESH_REGIONS,
    PRIMITIVE_MESH_VERTEX_BUFFER_LAYOUT,
    QUAD_REGION_VERTICES,
    RECT_REGION_VERTICES,
} from '../primitive_mesh.ts';

const FLOATS_PER_VERTEX = 5;

describe('primitive_mesh vertex data', () => {
    it('quad region (shape 0): 6 vertices, unit square corners, constant +Z normal', () => {
        expect(QUAD_REGION_VERTICES).toEqual([
            -0.5, -0.5, 0, 0, 1,
            0.5, -0.5, 0, 0, 1,
            0.5, 0.5, 0, 0, 1,
            -0.5, -0.5, 0, 0, 1,
            0.5, 0.5, 0, 0, 1,
            -0.5, 0.5, 0, 0, 1,
        ]);
    });

    it('rect region (shape 1): 12 vertices / 4 edge triangles, ported from RTRect.cs', () => {
        expect(RECT_REGION_VERTICES).toEqual([
            // Left triangle: normal (-1,0,0)
            -0.5, 0.5, -1, 0, 0,
            0, 0, -1, 0, 0,
            -0.5, -0.5, -1, 0, 0,
            // Top triangle: normal (0,1,0)
            0.5, 0.5, 0, 1, 0,
            0, 0, 0, 1, 0,
            -0.5, 0.5, 0, 1, 0,
            // Right triangle: normal (1,0,0)
            0.5, -0.5, 1, 0, 0,
            0, 0, 1, 0, 0,
            0.5, 0.5, 1, 0, 0,
            // Bottom triangle: normal (0,-1,0)
            -0.5, -0.5, 0, -1, 0,
            0, 0, 0, -1, 0,
            0.5, -0.5, 0, -1, 0,
        ]);
    });

    it('ellipse region (shape 2): flattened to 96 vertices (32 segments * 3), non-indexed', () => {
        expect(ELLIPSE_SEGMENTS).toBe(32);
        expect(ELLIPSE_REGION_VERTICES.length).toBe(96 * FLOATS_PER_VERTEX);
    });

    it('ellipse region: every 3rd vertex (triangle-local index 0) is the duplicated center, pos (0,0), normal (0,0,1)', () => {
        for (let tri = 0; tri < 32; tri++) {
            const base = tri * 3 * FLOATS_PER_VERTEX;
            expect(ELLIPSE_REGION_VERTICES.slice(base, base + FLOATS_PER_VERTEX)).toEqual([0, 0, 0, 0, 1]);
        }
    });

    it('ellipse region: rim vertices at 0/90/180/270 degrees have position-verbatim normals', () => {
        const rimVertexAt = (segment: number): number[] => {
            // Triangle `segment`'s 3rd vertex (triangle-local index 2) is rim[segment] - see
            // buildEllipseRegionVertices' vertex order (center, next-rim, this-rim).
            const base = (segment * 3 + 2) * FLOATS_PER_VERTEX;
            return ELLIPSE_REGION_VERTICES.slice(base, base + FLOATS_PER_VERTEX);
        };
        const expectClose = (actual: number[], expected: number[]): void => {
            expect(actual.length).toBe(expected.length);
            for (let i = 0; i < expected.length; i++) {
                expect(actual[i]).toBeCloseTo(expected[i], 6);
            }
        };

        expectClose(rimVertexAt(0), [0.5, 0, 0.5, 0, 0]); // 0 degrees
        expectClose(rimVertexAt(8), [0, 0.5, 0, 0.5, 0]); // 90 degrees
        expectClose(rimVertexAt(16), [-0.5, 0, -0.5, 0, 0]); // 180 degrees
        expectClose(rimVertexAt(24), [0, -0.5, 0, -0.5, 0]); // 270 degrees
    });
});

describe('PRIMITIVE_MESH_REGIONS', () => {
    it('is contiguous, non-overlapping, and matches each region array\'s vertex count', () => {
        expect(PRIMITIVE_MESH_REGIONS[0]).toEqual({ firstVertex: 0, vertexCount: 6 });
        expect(PRIMITIVE_MESH_REGIONS[1]).toEqual({ firstVertex: 6, vertexCount: 12 });
        expect(PRIMITIVE_MESH_REGIONS[2]).toEqual({ firstVertex: 18, vertexCount: 96 });

        expect(QUAD_REGION_VERTICES.length / FLOATS_PER_VERTEX).toBe(PRIMITIVE_MESH_REGIONS[0].vertexCount);
        expect(RECT_REGION_VERTICES.length / FLOATS_PER_VERTEX).toBe(PRIMITIVE_MESH_REGIONS[1].vertexCount);
        expect(ELLIPSE_REGION_VERTICES.length / FLOATS_PER_VERTEX).toBe(PRIMITIVE_MESH_REGIONS[2].vertexCount);
    });
});

describe('PRIMITIVE_MESH_VERTEX_BUFFER_LAYOUT', () => {
    it('interleaves position (float32x2) and normal (float32x3) in a 20-byte stride', () => {
        expect(PRIMITIVE_MESH_VERTEX_BUFFER_LAYOUT.arrayStride).toBe(20);
        expect(PRIMITIVE_MESH_VERTEX_BUFFER_LAYOUT.attributes).toEqual([
            { shaderLocation: 0, offset: 0, format: 'float32x2' },
            { shaderLocation: 1, offset: 8, format: 'float32x3' },
        ]);
    });
});
