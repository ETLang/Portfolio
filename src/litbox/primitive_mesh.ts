// Mesh regions for raytraced primitive shapes, ported from Unity's RTRect.cs/RTEllipse.cs to give
// the NormalRoughness G-Buffer real per-vertex normals instead of one flat constant normal - see
// raytraced_gbuffer.wgsl's computeWorldNormal for how these local normals become world-space ones.
// Deliberately separate from quad_mesh.ts (position-only, float32x2, still shared by
// SpriteResources) - this buffer adds a per-vertex normal and is consumed only by
// RaytracedResources.
//
// Region ids match PRIMITIVE_SHAPE_ID (primitive_shape.ts): 0 = unspecified (flat quad, constant
// +Z normal - the correct rendering for raytraced objects with no distinguishable edge, e.g.
// smoke/clouds/background atmosphere, not a placeholder), 1 = rect (RTRect.cs's 4-facet
// "pinwheel" - only used when primitiveShape is explicitly "rect"), 2 = ellipse (RTEllipse.cs's
// domed fan). All 3 regions are concatenated back-to-back into one GPUBuffer/vertex-range table so
// RaytracedResources can draw any region via draw(vertexCount, ..., firstVertex, ...) without a
// per-shape pipeline or an index buffer.

export interface PrimitiveMeshRegion {
    firstVertex: number;
    vertexCount: number;
}

const FLOATS_PER_VERTEX = 5; // position.xy (2) + normal.xyz (3), interleaved

// Region 0 (unspecified): same 6-vertex flat quad/winding as quad_mesh.ts's QUAD_VERTICES, with a
// constant local +Z normal added per vertex - pixel-identical to the pre-port flat-quad rendering.
// Exported (alongside the other two regions' arrays below) so primitive_mesh.test.ts can pin the
// exact ported numeric data directly, without round-tripping through a GPU buffer stub.
export const QUAD_REGION_VERTICES: number[] = [
    -0.5, -0.5, 0, 0, 1,
    0.5, -0.5, 0, 0, 1,
    0.5, 0.5, 0, 0, 1,
    -0.5, -0.5, 0, 0, 1,
    0.5, 0.5, 0, 0, 1,
    -0.5, 0.5, 0, 0, 1,
];

// Region 1 (rect): ported from RTRect.cs - 12 vertices / 4 triangles, one triangle per edge (that
// edge's 2 corners + the center point (0,0)), all 3 vertices of a triangle sharing that edge's
// outward axis-aligned normal - a faceted "pinwheel" look, not a smooth surface.
export const RECT_REGION_VERTICES: number[] = [
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
];

// Region 2 (ellipse): ported from RTEllipse.cs's 32-segment fan (center + 32 rim points),
// flattened from Unity's original indexed 33-vertex/32-triangle mesh into a plain (non-indexed)
// 96-vertex (32*3) triangle list, since RaytracedResources never binds an index buffer - the
// center vertex is duplicated per triangle instead of shared. Triangle i's vertex order (center,
// rim[(i+1)%32], rim[i]) matches RTEllipse.cs's index triple (0, (i+1)%32+1, i+1) exactly.
//
// The center normal is local (0,0,+1) - RTEllipse.cs uses (0,0,-1), flipped here to match this
// project's existing +Z "faces the camera" convention (see raytraced_gbuffer.wgsl's
// computeWorldNormal) rather than Unity's opposite camera-forward convention. This is a
// self-consistency call, not Unity fidelity: the only G-Buffer normal consumer
// (forward_monte_carlo.wgsl) reads normal.xy exclusively, so the Z sign is not correctness-critical.
//
// Rim vertex normals are each vertex's own local position verbatim (magnitude 0.5, matching
// RTEllipse.cs's raw data exactly rather than pre-normalizing it) - the vertex shader's eventual
// normalize() makes any positive-scalar magnitude choice equivalent, so this keeps the ported data
// traceable 1:1 against RTEllipse.cs.
//
// Generated (not hand-transcribed) to avoid transcription error across 32 segments;
// primitive_mesh.test.ts pins known angles (0/90/180/270 degrees) against exact expected values
// independently of this generator.
export const ELLIPSE_SEGMENTS = 32;
export function buildEllipseRegionVertices(): number[] {
    const verts: number[] = [];
    const rim = (i: number): { x: number; y: number } => {
        const angle = (i / ELLIPSE_SEGMENTS) * 2 * Math.PI;
        return { x: Math.cos(angle) * 0.5, y: Math.sin(angle) * 0.5 };
    };
    for (let i = 0; i < ELLIPSE_SEGMENTS; i++) {
        const next = rim((i + 1) % ELLIPSE_SEGMENTS);
        const cur = rim(i);
        verts.push(0, 0, 0, 0, 1); // center: pos (0,0,0), normal (0,0,1) - see file header
        verts.push(next.x, next.y, next.x, next.y, 0); // next-rim: normal = position verbatim
        verts.push(cur.x, cur.y, cur.x, cur.y, 0); // this-rim: normal = position verbatim
    }
    return verts;
}
export const ELLIPSE_REGION_VERTICES: number[] = buildEllipseRegionVertices();

const ALL_VERTICES: Float32Array = new Float32Array([
    ...QUAD_REGION_VERTICES,
    ...RECT_REGION_VERTICES,
    ...ELLIPSE_REGION_VERTICES,
]);

const QUAD_REGION: PrimitiveMeshRegion = { firstVertex: 0, vertexCount: QUAD_REGION_VERTICES.length / FLOATS_PER_VERTEX };
const RECT_REGION: PrimitiveMeshRegion = { firstVertex: QUAD_REGION.firstVertex + QUAD_REGION.vertexCount, vertexCount: RECT_REGION_VERTICES.length / FLOATS_PER_VERTEX };
const ELLIPSE_REGION: PrimitiveMeshRegion = { firstVertex: RECT_REGION.firstVertex + RECT_REGION.vertexCount, vertexCount: ELLIPSE_REGION_VERTICES.length / FLOATS_PER_VERTEX };

/** Keyed by the same 0/1/2 shape ids as PRIMITIVE_SHAPE_ID (primitive_shape.ts). */
export const PRIMITIVE_MESH_REGIONS: Record<number, PrimitiveMeshRegion> = {
    0: QUAD_REGION,
    1: RECT_REGION,
    2: ELLIPSE_REGION,
};

/**
 * Vertex buffer layout for pipelines that draw the primitive mesh: position (float32x2) + normal
 * (float32x3), interleaved - distinct from quad_mesh.ts's position-only QUAD_VERTEX_BUFFER_LAYOUT.
 */
export const PRIMITIVE_MESH_VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout = {
    arrayStride: FLOATS_PER_VERTEX * 4,
    attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x2' },
        { shaderLocation: 1, offset: 8, format: 'float32x3' },
    ],
};

let sharedDevice: GPUDevice | null = null;
let sharedVertexBuffer: GPUBuffer | null = null;

/**
 * The single GPUBuffer backing the primitive mesh (all 3 regions concatenated), shared by every
 * pipeline that draws it - mirrors quad_mesh.ts's getQuadVertexBuffer.
 */
export function getPrimitiveMeshVertexBuffer(device: GPUDevice): GPUBuffer {
    if (sharedDevice !== device) {
        sharedVertexBuffer = device.createBuffer({
            size: ALL_VERTICES.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(sharedVertexBuffer.getMappedRange()).set(ALL_VERTICES);
        sharedVertexBuffer.unmap();
        sharedDevice = device;
    }
    return sharedVertexBuffer!;
}
