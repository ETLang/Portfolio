// Unit quad, two triangles, matching SceneObject.scale semantics ([-0.5, 0.5]^2 local space).
const QUAD_VERTICES = new Float32Array([
    -0.5, -0.5,
    0.5, -0.5,
    0.5, 0.5,
    -0.5, -0.5,
    0.5, 0.5,
    -0.5, 0.5,
]);

export const QUAD_VERTEX_COUNT = QUAD_VERTICES.length / 2;

/** Vertex buffer layout for pipelines that draw the quad mesh (position-only, float32x2). */
export const QUAD_VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout = {
    arrayStride: 4 * 2,
    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
};

let sharedDevice: GPUDevice | null = null;
let sharedVertexBuffer: GPUBuffer | null = null;

/**
 * The single GPUBuffer backing the quad mesh, shared by every pipeline that
 * draws it - there's exactly one copy of this data on the GPU regardless of
 * how many places reference it.
 */
export function getQuadVertexBuffer(device: GPUDevice): GPUBuffer {
    if (sharedDevice !== device) {
        sharedVertexBuffer = device.createBuffer({
            size: QUAD_VERTICES.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(sharedVertexBuffer.getMappedRange()).set(QUAD_VERTICES);
        sharedVertexBuffer.unmap();
        sharedDevice = device;
    }
    return sharedVertexBuffer!;
}
