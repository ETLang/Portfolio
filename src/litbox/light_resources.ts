import type {
    AmbientLight,
    DirectionalLight,
    LaserLight,
    PointLight,
    Scene,
    Spotlight,
} from './scene.ts';
import type { SceneGraph } from './scene_graph.ts';

// erasableSyntaxOnly forbids `enum` - use a plain lookup instead.
const LIGHT_KIND: Record<'point' | 'spot' | 'laser' | 'directional' | 'ambient', number> = {
    point: 0,
    spot: 1,
    laser: 2,
    directional: 3,
    ambient: 4,
};

// Must match the LightData struct stride in simulation-consuming WGSL shaders (once written).
const LIGHT_DATA_STRIDE_BYTES = 64;

/**
 * Consolidated storage for all 5 light types. Lights have no draw behavior
 * of their own (pure simulation input), so rather than 5 near-empty manager
 * classes this builds a single flattened GPU storage buffer (one unified
 * struct per light) for the future simulation pass to consume, plus
 * per-kind CPU-side accessors for anything that needs typed fields.
 */
export class LightResources {
    private device: GPUDevice;
    private buffer: GPUBuffer;
    private count = 0;

    private pointLights: PointLight[] = [];
    private spotlights: Spotlight[] = [];
    private laserLights: LaserLight[] = [];
    private directionalLights: DirectionalLight[] = [];
    private ambientLights: AmbientLight[] = [];

    constructor(device: GPUDevice) {
        this.device = device;
        this.buffer = this.createBuffer(1);
    }

    public getBuffer(): GPUBuffer {
        return this.buffer;
    }

    public getCount(): number {
        return this.count;
    }

    public getPointLights(): readonly PointLight[] {
        return this.pointLights;
    }

    public getSpotlights(): readonly Spotlight[] {
        return this.spotlights;
    }

    public getLaserLights(): readonly LaserLight[] {
        return this.laserLights;
    }

    public getDirectionalLights(): readonly DirectionalLight[] {
        return this.directionalLights;
    }

    public getAmbientLights(): readonly AmbientLight[] {
        return this.ambientLights;
    }

    public updateFromScene(scene: Scene, sceneGraph: SceneGraph): void {
        this.pointLights = scene.pointLights;
        this.spotlights = scene.spotlights;
        this.laserLights = scene.laserLights;
        this.directionalLights = scene.directionalLights;
        this.ambientLights = scene.ambientLights;

        type Entry = { ownerId: number; color: { r: number; g: number; b: number; a: number }; intensity: number; bounces: number; kind: number; pinch: number };
        const entries: Entry[] = [
            ...this.pointLights.map(l => ({ ...l, kind: LIGHT_KIND.point, pinch: 0 })),
            ...this.spotlights.map(l => ({ ...l, kind: LIGHT_KIND.spot })),
            ...this.laserLights.map(l => ({ ...l, kind: LIGHT_KIND.laser, pinch: 0 })),
            ...this.directionalLights.map(l => ({ ...l, kind: LIGHT_KIND.directional, pinch: 0 })),
            ...this.ambientLights.map(l => ({ ...l, kind: LIGHT_KIND.ambient, pinch: 0 })),
        ];

        this.count = entries.length;
        if (this.buffer.size < Math.max(1, entries.length) * LIGHT_DATA_STRIDE_BYTES) {
            this.buffer.destroy();
            this.buffer = this.createBuffer(Math.max(1, entries.length));
        }

        if (entries.length === 0) {
            return;
        }

        const arrayBuffer = new ArrayBuffer(entries.length * LIGHT_DATA_STRIDE_BYTES);
        const view = new DataView(arrayBuffer);

        entries.forEach((entry, i) => {
            const base = i * LIGHT_DATA_STRIDE_BYTES;
            const world = sceneGraph.getWorldTransform(entry.ownerId);
            const obj = sceneGraph.getObject(entry.ownerId);
            const rotationRadians = ((obj?.rotation ?? 0) * Math.PI) / 180;

            // color: vec4
            view.setFloat32(base + 0, entry.color.r, true);
            view.setFloat32(base + 4, entry.color.g, true);
            view.setFloat32(base + 8, entry.color.b, true);
            view.setFloat32(base + 12, entry.color.a, true);
            // worldPosition: vec4 (xyz + pad)
            view.setFloat32(base + 16, world[12], true);
            view.setFloat32(base + 20, world[13], true);
            view.setFloat32(base + 24, world[14], true);
            view.setFloat32(base + 28, 0, true);
            // direction: vec4 (xyz + pad), derived from owner rotation
            view.setFloat32(base + 32, Math.cos(rotationRadians), true);
            view.setFloat32(base + 36, Math.sin(rotationRadians), true);
            view.setFloat32(base + 40, 0, true);
            view.setFloat32(base + 44, 0, true);
            // kind, intensity, bounces, pinch
            view.setUint32(base + 48, entry.kind, true);
            view.setFloat32(base + 52, entry.intensity, true);
            view.setFloat32(base + 56, entry.bounces, true);
            view.setFloat32(base + 60, entry.pinch, true);
        });

        this.device.queue.writeBuffer(this.buffer, 0, arrayBuffer);
    }

    private createBuffer(elementCount: number): GPUBuffer {
        return this.device.createBuffer({
            size: elementCount * LIGHT_DATA_STRIDE_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
    }
}
