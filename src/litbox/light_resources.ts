import type {
    AmbientLight,
    AnyLight,
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

// Must match the LightTransform/LightProperties struct strides in
// simulation-consuming WGSL shaders (once written).
const LIGHT_TRANSFORM_STRIDE_BYTES = 32;
const LIGHT_PROPERTIES_STRIDE_BYTES = 32;

type Entry = AnyLight & { kind: number; pinch: number };

/**
 * Consolidated storage for all 5 light types. Lights have no draw behavior
 * of their own (pure simulation input), so rather than 5 near-empty manager
 * classes this builds two flattened GPU storage buffers (one unified
 * transform struct and one unified properties struct per light) for the
 * future simulation pass to consume, plus per-kind CPU-side accessors for
 * anything that needs typed fields.
 *
 * Transform and properties are deliberately separate buffers: a light's
 * worldPosition/direction only change when its owning SceneObject's transform
 * does (see LitboxScene's transform dynamic/dirty marking, which cascades to
 * every light owned anywhere in the affected subtree), while color/intensity/
 * bounces/pinch change only when the light itself is marked dynamic/dirty -
 * two independent update schedules that would otherwise force an unrelated
 * rewrite of one when only the other changed.
 */
export class LightResources {
    private device: GPUDevice;
    private transformBuffer: GPUBuffer;
    private propertiesBuffer: GPUBuffer;
    private count = 0;

    private flatEntries: Entry[] = [];
    private ownerIndex = new Map<number, number[]>();

    private pointLights: PointLight[] = [];
    private spotlights: Spotlight[] = [];
    private laserLights: LaserLight[] = [];
    private directionalLights: DirectionalLight[] = [];
    private ambientLights: AmbientLight[] = [];

    constructor(device: GPUDevice) {
        this.device = device;
        this.transformBuffer = this.createBuffer(1, LIGHT_TRANSFORM_STRIDE_BYTES);
        this.propertiesBuffer = this.createBuffer(1, LIGHT_PROPERTIES_STRIDE_BYTES);
    }

    public getTransformBuffer(): GPUBuffer {
        return this.transformBuffer;
    }

    public getPropertiesBuffer(): GPUBuffer {
        return this.propertiesBuffer;
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

        this.flatEntries = [
            ...this.pointLights.map(l => ({ ...l, kind: LIGHT_KIND.point, pinch: 0 })),
            ...this.spotlights.map(l => ({ ...l, kind: LIGHT_KIND.spot, pinch: l.pinch })),
            ...this.laserLights.map(l => ({ ...l, kind: LIGHT_KIND.laser, pinch: 0 })),
            ...this.directionalLights.map(l => ({ ...l, kind: LIGHT_KIND.directional, pinch: 0 })),
            ...this.ambientLights.map(l => ({ ...l, kind: LIGHT_KIND.ambient, pinch: 0 })),
        ];

        this.ownerIndex = new Map();
        this.flatEntries.forEach((entry, i) => {
            const indices = this.ownerIndex.get(entry.ownerId);
            if (indices) {
                indices.push(i);
            } else {
                this.ownerIndex.set(entry.ownerId, [i]);
            }
        });

        this.count = this.flatEntries.length;
        const elementCount = Math.max(1, this.flatEntries.length);
        if (this.transformBuffer.size < elementCount * LIGHT_TRANSFORM_STRIDE_BYTES) {
            this.transformBuffer.destroy();
            this.transformBuffer = this.createBuffer(elementCount, LIGHT_TRANSFORM_STRIDE_BYTES);
        }
        if (this.propertiesBuffer.size < elementCount * LIGHT_PROPERTIES_STRIDE_BYTES) {
            this.propertiesBuffer.destroy();
            this.propertiesBuffer = this.createBuffer(elementCount, LIGHT_PROPERTIES_STRIDE_BYTES);
        }

        if (this.flatEntries.length === 0) {
            return;
        }

        const transformData = new ArrayBuffer(this.flatEntries.length * LIGHT_TRANSFORM_STRIDE_BYTES);
        const transformView = new DataView(transformData);
        const propertiesData = new ArrayBuffer(this.flatEntries.length * LIGHT_PROPERTIES_STRIDE_BYTES);
        const propertiesView = new DataView(propertiesData);

        this.flatEntries.forEach((entry, i) => {
            this.writeTransformInto(transformView, i * LIGHT_TRANSFORM_STRIDE_BYTES, entry, sceneGraph);
            this.writePropertiesInto(propertiesView, i * LIGHT_PROPERTIES_STRIDE_BYTES, entry);
        });

        this.device.queue.writeBuffer(this.transformBuffer, 0, transformData);
        this.device.queue.writeBuffer(this.propertiesBuffer, 0, propertiesData);
    }

    /** Targeted re-upload of the transform data for every light owned by `ownerId`. No-ops if the owner has no lights. */
    public refreshTransform(ownerId: number, sceneGraph: SceneGraph): void {
        const indices = this.ownerIndex.get(ownerId);
        if (!indices) {
            return;
        }
        for (const i of indices) {
            const scratch = new ArrayBuffer(LIGHT_TRANSFORM_STRIDE_BYTES);
            this.writeTransformInto(new DataView(scratch), 0, this.flatEntries[i], sceneGraph);
            this.device.queue.writeBuffer(this.transformBuffer, i * LIGHT_TRANSFORM_STRIDE_BYTES, scratch);
        }
    }

    /** Targeted re-upload of the properties data for every light owned by `ownerId`. No-ops if the owner has no lights. */
    public refreshProperties(ownerId: number): void {
        const indices = this.ownerIndex.get(ownerId);
        if (!indices) {
            return;
        }
        for (const i of indices) {
            const scratch = new ArrayBuffer(LIGHT_PROPERTIES_STRIDE_BYTES);
            this.writePropertiesInto(new DataView(scratch), 0, this.flatEntries[i]);
            this.device.queue.writeBuffer(this.propertiesBuffer, i * LIGHT_PROPERTIES_STRIDE_BYTES, scratch);
        }
    }

    private writeTransformInto(view: DataView, base: number, entry: Entry, sceneGraph: SceneGraph): void {
        const world = sceneGraph.getWorldTransform(entry.ownerId);
        const obj = sceneGraph.getObject(entry.ownerId);
        const rotationRadians = ((obj?.rotation ?? 0) * Math.PI) / 180;

        // worldPosition: vec4 (xyz + pad)
        view.setFloat32(base + 0, world[12], true);
        view.setFloat32(base + 4, world[13], true);
        view.setFloat32(base + 8, world[14], true);
        view.setFloat32(base + 12, 0, true);
        // direction: vec4 (xyz + pad), derived from owner rotation
        view.setFloat32(base + 16, Math.cos(rotationRadians), true);
        view.setFloat32(base + 20, Math.sin(rotationRadians), true);
        view.setFloat32(base + 24, 0, true);
        view.setFloat32(base + 28, 0, true);
    }

    private writePropertiesInto(view: DataView, base: number, entry: Entry): void {
        // color: vec4
        view.setFloat32(base + 0, entry.color.r, true);
        view.setFloat32(base + 4, entry.color.g, true);
        view.setFloat32(base + 8, entry.color.b, true);
        view.setFloat32(base + 12, entry.color.a, true);
        // kind, intensity, bounces, pinch
        view.setUint32(base + 16, entry.kind, true);
        view.setFloat32(base + 20, entry.intensity, true);
        view.setFloat32(base + 24, entry.bounces, true);
        view.setFloat32(base + 28, entry.pinch, true);
    }

    private createBuffer(elementCount: number, strideBytes: number): GPUBuffer {
        return this.device.createBuffer({
            size: elementCount * strideBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
    }
}
