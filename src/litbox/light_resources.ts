import type {
    AmbientLight,
    AnyLight,
    DirectionalLight,
    LaserLight,
    LightKind,
    PointLight,
    Scene,
    Spotlight,
} from './scene.ts';
import type { SceneGraph } from './scene_graph.ts';
import { Entry, PackedUniformArray } from './packed_uniform_array.ts';
import type { TransformResources } from './transform_resources.ts';

// erasableSyntaxOnly forbids `enum` - use a plain lookup instead.
const LIGHT_KIND: Record<LightKind, number> = {
    point: 0,
    spot: 1,
    laser: 2,
    directional: 3,
    ambient: 4,
};

// Must match the LightProperties struct stride in simulation-consuming WGSL shaders (once written).
const LIGHT_PROPERTIES_STRIDE_BYTES = 48;

interface LightRecord {
    light: AnyLight;
    kind: number;
    propertiesEntry: Entry;
}

/**
 * Consolidated storage for all 5 light types. Lights have no draw behavior of their own (pure
 * simulation input), so rather than 5 near-empty manager classes this builds one packed
 * properties array (color, kind, intensity, bounces, pinch, plus a transformIndex pointing into
 * the shared TransformResources array - see its class doc) for a future simulation pass to bind
 * directly as a storage buffer. No index buffer, no bind group: lights are consumed in bulk by
 * whatever future simulation shader iterates the whole array, not drawn per-instance.
 *
 * Position/direction are deliberately not stored here at all (unlike the old per-manager
 * transform buffer this replaces): a light's worldPosition/direction are derived by that future
 * shader from `transforms[properties[i].transformIndex]`, the same shared array
 * sprites/raytraced objects use, so a light sharing its owner with e.g. a sprite shares one
 * transform update between them.
 */
export class LightResources {
    private array: PackedUniformArray<LightRecord>;
    private records = new Map<AnyLight, LightRecord>();

    private pointLights: PointLight[] = [];
    private spotlights: Spotlight[] = [];
    private laserLights: LaserLight[] = [];
    private directionalLights: DirectionalLight[] = [];
    private ambientLights: AmbientLight[] = [];

    constructor(device: GPUDevice) {
        this.array = new PackedUniformArray<LightRecord>(device, LIGHT_PROPERTIES_STRIDE_BYTES);
    }

    public getBuffer(): GPUBuffer {
        return this.array.getBuffer();
    }

    public onBufferReplaced(cb: () => void): void {
        this.array.onBufferReplaced(cb);
    }

    public getCount(): number {
        return this.array.getCount();
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

    /**
     * Full teardown-and-rebuild from `scene`. Called only on an actual scene load/swap (see
     * LitboxSceneRenderer.rebuildFromScene, its only caller) - never per-frame, and never for a
     * single object's create/destroy/property change, which go through addLight/removeLight/
     * refreshProperties instead.
     */
    public loadFromScene(scene: Scene, sceneGraph: SceneGraph, transformResources: TransformResources): void {
        for (const light of [...this.records.keys()]) {
            this.removeLight(light, transformResources);
        }

        this.pointLights = scene.pointLights;
        this.spotlights = scene.spotlights;
        this.laserLights = scene.laserLights;
        this.directionalLights = scene.directionalLights;
        this.ambientLights = scene.ambientLights;

        for (const light of this.pointLights) this.addLight('point', light, sceneGraph, transformResources);
        for (const light of this.spotlights) this.addLight('spot', light, sceneGraph, transformResources);
        for (const light of this.laserLights) this.addLight('laser', light, sceneGraph, transformResources);
        for (const light of this.directionalLights) this.addLight('directional', light, sceneGraph, transformResources);
        for (const light of this.ambientLights) this.addLight('ambient', light, sceneGraph, transformResources);
    }

    /**
     * Resolves and uploads a single newly-created light, appending it without touching any
     * existing light's entry - the targeted counterpart (for a structural create op) to
     * loadFromScene's full rebuild.
     */
    public addLight(kind: LightKind, light: AnyLight, sceneGraph: SceneGraph, transformResources: TransformResources): void {
        const transformEntry = transformResources.ensureEntry(light.ownerId, sceneGraph);
        const kindValue = LIGHT_KIND[kind];
        const propertiesEntry = this.array.insertStatic(
            (view, byteOffset) => writeProperties(view, byteOffset, light, kindValue, transformEntry.index),
        );
        this.records.set(light, { light, kind: kindValue, propertiesEntry });
    }

    /**
     * Removes exactly one light's entry (matched by reference) and releases its transform
     * reference. The targeted counterpart (for a destroyLight structural op) to
     * loadFromScene's full rebuild. No-op if `light` isn't tracked.
     */
    public removeLight(light: AnyLight, transformResources: TransformResources): void {
        const record = this.records.get(light);
        if (!record) {
            return;
        }
        this.array.remove(record.propertiesEntry);
        this.records.delete(light);
        transformResources.releaseEntry(light.ownerId);
    }

    /** Targeted re-upload of the properties for `light`. No-op if untracked. */
    public refreshProperties(light: AnyLight, transformResources: TransformResources): void {
        const record = this.records.get(light);
        if (!record) {
            return;
        }
        const transformEntry = transformResources.getEntry(light.ownerId);
        const transformIndex = transformEntry ? transformEntry.index : 0;
        this.array.writeEntry(
            record.propertiesEntry,
            (view, byteOffset) => writeProperties(view, byteOffset, light, record.kind, transformIndex),
        );
    }

    /** Moves `light`'s properties entry into the dynamic region. No-op if untracked, or if already dynamic. */
    public markDynamic(light: AnyLight): void {
        const record = this.records.get(light);
        if (!record) {
            return;
        }
        this.array.markDynamic(record.propertiesEntry);
    }

    public flush(): void {
        this.array.flush();
    }
}

function isSpotlight(light: AnyLight): light is Spotlight {
    return 'pinch' in light;
}

function writeProperties(view: DataView, byteOffset: number, light: AnyLight, kind: number, transformIndex: number): void {
    // color: vec4
    view.setFloat32(byteOffset + 0, light.color.r, true);
    view.setFloat32(byteOffset + 4, light.color.g, true);
    view.setFloat32(byteOffset + 8, light.color.b, true);
    view.setFloat32(byteOffset + 12, light.color.a, true);
    // transformIndex, kind, intensity, bounces, pinch
    view.setUint32(byteOffset + 16, transformIndex, true);
    view.setUint32(byteOffset + 20, kind, true);
    view.setFloat32(byteOffset + 24, light.intensity, true);
    view.setFloat32(byteOffset + 28, light.bounces, true);
    view.setFloat32(byteOffset + 32, isSpotlight(light) ? light.pinch : 0, true);
    // Bytes 36-47 are unused padding (WGSL rounds the struct up to a 16-byte multiple).
}
