import {
    parseScene,
    type AmbientLight,
    type AnyLight,
    type Color,
    type DirectionalLight,
    type LaserLight,
    type LightKind,
    type PointLight,
    type RaytracedObject,
    type Scene,
    type SceneObject,
    type SceneSprite,
    type Spotlight,
    type Vector2,
} from './scene.ts';
import { DynamicSet } from './dynamic_set.ts';
import type { LitboxSceneRenderer } from '../litbox_scene_renderer.ts';

const ROOT_PARENT_ID = -1;

export interface CreateObjectOptions {
    name: string;
    /** Bare name or "/"-separated path (see resolvePath) of the new object's parent. Omit for a root object. */
    parent?: string;
    position?: Vector2;
    depth?: number;
    rotation?: number;
    scale?: Vector2;
    active?: boolean;
}

/** CreateObjectOptions plus the subset of SceneSprite fields worth overriding per call; the rest default to a plain, fully-opaque, unshaded white square. */
export interface CreateSpriteOptions extends CreateObjectOptions {
    layer?: number;
    sortOrder?: number;
    opacity?: number;
    image?: string;
    colorMod?: Color;
    ambient?: Color;
    emissive?: Color;
    simContribution?: Color;
    simBlur?: number;
    primitiveShape?: string;
}

/** CreateObjectOptions plus the subset of RaytracedObject fields worth overriding per call; the rest default to a plain, unshaded white primitive. */
export interface CreateRaytracedOptions extends CreateObjectOptions {
    logDensity?: number;
    roughness?: number;
    heightScale?: number;
    albedo?: Color;
    albedoMap?: string;
    logDensityMap?: string;
    sdfNormalMap?: string;
    primitiveShape?: string;
}

/** CreateObjectOptions plus the fields shared by every light kind; the rest default to a plain white light. */
export interface CreateLightOptions extends CreateObjectOptions {
    color?: Color;
    intensity?: number;
    bounces?: number;
}

/** CreateLightOptions plus the one field unique to spotlights. */
export interface CreateSpotlightOptions extends CreateLightOptions {
    pinch?: number;
}

/**
 * A pending structural change recorded by createObject/createSprite/createRaytraced/create<Light>/
 * destroyObject/destroySprite/destroyRaytraced/destroyLight/reparentObject, applied to the live
 * SceneGraph (and GPU resources) once per frame by LitboxSceneRenderer - the same two-phase split
 * as the dynamic/dirty transform flags below, since LitboxScene never touches SceneGraph directly.
 * `sprite`/`raytraced`/`light` are only present when the object was created via the matching
 * create* method, so the renderer knows to also upload that data alongside the new object.
 * `lightKind` accompanies `light` - which of the 5 light arrays it belongs to isn't recoverable
 * from the light object's own shape alone (point/laser/directional are structurally identical).
 */
type StructuralOp =
    | { type: 'create'; object: SceneObject; sprite?: SceneSprite; raytraced?: RaytracedObject; light?: AnyLight; lightKind?: LightKind }
    | { type: 'destroy'; rootId: number }
    | { type: 'destroySprite'; sprite: SceneSprite }
    | { type: 'destroyRaytraced'; raytraced: RaytracedObject }
    | { type: 'destroyLight'; light: AnyLight }
    | { type: 'reparent'; id: number; newParentId: number };

/**
 * Base class for scene-specific animation/interaction logic. Each JSON scene
 * (e.g. cornell_square) gets its own subclass wrapping the parsed `Scene`
 * data, so bespoke behavior that hooks into and drives that particular
 * content has somewhere to live rather than being bolted onto the renderer.
 *
 * Scene content is implicitly "static" (uploaded once, never revisited) unless
 * explicitly marked otherwise via the make*Dynamic/mark*Dirty methods below:
 * 'dynamic' entries are re-derived and re-uploaded every frame; 'dirty' entries
 * get the same treatment for exactly one frame, then revert to static. This
 * lets LitboxSceneRenderer skip re-uploading the (usually large majority of)
 * content that never changes after scene load.
 */
export abstract class LitboxScene {
    public readonly data: Scene;

    /**
     * The directory containing this scene's JSON file, relative to Vite's `BASE_URL` (e.g.
     * "scenes/" for a jsonPath of "scenes/cornell_square.json"). Texture paths in this scene's
     * data - both `textureAtlasKeys[].atlasName` and any un-atlassed image/map name - are
     * resolved relative to this directory, not to `BASE_URL` directly. Empty for scenes
     * constructed directly (e.g. in tests) rather than via `load()`.
     */
    public readonly baseUrl: string;

    private nameIndex = new Map<string, SceneObject[]>();

    private transformFlags = new DynamicSet<SceneObject>();
    private lightFlags = new DynamicSet<AnyLight>();
    private spriteFlags = new DynamicSet<SceneSprite>();
    private raytracedFlags = new DynamicSet<RaytracedObject>();

    private nextObjectId: number;
    private pendingStructuralOps: StructuralOp[] = [];

    constructor(data: Scene, baseUrl = '') {
        this.data = data;
        this.baseUrl = baseUrl;
        for (const obj of data.objects) {
            if (obj.name.includes('/')) {
                throw new Error(
                    `Litbox scene: object "${obj.name}" (id ${obj.id}) contains a "/" character, which is ` +
                    `reserved for make*/mark* path lookups (e.g. "Left Wall/Sprite"); rename it in the source scene.`,
                );
            }
            const matches = this.nameIndex.get(obj.name);
            if (matches) {
                matches.push(obj);
            } else {
                this.nameIndex.set(obj.name, [obj]);
            }
        }
        this.nextObjectId = Math.max(0, ...data.objects.map(o => o.id)) + 1;
    }

    /**
     * Fetches and parses the JSON at the concrete subclass's `jsonPath`
     * (resolved against Vite's `BASE_URL`), constructing an instance of it.
     */
    public static async load<T extends LitboxScene>(
        this: { new (data: Scene, baseUrl?: string): T; jsonPath: string },
    ): Promise<T> {
        const response = await fetch(`${import.meta.env.BASE_URL}${this.jsonPath}`);
        const baseUrl = this.jsonPath.slice(0, this.jsonPath.lastIndexOf('/') + 1);
        return new this(parseScene(await response.text()), baseUrl);
    }

    /**
     * Called once, when this scene is staged (or swapped in) via LitboxSceneRenderer.setScene,
     * immediately after its JSON has been loaded and parsed. Override to look up specific
     * objects/sprites/lights from `data` (e.g. via make*Dynamic) and stash references for use in
     * onFrame(), and/or to wire up interaction that needs the renderer itself - e.g. a canvas click
     * listener via renderer.getCanvas()/screenToWorld().
     */
    public onLoad(_renderer: LitboxSceneRenderer): void {}

    /**
     * Called at the start of every rendered frame, before the frame is
     * drawn. Override to drive animation or interaction state (e.g. mutate
     * positions/rotations on entries captured during onLoad()).
     */
    public onFrame(_deltaTimeSeconds: number): void {}

    // --- Scene-authoring API. `name` accepts either a bare unique SceneObject name, or
    // a "/"-separated path (e.g. "Left Wall/Sprite") that resolves each segment as a
    // unique direct child of the previous one - use a path when the bare name is
    // ambiguous (see resolvePath).

    /** Marks the named object's transform dynamic (re-derived/re-uploaded every frame) and returns its live struct. */
    public makeTransformDynamic(name: string): SceneObject {
        const obj = this.resolvePath(name);
        this.transformFlags.markDynamic(obj);
        return obj;
    }

    /** Marks the named object's Nth owned light (combined across all light kinds) dynamic and returns its live struct. */
    public makeLightDynamic(name: string, index = 0): AnyLight {
        const light = this.findLightByOwner(this.resolvePath(name), index);
        this.lightFlags.markDynamic(light);
        return light;
    }

    /** Marks the named object's Nth owned sprite dynamic and returns its live struct. */
    public makeSpriteDynamic(name: string, index = 0): SceneSprite {
        const sprite = this.findSpriteByOwner(this.resolvePath(name), index);
        this.spriteFlags.markDynamic(sprite);
        return sprite;
    }

    /** Marks the named object's Nth owned raytraced entry dynamic and returns its live struct. */
    public makeRayTracedDynamic(name: string, index = 0): RaytracedObject {
        const entry = this.findRaytracedByOwner(this.resolvePath(name), index);
        this.raytracedFlags.markDynamic(entry);
        return entry;
    }

    /** Marks the named object's transform dirty (re-derived/re-uploaded for exactly one frame). No-op if already dynamic. */
    public markTransformDirty(name: string): void {
        this.transformFlags.markDirty(this.resolvePath(name));
    }

    /** Marks the named object's Nth owned light dirty. No-op if already dynamic. */
    public markLightDirty(name: string, index = 0): void {
        this.lightFlags.markDirty(this.findLightByOwner(this.resolvePath(name), index));
    }

    /** Marks the named object's Nth owned sprite dirty. No-op if already dynamic. */
    public markSpriteDirty(name: string, index = 0): void {
        this.spriteFlags.markDirty(this.findSpriteByOwner(this.resolvePath(name), index));
    }

    /** Marks the named object's Nth owned raytraced entry dirty. No-op if already dynamic. */
    public markRayTracedDirty(name: string, index = 0): void {
        this.raytracedFlags.markDirty(this.findRaytracedByOwner(this.resolvePath(name), index));
    }

    // --- Structural authoring API. Mutates `data` immediately (so subsequent calls in the same
    // onFrame() see consistent state) and records a pending op for LitboxSceneRenderer to apply to
    // the live SceneGraph/GPU resources once per frame - LitboxScene never touches SceneGraph
    // directly, mirroring the dynamic/dirty split above.

    /** Creates a new SceneObject as a child of `options.parent` (or a root object if omitted) and returns it. */
    public createObject(options: CreateObjectOptions): SceneObject {
        const obj = this.buildObject(options);
        this.pendingStructuralOps.push({ type: 'create', object: obj });
        return obj;
    }

    /** Creates a new SceneObject (as createObject) with a single sprite attached to it, and returns the object. */
    public createSprite(options: CreateSpriteOptions): SceneObject {
        const obj = this.buildObject(options);
        const sprite: SceneSprite = {
            ownerId: obj.id,
            layer: options.layer ?? 0,
            sortOrder: options.sortOrder ?? 0,
            opacity: options.opacity ?? 1,
            image: options.image ?? '',
            colorMod: options.colorMod ?? { r: 1, g: 1, b: 1, a: 1 },
            ambient: options.ambient ?? { r: 1, g: 1, b: 1, a: 1 },
            emissive: options.emissive ?? { r: 0, g: 0, b: 0, a: 1 },
            simContribution: options.simContribution ?? { r: 0, g: 0, b: 0, a: 0 },
            simBlur: options.simBlur ?? 0,
            primitiveShape: options.primitiveShape ?? 'rect',
        };

        this.data.sprites.push(sprite);
        this.pendingStructuralOps.push({ type: 'create', object: obj, sprite });
        return obj;
    }

    /** Creates a new SceneObject (as createObject) with a single raytraced entry attached to it, and returns the object. */
    public createRaytraced(options: CreateRaytracedOptions): SceneObject {
        const obj = this.buildObject(options);
        const raytraced: RaytracedObject = {
            ownerId: obj.id,
            logDensity: options.logDensity ?? 0,
            roughness: options.roughness ?? 0.5,
            heightScale: options.heightScale ?? 1,
            albedo: options.albedo ?? { r: 1, g: 1, b: 1, a: 1 },
            albedoMap: options.albedoMap ?? '',
            logDensityMap: options.logDensityMap ?? '',
            sdfNormalMap: options.sdfNormalMap ?? '',
            primitiveShape: options.primitiveShape ?? 'rect',
        };

        this.data.raytraced.push(raytraced);
        this.pendingStructuralOps.push({ type: 'create', object: obj, raytraced });
        return obj;
    }

    /** Creates a new SceneObject (as createObject) with a single point light attached to it, and returns the object. */
    public createPointLight(options: CreateLightOptions): SceneObject {
        const obj = this.buildObject(options);
        const light: PointLight = this.lightDefaults(obj, options);
        this.data.pointLights.push(light);
        this.pendingStructuralOps.push({ type: 'create', object: obj, light, lightKind: 'point' });
        return obj;
    }

    /** Creates a new SceneObject (as createObject) with a single spotlight attached to it, and returns the object. */
    public createSpotlight(options: CreateSpotlightOptions): SceneObject {
        const obj = this.buildObject(options);
        const light: Spotlight = { ...this.lightDefaults(obj, options), pinch: options.pinch ?? 0.5 };
        this.data.spotlights.push(light);
        this.pendingStructuralOps.push({ type: 'create', object: obj, light, lightKind: 'spot' });
        return obj;
    }

    /** Creates a new SceneObject (as createObject) with a single laser light attached to it, and returns the object. */
    public createLaserLight(options: CreateLightOptions): SceneObject {
        const obj = this.buildObject(options);
        const light: LaserLight = this.lightDefaults(obj, options);
        this.data.laserLights.push(light);
        this.pendingStructuralOps.push({ type: 'create', object: obj, light, lightKind: 'laser' });
        return obj;
    }

    /** Creates a new SceneObject (as createObject) with a single directional light attached to it, and returns the object. */
    public createDirectionalLight(options: CreateLightOptions): SceneObject {
        const obj = this.buildObject(options);
        const light: DirectionalLight = this.lightDefaults(obj, options);
        this.data.directionalLights.push(light);
        this.pendingStructuralOps.push({ type: 'create', object: obj, light, lightKind: 'directional' });
        return obj;
    }

    /** Creates a new SceneObject (as createObject) with a single ambient light attached to it, and returns the object. */
    public createAmbientLight(options: CreateLightOptions): SceneObject {
        const obj = this.buildObject(options);
        const light: AmbientLight = this.lightDefaults(obj, options);
        this.data.ambientLights.push(light);
        this.pendingStructuralOps.push({ type: 'create', object: obj, light, lightKind: 'ambient' });
        return obj;
    }

    /** Constructs and registers (in `data.objects` and `nameIndex`) a new SceneObject; shared by createObject/createSprite/createRaytraced/create<Light>. */
    private buildObject(options: CreateObjectOptions): SceneObject {
        if (options.name.includes('/')) {
            throw new Error(
                `Litbox scene: object name "${options.name}" contains a "/" character, which is ` +
                `reserved for make*/mark* path lookups (e.g. "Left Wall/Sprite"); choose another name.`,
            );
        }
        const parentId = options.parent ? this.resolvePath(options.parent).id : ROOT_PARENT_ID;
        const obj: SceneObject = {
            active: options.active ?? true,
            id: this.nextObjectId++,
            name: options.name,
            parentId,
            position: options.position ?? { x: 0, y: 0 },
            depth: options.depth ?? 0,
            rotation: options.rotation ?? 0,
            scale: options.scale ?? { x: 1, y: 1 },
        };

        this.data.objects.push(obj);
        const matches = this.nameIndex.get(obj.name);
        if (matches) {
            matches.push(obj);
        } else {
            this.nameIndex.set(obj.name, [obj]);
        }

        return obj;
    }

    /** Builds the fields shared by every light kind, owned by `obj`; shared by createPointLight/createSpotlight/createLaserLight/createDirectionalLight/createAmbientLight. */
    private lightDefaults(obj: SceneObject, options: CreateLightOptions): { ownerId: number; color: Color; intensity: number; bounces: number } {
        return {
            ownerId: obj.id,
            color: options.color ?? { r: 1, g: 1, b: 1, a: 1 },
            intensity: options.intensity ?? 1,
            bounces: options.bounces ?? 1,
        };
    }

    /**
     * Destroys the named object and its whole subtree: removes them - and everything they own
     * (sprites, lights, raytraced entries) - from `data`, and drops any dynamic/dirty flags on
     * them. Throws without mutating anything if the cascade would remove a camera or simulation
     * owner; there's no supported way to recover the renderer's cached camera/simulation state
     * from that.
     */
    public destroyObject(name: string): void {
        const root = this.resolvePath(name);
        const cascade = this.collectDescendantIds(root.id);
        cascade.unshift(root.id);
        const cascadeIds = new Set(cascade);

        if (this.data.cameras.some(c => cascadeIds.has(c.ownerId)) || this.data.simulations.some(s => cascadeIds.has(s.ownerId))) {
            throw new Error(
                `Litbox scene: cannot destroy object "${root.name}" (id ${root.id}) - its subtree owns a ` +
                `camera or simulation, which isn't supported.`,
            );
        }

        const removedObjects = this.data.objects.filter(o => cascadeIds.has(o.id));
        this.data.objects = this.data.objects.filter(o => !cascadeIds.has(o.id));
        for (const obj of removedObjects) {
            this.transformFlags.delete(obj);
            const matches = this.nameIndex.get(obj.name);
            if (!matches) {
                continue;
            }
            const index = matches.indexOf(obj);
            if (index !== -1) {
                matches.splice(index, 1);
            }
            if (matches.length === 0) {
                this.nameIndex.delete(obj.name);
            }
        }

        const deleteLightFlag = (light: AnyLight): void => this.lightFlags.delete(light);
        this.data.pointLights = this.filterOwned(this.data.pointLights, cascadeIds, deleteLightFlag);
        this.data.spotlights = this.filterOwned(this.data.spotlights, cascadeIds, deleteLightFlag);
        this.data.laserLights = this.filterOwned(this.data.laserLights, cascadeIds, deleteLightFlag);
        this.data.directionalLights = this.filterOwned(this.data.directionalLights, cascadeIds, deleteLightFlag);
        this.data.ambientLights = this.filterOwned(this.data.ambientLights, cascadeIds, deleteLightFlag);
        this.data.sprites = this.filterOwned(this.data.sprites, cascadeIds, s => this.spriteFlags.delete(s));
        this.data.raytraced = this.filterOwned(this.data.raytraced, cascadeIds, r => this.raytracedFlags.delete(r));

        this.pendingStructuralOps.push({ type: 'destroy', rootId: root.id });
    }

    /** Destroys the named object's Nth owned sprite (index across just sprites, as findSpriteByOwner). The object itself and any other data it owns are untouched. */
    public destroySprite(name: string, index = 0): void {
        const sprite = this.findSpriteByOwner(this.resolvePath(name), index);
        this.data.sprites = this.data.sprites.filter(s => s !== sprite);
        this.spriteFlags.delete(sprite);
        this.pendingStructuralOps.push({ type: 'destroySprite', sprite });
    }

    /** Destroys the named object's Nth owned raytraced entry (index across just raytraced entries, as findRaytracedByOwner). The object itself and any other data it owns are untouched. */
    public destroyRaytraced(name: string, index = 0): void {
        const entry = this.findRaytracedByOwner(this.resolvePath(name), index);
        this.data.raytraced = this.data.raytraced.filter(r => r !== entry);
        this.raytracedFlags.delete(entry);
        this.pendingStructuralOps.push({ type: 'destroyRaytraced', raytraced: entry });
    }

    /** Destroys the named object's Nth owned light (combined across all light kinds, as findLightByOwner). The object itself and any other data it owns are untouched. */
    public destroyLight(name: string, index = 0): void {
        const light = this.findLightByOwner(this.resolvePath(name), index);
        this.data.pointLights = this.data.pointLights.filter(l => l !== light);
        this.data.spotlights = this.data.spotlights.filter(l => l !== light);
        this.data.laserLights = this.data.laserLights.filter(l => l !== light);
        this.data.directionalLights = this.data.directionalLights.filter(l => l !== light);
        this.data.ambientLights = this.data.ambientLights.filter(l => l !== light);
        this.lightFlags.delete(light);
        this.pendingStructuralOps.push({ type: 'destroyLight', light });
    }

    /**
     * Moves the named object (and its whole subtree) to a new parent, or to the scene root if
     * `newParent` is null. Throws without mutating anything on a self- or descendant-cycle, so a
     * bad call fails fast inside the caller's onFrame() rather than surfacing later inside the
     * renderer's frame loop.
     */
    public reparentObject(name: string, newParent: string | null): void {
        const obj = this.resolvePath(name);
        const newParentId = newParent ? this.resolvePath(newParent).id : ROOT_PARENT_ID;

        if (newParentId === obj.id) {
            throw new Error(`Litbox scene: cannot reparent object "${obj.name}" (id ${obj.id}) to itself.`);
        }
        if (newParentId !== ROOT_PARENT_ID && this.collectDescendantIds(obj.id).includes(newParentId)) {
            throw new Error(
                `Litbox scene: cannot reparent object "${obj.name}" (id ${obj.id}) to its own descendant (id ${newParentId}).`,
            );
        }

        obj.parentId = newParentId;
        this.pendingStructuralOps.push({ type: 'reparent', id: obj.id, newParentId });
    }

    // --- Renderer-facing plumbing. Internal use only.

    /** @internal Consumed once per frame by LitboxSceneRenderer, applied to the live SceneGraph/GPU resources. */
    public getPendingStructuralOps(): readonly StructuralOp[] {
        return this.pendingStructuralOps;
    }

    /** @internal Consumed once per frame by LitboxSceneRenderer, after it applies the ops above. */
    public clearPendingStructuralOps(): void {
        this.pendingStructuralOps = [];
    }

    /**
     * @internal Consumed once per frame by LitboxSceneRenderer. `transforms`/`lights`/
     * `sprites`/`raytraced` are the full dynamic-∪-dirty set (what needs a GPU data refresh
     * this frame); `persistent*` is the 'dynamic'-only subset (what should be moved into a
     * packed array's dynamic region, if not already there - a one-shot 'dirty' entry gets its
     * data refreshed but is never repositioned).
     */
    public getDynamicFrameState(): {
        transforms: readonly SceneObject[];
        lights: readonly AnyLight[];
        sprites: readonly SceneSprite[];
        raytraced: readonly RaytracedObject[];
        persistentTransforms: readonly SceneObject[];
        persistentLights: readonly AnyLight[];
        persistentSprites: readonly SceneSprite[];
        persistentRaytraced: readonly RaytracedObject[];
    } {
        return {
            transforms: this.transformFlags.activeThisFrame(),
            lights: this.lightFlags.activeThisFrame(),
            sprites: this.spriteFlags.activeThisFrame(),
            raytraced: this.raytracedFlags.activeThisFrame(),
            persistentTransforms: this.transformFlags.dynamicOnly(),
            persistentLights: this.lightFlags.dynamicOnly(),
            persistentSprites: this.spriteFlags.dynamicOnly(),
            persistentRaytraced: this.raytracedFlags.dynamicOnly(),
        };
    }

    /** @internal Consumed once per frame by LitboxSceneRenderer, after it processes the frame state above. */
    public clearFrameDirtyFlags(): void {
        this.transformFlags.clearDirty();
        this.lightFlags.clearDirty();
        this.spriteFlags.clearDirty();
        this.raytracedFlags.clearDirty();
    }

    // --- Structural authoring helpers.

    /** Strict descendants of `id` (excludes `id` itself), depth-first, cycle-guarded - mirrors SceneGraph.getDescendantIds. */
    private collectDescendantIds(id: number, result: number[] = [], visiting: Set<number> = new Set()): number[] {
        if (visiting.has(id)) {
            console.warn(`Litbox scene: cycle detected involving object id ${id} while collecting descendants.`);
            return result;
        }
        visiting.add(id);
        for (const obj of this.data.objects) {
            if (obj.parentId === id) {
                result.push(obj.id);
                this.collectDescendantIds(obj.id, result, visiting);
            }
        }
        visiting.delete(id);
        return result;
    }

    /** Splits `items` by ownerId membership in `removedIds`, calling `onRemove` for each dropped entry and returning the survivors. */
    private filterOwned<T extends { ownerId: number }>(items: T[], removedIds: Set<number>, onRemove: (item: T) => void): T[] {
        const kept: T[] = [];
        for (const item of items) {
            if (removedIds.has(item.ownerId)) {
                onRemove(item);
            } else {
                kept.push(item);
            }
        }
        return kept;
    }

    // --- Name/path resolution.

    /**
     * Resolves a bare unique SceneObject name, or a "/"-separated path where each
     * subsequent segment is a unique direct child (by parentId) of the previous
     * segment's object. Throws on a zero- or multiple-match segment, since
     * SceneObject.name is not unique in real exported scenes (many siblings share
     * generic names like "Sprite"/"Traced") - callers must disambiguate via a path.
     */
    private resolvePath(path: string): SceneObject {
        const segments = path.split('/');
        let current = this.resolveByName(segments[0]);
        for (let i = 1; i < segments.length; i++) {
            current = this.resolveChildByName(current, segments[i]);
        }
        return current;
    }

    private resolveByName(name: string): SceneObject {
        const matches = this.nameIndex.get(name) ?? [];
        if (matches.length === 0) {
            throw new Error(`Litbox scene: no SceneObject named "${name}" found.`);
        }
        if (matches.length > 1) {
            throw new Error(
                `Litbox scene: ${matches.length} SceneObjects named "${name}" found (ids: ` +
                `${matches.map(o => o.id).join(', ')}); disambiguate with a "Parent/${name}" path.`,
            );
        }
        return matches[0];
    }

    private resolveChildByName(parent: SceneObject, name: string): SceneObject {
        const matches = this.data.objects.filter(o => o.parentId === parent.id && o.name === name);
        if (matches.length === 0) {
            throw new Error(`Litbox scene: object "${parent.name}" (id ${parent.id}) has no child named "${name}".`);
        }
        if (matches.length > 1) {
            throw new Error(
                `Litbox scene: object "${parent.name}" (id ${parent.id}) has ${matches.length} children named ` +
                `"${name}" (ids: ${matches.map(o => o.id).join(', ')}); ambiguous path segment.`,
            );
        }
        return matches[0];
    }

    private findLightByOwner(owner: SceneObject, index: number): AnyLight {
        const combined: AnyLight[] = [
            ...this.data.pointLights,
            ...this.data.spotlights,
            ...this.data.laserLights,
            ...this.data.directionalLights,
            ...this.data.ambientLights,
        ].filter(l => l.ownerId === owner.id);
        const light = combined[index];
        if (!light) {
            throw new Error(
                `Litbox scene: object "${owner.name}" (id ${owner.id}) has no light at combined index ${index} ` +
                `(found ${combined.length} across all kinds).`,
            );
        }
        return light;
    }

    private findSpriteByOwner(owner: SceneObject, index: number): SceneSprite {
        const matches = this.data.sprites.filter(s => s.ownerId === owner.id);
        const sprite = matches[index];
        if (!sprite) {
            throw new Error(
                `Litbox scene: object "${owner.name}" (id ${owner.id}) has no sprite at index ${index} ` +
                `(found ${matches.length}).`,
            );
        }
        return sprite;
    }

    private findRaytracedByOwner(owner: SceneObject, index: number): RaytracedObject {
        const matches = this.data.raytraced.filter(r => r.ownerId === owner.id);
        const entry = matches[index];
        if (!entry) {
            throw new Error(
                `Litbox scene: object "${owner.name}" (id ${owner.id}) has no raytraced entry at index ${index} ` +
                `(found ${matches.length}).`,
            );
        }
        return entry;
    }
}
