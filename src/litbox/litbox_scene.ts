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
    type SceneCamera,
    type SceneObject,
    type SceneSprite,
    type Spotlight,
    type Vector2,
} from './scene.ts';
import { DynamicSet } from './dynamic_set.ts';
import type { LitboxSceneRenderer } from '../litbox_scene_renderer.ts';
import type { DenoiserTunables } from './simulation.ts';

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
    sortOrder?: number;
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

/** The subset of CreateObjectOptions that makes sense to override on a cloneObject() call - `name` is inherited from the template, not chosen. */
export interface CloneObjectOptions {
    parent?: string;
    position?: Vector2;
    depth?: number;
    rotation?: number;
    scale?: Vector2;
    active?: boolean;
}

/** CreateLightOptions plus the one field unique to spotlights. */
export interface CreateSpotlightOptions extends CreateLightOptions {
    pinch?: number;
}

/**
 * A single scene-specific tunable (e.g. a light's rotation, an object's rotation, a light's
 * intensity) that a LitboxScene subclass exposes to the configuration UI as a slider - see
 * LitboxScene.addSlider. `getValue`/`setValue` close over whatever live struct(s) the subclass
 * captured in onLoad(), so the UI never needs to know the scene's internal shape.
 */
export interface SceneSlider {
    label: string;
    min: number;
    max: number;
    step: number;
    getValue: () => number;
    setValue: (value: number) => void;
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

    /** Scene-specific sliders registered via addSlider(), in registration order - see getSliders(). */
    private sliders: SceneSlider[] = [];

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

    /**
     * Per-scene defaults for DenoiserTunables (see its own doc comment), applied on top of
     * DEFAULT_DENOISER_TUNABLES every time this scene loads (see
     * SimulationResources.loadFromScene) - the ideal denoiser "look" varies enough scene-to-scene
     * that one global default isn't a good fit for all of them, and chasing an automated way to
     * detect which situation applies was judged out of scope. Returns null (no override -
     * DEFAULT_DENOISER_TUNABLES applies as-is) unless a subclass overrides it. Deliberately a
     * fixed baseline, not a per-scene delta preserved across switches: whatever the user tweaked
     * via the tunables panel for the previously active scene is intentionally discarded, not
     * carried over.
     */
    public getDenoiserTunables(): Partial<DenoiserTunables> | null {
        return null;
    }

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

    /**
     * Marks the named (or directly referenced) object's transform dirty (re-derived/re-uploaded
     * for exactly one frame). No-op if already dynamic. Pass a direct SceneObject reference
     * (e.g. one returned by cloneObject) instead of a name/path once more than one object shares
     * that name, since resolvePath can no longer find it unambiguously.
     */
    public markTransformDirty(target: string | SceneObject): void {
        this.transformFlags.markDirty(typeof target === 'string' ? this.resolvePath(target) : target);
    }

    /** Marks the named (or directly referenced) object's Nth owned light dirty - see markTransformDirty for the reference form. No-op if already dynamic. */
    public markLightDirty(target: string | AnyLight, index = 0): void {
        this.lightFlags.markDirty(typeof target === 'string' ? this.findLightByOwner(this.resolvePath(target), index) : target);
    }

    /** Marks the named (or directly referenced) object's Nth owned sprite dirty - see markTransformDirty for the reference form. No-op if already dynamic. */
    public markSpriteDirty(target: string | SceneSprite, index = 0): void {
        this.spriteFlags.markDirty(typeof target === 'string' ? this.findSpriteByOwner(this.resolvePath(target), index) : target);
    }

    /** Marks the named (or directly referenced) object's Nth owned raytraced entry dirty - see markTransformDirty for the reference form. No-op if already dynamic. */
    public markRayTracedDirty(target: string | RaytracedObject, index = 0): void {
        this.raytracedFlags.markDirty(typeof target === 'string' ? this.findRaytracedByOwner(this.resolvePath(target), index) : target);
    }

    // --- Configuration UI API. Lets a subclass expose bespoke, scene-specific tunables (a
    // light's rotation, an object's rotation, a light's intensity, etc.) as sliders in the
    // configuration page's scene-specific panel, alongside the fixed Litbox/denoiser panels
    // (see scene_properties_panel.ts for the sibling pattern to denoiser_tunables_panel.ts).

    /**
     * Returns the named object's live struct without marking its transform dynamic/dirty -
     * unlike makeTransformDynamic, this doesn't opt it into continuous per-frame GPU re-upload.
     * Intended for onLoad() to capture a struct for a slider's getValue/setValue (see addSlider);
     * pair a setValue that mutates the returned struct with a markTransformDirty() call so the
     * change actually reaches the GPU.
     */
    public getObject(name: string): SceneObject {
        return this.resolvePath(name);
    }

    /**
     * Resolves a "/"-separated path of direct-child names, rooted at a specific object reference
     * rather than a globally unique name - use this instead of getObject's name-only path once
     * more than one object shares a name (e.g. several cloneObject() instances of the same
     * template, which all keep the template's object names).
     */
    public resolveRelativePath(root: SceneObject, path: string): SceneObject {
        let current = root;
        for (const segment of path.split('/')) {
            current = this.resolveChildByName(current, segment);
        }
        return current;
    }

    /** Returns `owner`'s Nth owned sprite, without marking it dynamic/dirty - see resolveRelativePath for why a reference (not a name) is needed for a cloneObject() instance. */
    public getSprite(owner: SceneObject, index = 0): SceneSprite {
        return this.findSpriteByOwner(owner, index);
    }

    /** Returns `owner`'s Nth owned raytraced entry, without marking it dynamic/dirty - see getSprite. */
    public getRaytraced(owner: SceneObject, index = 0): RaytracedObject {
        return this.findRaytracedByOwner(owner, index);
    }

    /** Returns `owner`'s Nth owned light (combined across all kinds), without marking it dynamic/dirty - see getSprite. */
    public getLight(owner: SceneObject, index = 0): AnyLight {
        return this.findLightByOwner(owner, index);
    }

    /**
     * Makes the named object's camera the one LitboxSceneRenderer.getActiveCamera picks, by
     * moving it to the front of `data.cameras` (that method takes the first camera whose owner
     * is active in the hierarchy). Needed because a Unity-exported scene's camera order reflects
     * whatever order Unity happened to serialize its Camera components in, not which one is meant
     * to be the "real" one - e.g. a UI-overlay camera (a perf display's Canvas) can easily end up
     * ordered before the actual scene camera. Call from onLoad(); every scene exported from Unity
     * names its main camera "Main Camera", so `setActiveCamera('Main Camera')` is the expected
     * call for most scenes.
     */
    public setActiveCamera(name: string): void {
        const obj = this.resolvePath(name);
        const camera = this.data.cameras.find(c => c.ownerId === obj.id);
        if (!camera) {
            throw new Error(`Litbox scene: object "${obj.name}" (id ${obj.id}) owns no camera.`);
        }
        this.data.cameras = [camera, ...this.data.cameras.filter((c: SceneCamera) => c !== camera)];
    }

    /**
     * Registers a scene-specific tunable to appear as a slider in the configuration UI's
     * scene-specific panel. Call from onLoad(), after capturing whatever live struct(s)
     * getValue/setValue close over (e.g. via getObject/makeLightDynamic). `setValue` is
     * responsible for calling the appropriate markTransformDirty/markLightDirty/etc. itself, if
     * the mutated struct isn't already dynamic.
     */
    protected addSlider(label: string, min: number, max: number, step: number, getValue: () => number, setValue: (value: number) => void): void {
        this.sliders.push({ label, min, max, step, getValue, setValue });
    }

    /** @internal Consumed by scene_properties_panel.ts to render this scene's slider panel - see addSlider. */
    public getSliders(): readonly SceneSlider[] {
        return this.sliders;
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
            sortOrder: options.sortOrder ?? 0,
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

    /**
     * Deep-clones `target`'s whole subtree (the object plus every descendant, and anything each
     * one owns - a sprite, a raytraced entry, a light) as a fresh set of objects, and returns the
     * cloned root. Every cloned object below the root keeps the source's relative transform
     * verbatim; `options` overrides just the root's own transform/parent/active fields (e.g. to
     * place a newly spawned instance), mirroring createObject's options shape.
     *
     * `target` accepts a name/path (as elsewhere) or a direct SceneObject reference - a reference
     * is required once more than one clone of the same template exists, since the template's own
     * name is no longer unique enough for resolvePath to find it unambiguously.
     */
    public cloneObject(target: string | SceneObject, options: CloneObjectOptions = {}): SceneObject {
        const template = typeof target === 'string' ? this.resolvePath(target) : target;
        const sourceIds = [template.id, ...this.collectDescendantIds(template.id)];
        const idMap = new Map<number, number>();
        const clones: SceneObject[] = [];

        for (const sourceId of sourceIds) {
            const source = this.data.objects.find(o => o.id === sourceId)!;
            const isRoot = sourceId === template.id;
            const parentId = isRoot
                ? (options.parent ? this.resolvePath(options.parent).id : source.parentId)
                : idMap.get(source.parentId)!;
            const clone: SceneObject = {
                active: isRoot ? (options.active ?? source.active) : source.active,
                id: this.nextObjectId++,
                name: source.name,
                parentId,
                position: isRoot ? (options.position ?? { ...source.position }) : { ...source.position },
                depth: isRoot ? (options.depth ?? source.depth) : source.depth,
                rotation: isRoot ? (options.rotation ?? source.rotation) : source.rotation,
                scale: isRoot ? (options.scale ?? { ...source.scale }) : { ...source.scale },
            };
            idMap.set(sourceId, clone.id);
            this.data.objects.push(clone);
            const matches = this.nameIndex.get(clone.name);
            if (matches) {
                matches.push(clone);
            } else {
                this.nameIndex.set(clone.name, [clone]);
            }
            clones.push(clone);
        }

        for (let i = 0; i < sourceIds.length; i++) {
            const sourceId = sourceIds[i];
            const clone = clones[i];

            const sourceSprite = this.data.sprites.find(s => s.ownerId === sourceId);
            const clonedSprite = sourceSprite ? this.cloneSprite(sourceSprite, clone.id) : undefined;
            if (clonedSprite) {
                this.data.sprites.push(clonedSprite);
            }

            const sourceRaytraced = this.data.raytraced.find(r => r.ownerId === sourceId);
            const clonedRaytraced = sourceRaytraced ? this.cloneRaytraced(sourceRaytraced, clone.id) : undefined;
            if (clonedRaytraced) {
                this.data.raytraced.push(clonedRaytraced);
            }

            const sourceLight = this.findLightWithKindByOwnerId(sourceId);
            let clonedLight: AnyLight | undefined;
            let lightKind: LightKind | undefined;
            if (sourceLight) {
                clonedLight = { ...sourceLight.light, ownerId: clone.id, color: { ...sourceLight.light.color } };
                lightKind = sourceLight.kind;
                switch (lightKind) {
                    case 'point': this.data.pointLights.push(clonedLight as PointLight); break;
                    case 'spot': this.data.spotlights.push(clonedLight as Spotlight); break;
                    case 'laser': this.data.laserLights.push(clonedLight as LaserLight); break;
                    case 'directional': this.data.directionalLights.push(clonedLight as DirectionalLight); break;
                    case 'ambient': this.data.ambientLights.push(clonedLight as AmbientLight); break;
                }
            }

            this.pendingStructuralOps.push({
                type: 'create',
                object: clone,
                sprite: clonedSprite,
                raytraced: clonedRaytraced,
                light: clonedLight,
                lightKind,
            });
        }

        return clones[0];
    }

    /** Deep-copies a SceneSprite's own Color fields (never shares them with the source) for a new owner - shared by cloneObject. */
    private cloneSprite(source: SceneSprite, ownerId: number): SceneSprite {
        return {
            ...source,
            ownerId,
            colorMod: { ...source.colorMod },
            ambient: { ...source.ambient },
            emissive: { ...source.emissive },
            simContribution: { ...source.simContribution },
        };
    }

    /** Deep-copies a RaytracedObject's own Color field for a new owner - shared by cloneObject. */
    private cloneRaytraced(source: RaytracedObject, ownerId: number): RaytracedObject {
        return { ...source, ownerId, albedo: { ...source.albedo } };
    }

    /** Finds whichever of the 5 light arrays owns `ownerId`, and which kind it is - shared by cloneObject (StructuralOp needs the kind alongside the light itself). */
    private findLightWithKindByOwnerId(ownerId: number): { light: AnyLight; kind: LightKind } | undefined {
        const point = this.data.pointLights.find(l => l.ownerId === ownerId);
        if (point) return { light: point, kind: 'point' };
        const spot = this.data.spotlights.find(l => l.ownerId === ownerId);
        if (spot) return { light: spot, kind: 'spot' };
        const laser = this.data.laserLights.find(l => l.ownerId === ownerId);
        if (laser) return { light: laser, kind: 'laser' };
        const directional = this.data.directionalLights.find(l => l.ownerId === ownerId);
        if (directional) return { light: directional, kind: 'directional' };
        const ambient = this.data.ambientLights.find(l => l.ownerId === ownerId);
        if (ambient) return { light: ambient, kind: 'ambient' };
        return undefined;
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
     * from that. `target` accepts a name/path or a direct SceneObject reference - see
     * markTransformDirty for why a reference is needed once more than one object shares a name.
     */
    public destroyObject(target: string | SceneObject): void {
        const root = typeof target === 'string' ? this.resolvePath(target) : target;
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
