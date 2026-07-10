import { parseScene, type AnyLight, type RaytracedObject, type Scene, type SceneObject, type SceneSprite } from './scene.ts';
import { DynamicSet } from './dynamic_set.ts';

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

    private nameIndex = new Map<string, SceneObject[]>();

    private transformFlags = new DynamicSet<SceneObject>();
    private lightFlags = new DynamicSet<AnyLight>();
    private spriteFlags = new DynamicSet<SceneSprite>();
    private raytracedFlags = new DynamicSet<RaytracedObject>();

    constructor(data: Scene) {
        this.data = data;
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
    }

    /**
     * Fetches and parses the JSON at the concrete subclass's `jsonPath`
     * (resolved against Vite's `BASE_URL`), constructing an instance of it.
     */
    public static async load<T extends LitboxScene>(
        this: { new (data: Scene): T; jsonPath: string },
    ): Promise<T> {
        const response = await fetch(`${import.meta.env.BASE_URL}${this.jsonPath}`);
        return new this(parseScene(await response.text()));
    }

    /**
     * Called once, immediately after the scene's JSON has been loaded and
     * parsed. Override to look up specific objects/sprites/lights from
     * `data` (e.g. via make*Dynamic) and stash references for use in onFrame().
     */
    public onLoad(): void {}

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

    // --- Renderer-facing plumbing. Internal use only.

    /** @internal Consumed once per frame by LitboxSceneRenderer. */
    public getDynamicFrameState(): {
        transforms: readonly SceneObject[];
        lights: readonly AnyLight[];
        sprites: readonly SceneSprite[];
        raytraced: readonly RaytracedObject[];
    } {
        return {
            transforms: this.transformFlags.activeThisFrame(),
            lights: this.lightFlags.activeThisFrame(),
            sprites: this.spriteFlags.activeThisFrame(),
            raytraced: this.raytracedFlags.activeThisFrame(),
        };
    }

    /** @internal Consumed once per frame by LitboxSceneRenderer, after it processes the frame state above. */
    public clearFrameDirtyFlags(): void {
        this.transformFlags.clearDirty();
        this.lightFlags.clearDirty();
        this.spriteFlags.clearDirty();
        this.raytracedFlags.clearDirty();
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
