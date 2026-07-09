import { parseScene, type Scene } from './scene.ts';

/**
 * Base class for scene-specific animation/interaction logic. Each JSON scene
 * (e.g. cornell_square) gets its own subclass wrapping the parsed `Scene`
 * data, so bespoke behavior that hooks into and drives that particular
 * content has somewhere to live rather than being bolted onto the renderer.
 */
export abstract class LitboxScene {
    public readonly data: Scene;

    constructor(data: Scene) {
        this.data = data;
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
     * `data` and stash references for use in onFrame().
     */
    public onLoad(): void {}

    /**
     * Called at the start of every rendered frame, before the frame is
     * drawn. Override to drive animation or interaction state (e.g. mutate
     * positions/rotations on entries captured during onLoad()).
     */
    public onFrame(_deltaTimeSeconds: number): void {}
}
