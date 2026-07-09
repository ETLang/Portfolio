import { LitboxScene } from '../litbox_scene.ts';

/** Scene-specific animation/interaction logic for cornell_square.json. */
export class CornellSquareScene extends LitboxScene {
    public static readonly jsonPath = 'scenes/cornell_square.json';

    public override onLoad(): void {
        // TODO: look up specific objects/sprites/lights from this.data and
        // stash references here for use in onFrame().
    }

    public override onFrame(_deltaTimeSeconds: number): void {
        // TODO: drive this scene's animation/interaction state each frame.
    }
}
