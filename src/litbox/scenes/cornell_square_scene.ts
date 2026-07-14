import { LitboxScene } from '../litbox_scene.ts';
import type { AnyLight, SceneObject } from '../scene.ts';

/** Scene-specific animation/interaction logic for cornell_square.json. */
export class CornellSquareScene extends LitboxScene {
    public static readonly jsonPath = 'scenes/cornell_square.json';

    private rotatingSquare!: SceneObject;
    //private overheadLight!: AnyLight;

    public override onLoad(): void {
        this.rotatingSquare = this.makeTransformDynamic('Rotating Square');
        this.rotatingSquare.active = true; // inactive by default in the exported scene
        //this.overheadLight = this.makeLightDynamic('Overhead Light');
    }

    public override onFrame(deltaTimeSeconds: number): void {
        this.rotatingSquare.rotation += 90 * deltaTimeSeconds;
        //this.overheadLight.intensity = 1 + 0.5 * Math.sin(performance.now() / 500);
    }
}
