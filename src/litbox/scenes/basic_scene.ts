import { LitboxScene } from '../litbox_scene.ts';
//import type { SceneObject } from '../scene.ts';

/** Scene-specific animation/interaction logic for cornell_square.json. */
export class BasicScene extends LitboxScene {
    public static readonly jsonPath = 'scenes/basic.json';

    //private rotatingSquare!: SceneObject;

    public override onLoad(): void {
        this.setActiveCamera('Main Camera');

        // this.rotatingSquare = this.makeTransformDynamic('Rotating Square');
        // this.rotatingSquare.active = true; // inactive by default in the exported scene
        // this.makeLightDynamic('Overhead Light');

        // // Rotation lives on the light's owning SceneObject, not on the light itself.
        // const overheadLightObject = this.getObject('Overhead Light');
        // this.addSlider('Overhead Light Rotation', -90, 90, 1,
        //     () => overheadLightObject.rotation,
        //     (value) => {
        //         overheadLightObject.rotation = value;
        //         this.markTransformDirty('Overhead Light');
        //     });
    }

    public override onFrame(deltaTimeSeconds: number): void {
        deltaTimeSeconds;
        //this.rotatingSquare.rotation += 90 * deltaTimeSeconds;
    }
}
