import { LitboxScene } from '../litbox_scene.ts';
import type { SceneObject } from '../scene.ts';
import type { DenoiserTunables } from '../simulation.ts';

/**
 * See LitboxScene.getDenoiserTunables's doc comment. densityBlurFalloff raised well above the
 * default - the rotating square here reads as a cloudy volume dominated by subsurface scattering
 * rather than a solid surface (contrast BattleScene's pinned default), so blur suppression should
 * ramp in much more gradually with optical depth instead of clamping down fast.
 */
const DENOISER_TUNABLES: Partial<DenoiserTunables> = {
    densityBlurFalloff: 0.1,
};

/** Scene-specific animation/interaction logic for cornell_square.json. */
export class CornellSquareScene extends LitboxScene {
    public static readonly jsonPath = 'scenes/cornell_square.json';

    private rotatingSquare!: SceneObject;

    public override onLoad(): void {
        this.rotatingSquare = this.makeTransformDynamic('Rotating Square');
        this.rotatingSquare.active = true; // inactive by default in the exported scene
        this.makeLightDynamic('Overhead Light');

        // Rotation lives on the light's owning SceneObject, not on the light itself.
        const overheadLightObject = this.getObject('Overhead Light');
        this.addSlider('Overhead Light Rotation', -90, 90, 1,
            () => overheadLightObject.rotation,
            (value) => {
                overheadLightObject.rotation = value;
                this.markTransformDirty('Overhead Light');
            });
    }

    public override onFrame(deltaTimeSeconds: number): void {
        this.rotatingSquare.rotation += 90 * deltaTimeSeconds;
    }

    public override getDenoiserTunables(): Partial<DenoiserTunables> {
        return DENOISER_TUNABLES;
    }
}
