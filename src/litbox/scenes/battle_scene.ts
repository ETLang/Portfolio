import { LitboxScene } from '../litbox_scene.ts';

/** Scene-specific animation/interaction logic for battle.json. */
export class BattleScene extends LitboxScene {
    public static readonly jsonPath = 'scenes/battle.json';

    public override onLoad(): void {
        this.setActiveCamera('Main Camera');
    }
}
