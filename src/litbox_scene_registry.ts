import type { LitboxScene } from './litbox/litbox_scene.ts';
import { CornellSquareScene } from './litbox/scenes/cornell_square_scene.ts';

/** One selectable entry in the configuration page's scene dropdown (see main.ts). */
export interface SceneRegistryEntry {
    label: string;
    load: () => Promise<LitboxScene>;
}

/** Every scene selectable from the configuration page's scene dropdown - add a new scene here to make it selectable. */
export const SCENE_REGISTRY: Record<string, SceneRegistryEntry> = {
    'cornell-square': { label: 'Cornell Square', load: () => CornellSquareScene.load() },
};

export const DEFAULT_SCENE_KEY: keyof typeof SCENE_REGISTRY = 'cornell-square';
