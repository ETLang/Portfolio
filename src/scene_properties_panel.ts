import type { LitboxScene } from './litbox/litbox_scene.ts';

/**
 * Expandable list of slider+textbox pairs, one per scene-specific tunable the active scene
 * registered via LitboxScene.addSlider (e.g. a light's rotation) - sibling to
 * denoiser_tunables_panel.ts's getDenoiserTunablesPanel, following the same pairing convention:
 * both controls in a row share the same `data-scene-slider-index` attribute (the slider's index
 * into `scene.getSliders()`) so main.ts's delegated listener can write the right one and keep the
 * paired control in sync. `scene` is null before a scene has finished loading.
 */
export function getScenePropertiesPanel(scene: LitboxScene | null): string {
    const sliders = scene?.getSliders() ?? [];
    if (sliders.length === 0) {
        return `
        <details class="scene-properties">
            <summary>Scene Properties</summary>
            <p class="scene-properties-empty">This scene has no scene-specific properties.</p>
        </details>`;
    }

    const rows = sliders.map(({ label, min, max, step, getValue }, index) => {
        const value = getValue();
        return `
            <div class="scene-param">
                <label for="scene-param-${index}-slider">${label}</label>
                <div class="scene-param-controls">
                    <input type="range" id="scene-param-${index}-slider" class="scene-param-slider" data-scene-slider-index="${index}"
                        min="${min}" max="${max}" step="${step}" value="${value}">
                    <input type="number" class="scene-param-number" data-scene-slider-index="${index}"
                        min="${min}" max="${max}" step="${step}" value="${value}">
                </div>
            </div>`;
    }).join('');

    return `
        <details class="scene-properties" open>
            <summary>Scene Properties</summary>
            <div class="scene-properties-list">${rows}
            </div>
        </details>`;
}
