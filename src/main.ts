import './style.css';
import { marked } from 'marked';
import { ModalDialog } from './modal-dialog.ts';
import { LitboxSceneRenderer, type WebGpuInitFailureReason } from './litbox_scene_renderer.ts';
import { getAboutPageContent } from './about.ts';
import { getContactForm } from './contact-form.ts';
import { formatRate } from './litbox/performance_metrics.ts';
import { getDenoiserTunablesPanel } from './denoiser_tunables_panel.ts';
import { getSimulationTunablesPanel } from './simulation_tunables_panel.ts';
import { getScenePropertiesPanel } from './scene_properties_panel.ts';
import { DEFAULT_DENOISER_TUNABLES, type DenoiserTunables, DEFAULT_SIMULATION_TUNABLES, type SimulationTunables } from './litbox/simulation.ts';
import { SCENE_REGISTRY, DEFAULT_SCENE_KEY } from './litbox_scene_registry.ts';
import introMdText from './intro.md?raw';

// Import markdown files as URLs. Vite will handle resolving these paths correctly
// for both development and production builds.
import specialistResumeText from './resumes/resume-specialist.md?raw';
import generalistResumeText from './resumes/resume-generalist.md?raw';

// --- DOM ELEMENT SELECTION ---
const appContainer = document.querySelector('.app-container') as HTMLElement;
const activityBarButtons = document.querySelectorAll('.activity-bar button');
const sidebarPane = document.querySelector('.sidebar-pane') as HTMLElement;
const workspaceViewport = document.querySelector('.workspace-viewport') as HTMLElement;
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const resumeView = document.querySelector('.resume-view') as HTMLElement;
const canvasLoadingOverlay = document.querySelector('.canvas-loading-overlay') as HTMLElement;
const webgpuErrorOverlay = document.querySelector('.webgpu-error-overlay') as HTMLElement;
const webgpuErrorReason = document.querySelector('.webgpu-error-reason') as HTMLElement;
const webgpuErrorTips = document.querySelector('.webgpu-error-tips') as HTMLElement;
const webgpuStatus = document.getElementById('webgpu-status') as HTMLElement;
const sceneStatusText = document.getElementById('scene-status-text') as HTMLElement;

/** Refreshes the status bar's center panel from the currently active scene - see LitboxScene.getStatusText. */
function refreshSceneStatusText(renderer: LitboxSceneRenderer): void {
    sceneStatusText.textContent = renderer.getActiveScene()?.getStatusText() ?? '';
}

// --- CANVAS LOADING SPINNER ---
// Covers both the initial WebGPU/scene bring-up (start()) and later scene swaps/resizes
// (setScene()/resizeSimulation()), since both go through the same slow path - GPU resource
// rebuild plus texture/EXR fetches (see LitboxSceneRenderer.rebuildFromScene) - that's what
// makes the canvas appear to "hang" for a few seconds without this.
//
// Tracked separately from the overlay's own `hidden` state, mirroring webgpuErrorActive above:
// the overlay lives inside .workspace-viewport, which stays visible on the "about" view (behind
// resumeView), so it needs to stay hidden there too regardless of whether a load is in flight.
let canvasLoadingActive = true;

function updateCanvasLoadingVisibility(isAboutView: boolean): void {
    canvasLoadingOverlay.hidden = !canvasLoadingActive || isAboutView;
}

function showCanvasLoading(): void {
    canvasLoadingActive = true;
    updateCanvasLoadingVisibility(appContainer.dataset.activeView === 'about');
}

function hideCanvasLoading(): void {
    canvasLoadingActive = false;
    updateCanvasLoadingVisibility(appContainer.dataset.activeView === 'about');
}

// --- WEBGPU FAILURE DIAGNOSTICS ---
// initWebGPU() (see litbox_scene_renderer.ts) fails silently by design - it's a background
// resource-setup step, not something worth throwing over - so the UI has to actively check
// for it here rather than relying on a rejected promise. Without this, a failed init used to
// just leave a blank canvas with the status bar's hardcoded "WebGPU: Active" claiming
// otherwise (see 🐛 note in git history / CLAUDE.md) - misleading during exactly the moment a
// visitor most needs an explanation.
const WEBGPU_FAILURE_MESSAGES: Record<WebGpuInitFailureReason, { reason: string; tips: string[] }> = {
    'unsupported': {
        reason: "Your browser doesn't support the WebGPU API this scene needs to render.",
        tips: [
            "Try a recent version of Chrome, Edge, or Safari 18+ - WebGPU isn't available in every browser yet.",
            "On Firefox, WebGPU currently has to be enabled manually via about:config.",
        ],
    },
    'no-adapter': {
        reason: "Your browser supports WebGPU, but couldn't find or allocate a GPU for it to use.",
        tips: [
            "Reboot the device - browsers disable GPU acceleration after repeated GPU-process crashes, and this usually clears on restart.",
            "On Chrome, visit chrome://gpu and check the \"Problems Detected\" section for the specific cause.",
            "Turn off battery saver / low-power mode, which can restrict GPU access.",
            "Close other GPU-heavy tabs or apps and try again.",
            "Make sure the browser and OS are up to date.",
        ],
    },
    'device-error': {
        reason: "Your browser found a GPU, but something went wrong while setting it up.",
        tips: [
            "Reboot the device and try again.",
            "Try a different browser.",
            "Open the browser console for more detail on the underlying error.",
        ],
    },
};

// Tracked separately from the overlay's own display style because the overlay lives inside
// .workspace-viewport, which stays visible on every view except "about" (see updateView's
// canvas/resumeView toggle) - it needs to hide there too rather than covering the resume text,
// then reappear if the visitor navigates back to "intro"/"litbox".
let webgpuErrorActive = false;

function updateWebGpuErrorVisibility(isAboutView: boolean): void {
    webgpuErrorOverlay.style.display = (webgpuErrorActive && !isAboutView) ? 'flex' : 'none';
}

function showWebGpuError(reason: WebGpuInitFailureReason): void {
    const info = WEBGPU_FAILURE_MESSAGES[reason];
    webgpuErrorReason.textContent = info.reason;
    webgpuErrorTips.replaceChildren(...info.tips.map(tip => {
        const li = document.createElement('li');
        li.textContent = tip;
        return li;
    }));
    webgpuErrorActive = true;
    updateWebGpuErrorVisibility(appContainer.dataset.activeView === 'about');
    webgpuStatus.textContent = '🔴 WebGPU: Unavailable';
}

// --- CONSOLE LOG CAPTURE ---
const consoleContainer = document.getElementById('console-container');

if (consoleContainer) {
    const consoleContainerNonNull: HTMLElement = consoleContainer;
    const originalLog = console.log;
    const originalWarn = console.warn;
    const originalError = console.error;

    function appendLog(message: string, type: 'log' | 'warn' | 'error') {
        const logEntry = document.createElement('div');
        logEntry.textContent = message;
        logEntry.classList.add(`log-${type}`);
        consoleContainerNonNull.appendChild(logEntry);
        // Limit the number of messages to prevent performance issues
        if (consoleContainerNonNull.children.length > 50) {
            consoleContainerNonNull.removeChild(consoleContainerNonNull.children[0]);
        }
        // Relay to the Vite dev server's terminal (see vite.config.ts's consoleLogRelay
        // plugin) so mobile-device console output is visible without a tethered devtools
        // session. import.meta.env.DEV keeps this out of the deployed GitHub Pages build,
        // where there's no dev server listening on the other end.
        if (import.meta.env.DEV) {
            navigator.sendBeacon('/__consolelog', `[${type}] ${message}`);
        }
    }

    console.log = (...args: any[]) => {
        originalLog(...args);
        appendLog(args.map(arg => String(arg)).join(' '), 'log');
    };

    console.warn = (...args: any[]) => {
        originalWarn(...args);
        appendLog(args.map(arg => String(arg)).join(' '), 'warn');
    };

    console.error = (...args: any[]) => {
        originalError(...args);
        appendLog(args.map(arg => String(arg)).join(' '), 'error');
    };

    // Uncaught exceptions and unhandled promise rejections don't go through
    // console.error, so without these they'd never reach the on-page overlay.
    window.addEventListener('error', (event) => {
        appendLog(`Uncaught error: ${event.message} (${event.filename}:${event.lineno}:${event.colno})`, 'error');
    });

    window.addEventListener('unhandledrejection', (event) => {
        appendLog(`Unhandled promise rejection: ${String(event.reason)}`, 'error');
    });
}


// --- SCENE SELECTION ---
// Key into SCENE_REGISTRY for whichever scene is currently loaded - kept in sync by the
// scene-select 'change' handler below, so revisiting the litbox view (which regenerates
// sidebarPane.innerHTML from scratch) can restore the dropdown's selection.
let activeSceneKey: string = DEFAULT_SCENE_KEY;

const sceneSelectOptions = Object.entries(SCENE_REGISTRY)
    .map(([key, { label }]) => `<option value="${key}">${label}</option>`)
    .join('');

// --- VIEW DATA MODEL ---
const viewContent = {
    intro: {
        sidebar: introMdText,
    },
    litbox: {
        sidebar: `
            <h3>Litbox Config</h3>
            <div class="litbox-config">
                <div class="litbox-config-row">
                    <label for="scene-select">Scene</label>
                    <select id="scene-select" class="scene-select">${sceneSelectOptions}</select>
                </div>
            </div>
            <div id="scene-properties-container"></div>
            <div class="litbox-config">
                <div class="litbox-config-row">
                    <label for="exposure-slider">Exposure</label>
                    <div class="litbox-config-controls">
                        <input type="range" id="exposure-slider" class="slider litbox-config-slider" data-litbox-param="exposure"
                            min="-4" max="4" step="0.1" value="0">
                        <input type="number" class="litbox-config-number" data-litbox-param="exposure"
                            min="-4" max="4" step="0.1" value="0">
                    </div>
                </div>
                <div class="litbox-config-row litbox-config-checkbox-row">
                    <label><input type="checkbox" id="tonemap-toggle" checked> Tone mapping</label>
                </div>
                <div class="litbox-config-row litbox-config-checkbox-row">
                    <label><input type="checkbox" id="denoiser-toggle" checked> Denoising</label>
                </div>
            </div>
        `,
    },
    fractals: {
        sidebar: `
            <h3>Fractal Parameters</h3>
            <p>Zoom: <input type="range" min="1" max="1000" value="100" class="slider"></p>
            <p>Max Iterations: <input type="range" min="10" max="1000" value="200" class="slider"></p>
        `,
    },
    about: {
        content: getAboutPageContent(),
    },
    contact: {
        sidebar: getContactForm(),
    }
};

type ViewKey = keyof typeof viewContent;

/** The active scene's raw (pre-device-scale) simulation resolution, for the simulation tunables panel's width/height rows - see LitboxSceneRenderer.resizeSimulation's own doc comment for why this reads the scene's own JSON rather than SimulationResources' device-scaled effective resolution. */
function getRawSimulationSize(): { width: number; height: number } {
    const simulation = litboxRenderer?.getActiveScene()?.data.simulations[0];
    return simulation ? { width: simulation.width, height: simulation.height } : { width: 0, height: 0 };
}

/**
 * Regenerates both tunables panels from current state - shared by the scene-switch and
 * simulation-resize handlers below, both of which reset denoiserTunables/simulationTunables to
 * the newly-loaded scene's own defaults (see SimulationResources.loadFromScene) and so need the
 * displayed panels refreshed rather than left showing the previous (possibly user-tweaked) values.
 */
function refreshTunablesPanels(): void {
    if (!litboxRenderer) {
        return;
    }
    const denoiserTunablesEl = sidebarPane.querySelector('.denoiser-tunables');
    if (denoiserTunablesEl) {
        denoiserTunablesEl.outerHTML = getDenoiserTunablesPanel(litboxRenderer.getSimulationResources().denoiserTunables);
    }
    const simulationTunablesEl = sidebarPane.querySelector('.simulation-tunables');
    if (simulationTunablesEl) {
        simulationTunablesEl.outerHTML = getSimulationTunablesPanel(litboxRenderer.getSimulationResources().simulationTunables, getRawSimulationSize());
    }
}

// --- VIEW SWITCHING LOGIC ---
async function updateView(view: ViewKey) {
    // Update container attribute for CSS targeting
    appContainer.dataset.activeView = view;

    // Update active button state
    activityBarButtons.forEach(button => {
        button.classList.toggle('active', (button as HTMLElement).dataset.view === view);
    });

    const isAboutView = view === 'about';

    if (!isAboutView) {
        const content = viewContent[view] as { sidebar: string };
        sidebarPane.classList.toggle('markdown-content', view === 'intro');
        if (view === 'intro') {
            sidebarPane.innerHTML = await marked.parse(content.sidebar);
        } else {
            sidebarPane.innerHTML = content.sidebar;
        }
        if (view === 'litbox') {
            // Read live values (not baked into the static viewContent.litbox.sidebar string) so
            // revisiting this view after adjusting a control shows what's actually running, not a
            // reset to the markup's hardcoded defaults. Falls back to defaults if the scene hasn't
            // finished loading yet. Bug fix: the exposure slider/tonemap-toggle/denoiser-toggle
            // used to skip this and always come back showing their hardcoded markup state (e.g.
            // "checked") even after being changed, since sidebarPane.innerHTML above regenerates
            // them fresh from the static template every time.
            const simulationTunables = litboxRenderer?.getSimulationResources().simulationTunables ?? DEFAULT_SIMULATION_TUNABLES;
            sidebarPane.insertAdjacentHTML('beforeend', getSimulationTunablesPanel(simulationTunables, getRawSimulationSize()));

            const tunables = litboxRenderer?.getSimulationResources().denoiserTunables ?? DEFAULT_DENOISER_TUNABLES;
            sidebarPane.insertAdjacentHTML('beforeend', getDenoiserTunablesPanel(tunables));

            const sceneSelect = document.getElementById('scene-select') as HTMLSelectElement | null;
            if (sceneSelect) {
                sceneSelect.value = activeSceneKey;
            }
            const scenePropertiesContainer = document.getElementById('scene-properties-container');
            if (scenePropertiesContainer) {
                scenePropertiesContainer.innerHTML = getScenePropertiesPanel(litboxRenderer?.getActiveScene() ?? null, canvasLoadingActive);
            }

            const exposureValue = String(litboxRenderer?.exposureOverride ?? 0);
            sidebarPane.querySelectorAll<HTMLInputElement>('[data-litbox-param="exposure"]').forEach(el => {
                el.value = exposureValue;
            });
            const tonemapToggle = document.getElementById('tonemap-toggle') as HTMLInputElement | null;
            if (tonemapToggle) {
                tonemapToggle.checked = litboxRenderer?.tonemapEnabled ?? true;
            }
            const denoiserToggle = document.getElementById('denoiser-toggle') as HTMLInputElement | null;
            if (denoiserToggle) {
                denoiserToggle.checked = litboxRenderer?.getSimulationResources().denoiserEnabled ?? true;
            }
        }
    }

    // Show/hide main content
    resumeView.style.display = isAboutView ? 'block' : 'none';
    canvas.style.display = isAboutView ? 'none' : 'block';
    updateWebGpuErrorVisibility(isAboutView);
    updateCanvasLoadingVisibility(isAboutView);

    updateLayout();
}

// --- EVENT LISTENERS ---
sidebarPane.addEventListener('click', async (e: MouseEvent) => {
    // Handle navigation for links within the sidebar, like in the intro markdown.
    const target = e.target as HTMLElement;
    // Use .closest('a') to handle clicks on elements inside a link (e.g. <strong>)
    const anchor = target.closest('a');

    if (anchor && anchor.getAttribute('href') === '/about') {
        e.preventDefault();
        // This is a link to an internal "activity", so switch views instead of navigating.
        await updateView('about');
    }
});

sidebarPane.addEventListener('input', (e: Event) => {
    const target = e.target as HTMLElement;

    // Litbox Config panel (Exposure): the slider and textbox share the same data-litbox-param
    // attribute, mirroring the denoiser/simulation tunables panels' data-param pairing below.
    const litboxParam = target.dataset.litboxParam;
    if (litboxParam) {
        const value = parseFloat((target as HTMLInputElement).value);
        if (Number.isNaN(value)) {
            return;
        }
        const row = target.closest('.litbox-config-row');
        const pairedSelector = target.classList.contains('litbox-config-slider') ? '.litbox-config-number' : '.litbox-config-slider';
        const paired = row?.querySelector<HTMLInputElement>(pairedSelector);
        if (paired) {
            paired.value = String(value);
        }
        if (litboxParam === 'exposure' && litboxRenderer) {
            litboxRenderer.exposureOverride = value;
        }
        return;
    }

    // Denoiser tunables panel (see denoiser_tunables_panel.ts): the slider and textbox in a row
    // share the same data-param attribute (the DenoiserTunables key), differing only by class -
    // one delegated handler for all of them instead of one id-based branch per parameter.
    const param = target.dataset.param as keyof DenoiserTunables | undefined;
    if (param && litboxRenderer) {
        const value = parseFloat((target as HTMLInputElement).value);
        if (Number.isNaN(value)) {
            return;
        }
        litboxRenderer.getSimulationResources().denoiserTunables[param] = value;
        // Keep the OTHER control in this row (slider <-> textbox) in sync with whichever one the
        // user just edited.
        const row = target.closest('.denoiser-param');
        const pairedSelector = target.classList.contains('denoiser-param-slider') ? '.denoiser-param-number' : '.denoiser-param-slider';
        const paired = row?.querySelector<HTMLInputElement>(pairedSelector);
        if (paired) {
            paired.value = String(value);
        }
        return;
    }

    // Simulation tunables panel (see simulation_tunables_panel.ts): the slider and textbox in a
    // row share the same data-sim-param attribute (a SimulationTunables key) - same pairing
    // convention as the denoiser panel above, kept as a separate attribute (not data-param) so it
    // can't collide with a DenoiserTunables key and get routed into the wrong tunables object.
    const simParam = target.dataset.simParam as keyof SimulationTunables | undefined;
    if (simParam && litboxRenderer) {
        const value = parseFloat((target as HTMLInputElement).value);
        if (Number.isNaN(value)) {
            return;
        }
        litboxRenderer.getSimulationResources().simulationTunables[simParam] = value;
        const row = target.closest('.simulation-param');
        const pairedSelector = target.classList.contains('simulation-param-slider') ? '.simulation-param-number' : '.simulation-param-slider';
        const paired = row?.querySelector<HTMLInputElement>(pairedSelector);
        if (paired) {
            paired.value = String(value);
        }
        return;
    }

    // Simulation tunables panel's width/height rows: same slider/textbox pairing as above, but
    // deliberately NOT written back here - a resize needs an async GPU resource rebuild
    // (LitboxSceneRenderer.resizeSimulation), so it's handled on 'change' below, not on every
    // 'input' tick.
    if (target.dataset.simSize) {
        const value = parseFloat((target as HTMLInputElement).value);
        if (Number.isNaN(value)) {
            return;
        }
        const row = target.closest('.simulation-param');
        const pairedSelector = target.classList.contains('simulation-param-slider') ? '.simulation-param-number' : '.simulation-param-slider';
        const paired = row?.querySelector<HTMLInputElement>(pairedSelector);
        if (paired) {
            paired.value = String(value);
        }
        return;
    }

    // Scene Properties panel (see scene_properties_panel.ts): the slider and textbox in a row
    // share the same data-scene-slider-index attribute (the slider's index into
    // scene.getSliders()) - same pairing convention as the two panels above.
    const sceneSliderIndexAttr = target.dataset.sceneSliderIndex;
    if (sceneSliderIndexAttr !== undefined) {
        const activeScene = litboxRenderer?.getActiveScene();
        const index = parseInt(sceneSliderIndexAttr, 10);
        const value = parseFloat((target as HTMLInputElement).value);
        if (Number.isNaN(value) || !activeScene) {
            return;
        }
        const slider = activeScene.getSliders()[index];
        if (slider) {
            slider.setValue(value);
        }
        const row = target.closest('.scene-param');
        const pairedSelector = target.classList.contains('scene-param-slider') ? '.scene-param-number' : '.scene-param-slider';
        const paired = row?.querySelector<HTMLInputElement>(pairedSelector);
        if (paired) {
            paired.value = String(value);
        }
    }
});

sidebarPane.addEventListener('change', async (e: Event) => {
    const target = e.target as HTMLElement;
    if (target.id === 'tonemap-toggle' && litboxRenderer) {
        litboxRenderer.tonemapEnabled = (target as HTMLInputElement).checked;
    } else if (target.id === 'denoiser-toggle' && litboxRenderer) {
        litboxRenderer.getSimulationResources().denoiserEnabled = (target as HTMLInputElement).checked;
    } else if (target.id === 'scene-select' && litboxRenderer) {
        const key = (target as HTMLSelectElement).value;
        const entry = SCENE_REGISTRY[key];
        if (!entry) {
            return;
        }
        activeSceneKey = key;
        showCanvasLoading();
        const scene = await entry.load();
        await litboxRenderer.setScene(scene);
        hideCanvasLoading();
        refreshSceneStatusText(litboxRenderer);
        // The newly-loaded scene has its own slider set (and its own live values), so the panel
        // needs regenerating rather than just re-reading the same controls in place.
        const scenePropertiesContainer = document.getElementById('scene-properties-container');
        if (scenePropertiesContainer) {
            scenePropertiesContainer.innerHTML = getScenePropertiesPanel(litboxRenderer.getActiveScene(), false);
        }
        // setScene() just reset denoiserTunables/simulationTunables to the new scene's own
        // defaults (see LitboxScene.getDenoiserTunables/getSimulationTunables) - regenerate both
        // panels so they don't keep showing the previous scene's (possibly user-tweaked) values.
        refreshTunablesPanels();
    } else if (target.dataset.simSize && litboxRenderer) {
        // Simulation tunables panel's width/height rows (see simulation_tunables_panel.ts) - a
        // resize needs both dimensions together, regardless of which row's control the user just
        // changed, so read both current number-box values rather than just `target`'s own.
        const widthInput = sidebarPane.querySelector<HTMLInputElement>('.simulation-tunables [data-sim-size="width"].simulation-param-number');
        const heightInput = sidebarPane.querySelector<HTMLInputElement>('.simulation-tunables [data-sim-size="height"].simulation-param-number');
        const width = parseFloat(widthInput?.value ?? '');
        const height = parseFloat(heightInput?.value ?? '');
        if (Number.isNaN(width) || Number.isNaN(height)) {
            return;
        }
        showCanvasLoading();
        await litboxRenderer.resizeSimulation(width, height);
        hideCanvasLoading();
        // resizeSimulation reloads the scene at the new resolution, which resets
        // denoiserTunables/simulationTunables to this scene's defaults the same way an actual
        // scene switch does - regenerate both panels for the same reason as the scene-select
        // branch above.
        refreshTunablesPanels();
    }
});

activityBarButtons.forEach(button => {
    button.addEventListener('click', async () => {
        const view = (button as HTMLElement).dataset.view;
        if (view && view in viewContent) {
            await updateView(view as ViewKey);
        }
    });
});

// --- LAYOUT & RESIZE LOGIC ---
function updateLayout() {
    // To break the feedback loop, we must measure the viewport's size without
    // the influence of the canvas's aspect ratio.
    // 1. Temporarily hide the canvas so it doesn't affect the layout.
    const originalDisplay = canvas.style.display;
    canvas.style.display = 'none';

    // 2. Now, the viewport's dimensions are purely determined by the CSS grid.
    const rect = workspaceViewport.getBoundingClientRect();

    // 3. Restore the canvas's visibility.
    canvas.style.display = originalDisplay;

    // 4. Apply the correct dimensions to the canvas.
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
        // Manually trigger a render to avoid stretching during resize.
        litboxRenderer?.render();
    }
}

// --- INITIALIZE ---
// Set default view
updateView('intro');

// Initialize the Litbox scene renderer (shared by the intro and litbox views).
let litboxRenderer: LitboxSceneRenderer | null = null;
if (canvas) {
    const renderer = new LitboxSceneRenderer(canvas);

    // Set initial size
    updateLayout();

    // Use ResizeObserver on the main container to react to any size changes
    const resizeObserver = new ResizeObserver(() => {
        updateLayout();
    });
    resizeObserver.observe(appContainer);

    SCENE_REGISTRY[DEFAULT_SCENE_KEY].load()
        .then(scene => renderer.setScene(scene))
        .then(() => renderer.start())
        .then(() => {
            hideCanvasLoading();
            if (renderer.initFailureReason) {
                showWebGpuError(renderer.initFailureReason);
                return;
            }
            litboxRenderer = renderer;
            refreshSceneStatusText(renderer);
            // Exposed for manual debugging from the devtools console, e.g.
            // `litboxRenderer.debugView = 'lightmap'` - see LitboxSceneRenderer.debugView.
            (window as unknown as { litboxRenderer: LitboxSceneRenderer }).litboxRenderer = renderer;
            // If the user is already sitting on the Litbox tab, its sidebar was built while
            // litboxRenderer was still null (updateView('litbox') falls back to defaults/empty
            // panels in that case - see its own comment) - re-run it now so Scene Properties and
            // the other live-state panels populate without the user needing to tab away and back.
            if (appContainer.dataset.activeView === 'litbox') {
                void updateView('litbox');
            }
        })
        .catch(error => {
            hideCanvasLoading();
            console.error('Failed to start Litbox scene renderer:', error);
            showWebGpuError('device-error');
        });
} else {
    console.error("Canvas element not found!");
}

// --- PERFORMANCE METRICS DISPLAY ---
const perfFpsEl = document.getElementById('perf-fps');
const perfPhotonsEl = document.getElementById('perf-photons');
const PERF_DISPLAY_UPDATE_INTERVAL_MS = 250;

setInterval(() => {
    if (!litboxRenderer) {
        return;
    }
    if (perfFpsEl) {
        perfFpsEl.textContent = formatRate(litboxRenderer.getFps());
    }
    if (perfPhotonsEl) {
        perfPhotonsEl.textContent = formatRate(litboxRenderer.getPhotonWritesPerSecond());
    }
}, PERF_DISPLAY_UPDATE_INTERVAL_MS);

// --- CONTACT MODAL ---
const contactLink = document.getElementById('contact-link');
if (contactLink) {
    contactLink.addEventListener('click', (e) => {
        e.preventDefault();
        const modal = ModalDialog.getInstance();
        modal.show(getContactForm());
    });
}

/**
 * Parses markdown text and injects it into a container.
 * @param text The markdown text to parse.
 * @param containerSelector The CSS selector for the container element.
 */
async function renderResume(text: string, containerSelector: string) {
    try {
        const container = document.querySelector(containerSelector);
        if (container) {
            container.classList.add('markdown-content');
            container.innerHTML = await marked.parse(text);
        }
    } catch (e) {
        console.error(`Failed to render resume in ${containerSelector}:`, e);
    }
}

// --- FETCH AND RENDER RESUMES ---
renderResume(specialistResumeText, '.resume-view-specialist');
renderResume(generalistResumeText, '.resume-view-generalist');
