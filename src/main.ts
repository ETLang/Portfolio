import './style.css';
import { marked } from 'marked';
import { ModalDialog } from './modal-dialog.ts';
import { LitboxSceneRenderer } from './litbox_scene_renderer.ts';
import { CornellSquareScene } from './litbox/scenes/cornell_square_scene.ts';
import { getAboutPageContent } from './about.ts';
import { getContactForm } from './contact-form.ts';
import { formatRate } from './litbox/performance_metrics.ts';
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


// --- VIEW DATA MODEL ---
const viewContent = {
    intro: {
        sidebar: introMdText,
    },
    litbox: {
        sidebar: `
            <h3>Litbox Config</h3>
            <p>Rays/Pixel: <input type="range" min="1" max="100" value="50" class="slider"></p>
            <p>Bounce Depth: <input type="range" min="1" max="10" value="5" class="slider"></p>
            <p>Exposure: <input type="range" id="exposure-slider" min="-4" max="4" step="0.1" value="0" class="slider"></p>
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
    }

    // Show/hide main content
    resumeView.style.display = isAboutView ? 'block' : 'none';
    canvas.style.display = isAboutView ? 'none' : 'block';

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
    if (target.id === 'exposure-slider' && litboxRenderer) {
        litboxRenderer.exposureOverride = parseFloat((target as HTMLInputElement).value);
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

    CornellSquareScene.load()
        .then(scene => renderer.setScene(scene))
        .then(() => renderer.start())
        .then(() => { litboxRenderer = renderer; })
        .catch(error => console.error('Failed to start Litbox scene renderer:', error));
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
