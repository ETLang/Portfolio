/**
 * Environmental signals for simulation-tuning decisions (see simulation.ts's
 * getSimulationDeviceProfile) - deliberately NOT the same thing as style.css's <=768px mobile
 * layout breakpoint. That breakpoint is fine for CSS layout (a reasonable, if approximate,
 * responsive-design cutoff); it's the wrong tool for "should this device get a cheaper
 * simulation," since a resized desktop window or a landscape tablet would misclassify. These
 * signals answer the platform/GPU question directly instead of proxying it through viewport width.
 */

export type Platform = 'ios' | 'android' | 'desktop';

/** The subset of `navigator` this module actually reads - lets detectPlatform be tested without a real browser (or worked around Node's own minimal built-in `navigator`, which has none of these fields - see its test file). */
export interface NavigatorLike {
    userAgentData?: { platform?: string; mobile?: boolean };
    userAgent: string;
    maxTouchPoints?: number;
}

/**
 * GPU vendor strings (from GPUAdapterInfo.vendor, matched case-insensitively) known to handle this
 * project's Monte Carlo integrator's scattered atomic writes and incoherent texture reads well,
 * overriding the platform-based default in isGpuRandomAccessFriendly - see
 * forward_monte_carlo.wgsl and CLAUDE.md's mobile-perf-tuning notes for why that pattern is
 * specifically hard on mobile TBDR GPUs tuned for coherent rasterization, not scattered compute.
 */
const RANDOM_ACCESS_FRIENDLY_VENDORS = new Set(['apple']);
/** ...vendors confirmed/expected to handle it poorly, overriding the default the other direction (e.g. an unusual desktop GPU from a normally-mobile vendor). */
const RANDOM_ACCESS_UNFRIENDLY_VENDORS = new Set(['imagination', 'arm', 'qualcomm']);

/**
 * Best-effort iOS/Android/desktop classification. Prefers the User-Agent Client Hints API
 * (Chromium only - Safari/WebKit has stated it won't implement UA-CH) for a direct, OS-reported
 * answer; falls back to UA-string sniffing plus a touch-capability check for iPad, which has
 * reported a desktop-Safari-style "Macintosh" UA string by default since iPadOS 13 (a real Mac
 * reports maxTouchPoints === 0; an iPad reporting a Mac UA still reports touch support).
 */
export function detectPlatform(nav: NavigatorLike): Platform {
    const uaPlatform = nav.userAgentData?.platform?.toLowerCase();
    if (uaPlatform === 'ios') return 'ios';
    if (uaPlatform === 'android') return 'android';
    if (uaPlatform) return 'desktop'; // Client Hints reported something else (windows/macos/linux/chrome os/...) - not mobile.

    const ua = nav.userAgent;
    if (/iPhone|iPod/.test(ua)) return 'ios';
    if (/iPad/.test(ua) || (/Macintosh/.test(ua) && (nav.maxTouchPoints ?? 0) > 1)) return 'ios';
    if (/Android/.test(ua)) return 'android';
    return 'desktop';
}

/**
 * Whether the GPU is known to handle this project's Monte Carlo integrator's scattered atomic
 * writes and incoherent texture reads well. Base assumption: desktop GPUs do, mobile GPUs don't -
 * a recognized vendor string (see RANDOM_ACCESS_*_VENDORS above) overrides that default in either
 * direction. `adapterVendor` is GPUAdapterInfo.vendor - empty string (spec-permitted when a
 * browser doesn't report it) falls through to the platform-only default.
 */
export function isGpuRandomAccessFriendly(platform: Platform, adapterVendor: string): boolean {
    const vendor = adapterVendor.trim().toLowerCase();
    if (RANDOM_ACCESS_FRIENDLY_VENDORS.has(vendor)) return true;
    if (RANDOM_ACCESS_UNFRIENDLY_VENDORS.has(vendor)) return false;
    return platform === 'desktop';
}

// --- Module-singleton accessors: the "accessible from anywhere" surface the pure functions above
// are wrapped in, so callers (e.g. SimulationResources) don't need navigator/adapter plumbed
// through their constructors. Platform is knowable synchronously at any time; GPU vendor isn't
// known until LitboxSceneRenderer.initWebGPU() resolves an adapter, so isRandomAccessFriendlyGpu()
// falls back to the platform-only default until analyzeExecutionEnvironment() is called - by the time any
// scene actually loads (the only real consumer), that's already happened.

let cachedPlatform: Platform | null = null;
let gpuVendor = '';

/** Lazily detected once per page load and cached - the platform doesn't change mid-session. */
export function getPlatform(): Platform {
    if (cachedPlatform === null) {
        cachedPlatform = detectPlatform(navigator as NavigatorLike);
    }
    return cachedPlatform;
}

/** Called once by LitboxSceneRenderer.initWebGPU() with the resolved GPUAdapter's info.vendor. */
export function analyzeExecutionEnvironment(gpuVendorStr: string): void {
    gpuVendor = gpuVendorStr;
}

export function isRandomAccessFriendlyGpu(): boolean {
    return isGpuRandomAccessFriendly(getPlatform(), gpuVendor);
}
