/**
 * Pure, GPU-free helpers backing the on-screen FPS / Photons-per-second readouts (see
 * LitboxSceneRenderer's getFps/getPhotonWritesPerSecond and main.ts's display code). Kept
 * separate from any GPU-touching code so the rate math and formatting are unit-testable
 * without a device.
 */

/** Frames (or other events) per second, averaged over a rolling window rather than a single frame-to-frame delta, to keep a jittery per-frame timing from making the readout unreadable. */
export class RollingRateCounter {
    private readonly windowMs: number;
    private count = 0;
    private windowStartMs: number | null = null;
    private lastRate = 0;

    constructor(windowMs: number) {
        this.windowMs = windowMs;
    }

    /** Call once per event (e.g. once per rendered frame), with the event's timestamp. */
    public tick(nowMs: number): void {
        if (this.windowStartMs === null) {
            this.windowStartMs = nowMs;
            this.count = 1;
            return;
        }
        this.count++;
        const elapsed = nowMs - this.windowStartMs;
        if (elapsed >= this.windowMs) {
            this.lastRate = computeRateFromDelta(this.count, elapsed);
            this.count = 0;
            this.windowStartMs = nowMs;
        }
    }

    /** The rate as of the last completed window; 0 until the first window closes. */
    public getRate(): number {
        return this.lastRate;
    }
}

/** Rate (per second) implied by a change of `deltaCount` over `elapsedMs`. 0 if elapsedMs isn't positive, rather than Infinity/NaN. */
export function computeRateFromDelta(deltaCount: number, elapsedMs: number): number {
    return elapsedMs > 0 ? (deltaCount * 1000) / elapsedMs : 0;
}

/** Renders a positive, already-suffix-scaled magnitude (e.g. 2.34 for "2.34M") to exactly 3 significant digits. */
function toThreeSignificantDigits(scaled: number): string {
    if (scaled >= 100) return scaled.toFixed(0);
    if (scaled >= 10) return scaled.toFixed(1);
    return scaled.toFixed(2);
}

/** Formats a per-second count for compact UI display: whole-number precision, no scientific notation, K/M/B suffixes above 1000 instead, each to 3 significant digits. */
export function formatRate(value: number): string {
    const sign = value < 0 ? '-' : '';
    const abs = Math.abs(value);
    if (abs >= 1e9) return `${sign}${toThreeSignificantDigits(abs / 1e9)}B`;
    if (abs >= 1e6) return `${sign}${toThreeSignificantDigits(abs / 1e6)}M`;
    if (abs >= 1e3) return `${sign}${toThreeSignificantDigits(abs / 1e3)}K`;
    return `${sign}${Math.round(abs)}`;
}
