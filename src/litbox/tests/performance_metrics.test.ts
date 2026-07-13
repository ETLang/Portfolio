import { describe, expect, it } from 'vitest';
import { RollingRateCounter, computeRateFromDelta, formatRate } from '../performance_metrics.ts';

describe('computeRateFromDelta', () => {
    it('scales a count over an elapsed-ms window up to a per-second rate', () => {
        expect(computeRateFromDelta(30, 500)).toBeCloseTo(60);
        expect(computeRateFromDelta(1000, 1000)).toBeCloseTo(1000);
    });

    it('returns 0 instead of Infinity/NaN when elapsedMs is not positive', () => {
        expect(computeRateFromDelta(30, 0)).toBe(0);
        expect(computeRateFromDelta(30, -5)).toBe(0);
    });
});

describe('RollingRateCounter', () => {
    it('reports 0 until the first window closes', () => {
        const counter = new RollingRateCounter(100);
        counter.tick(0);
        counter.tick(50);
        expect(counter.getRate()).toBe(0);
    });

    it('reports frames-per-second once a window elapses', () => {
        const counter = new RollingRateCounter(100);
        counter.tick(0);
        for (let t = 10; t <= 100; t += 10) {
            counter.tick(t);
        }
        // 11 ticks (the initial tick(0) plus 10 more) over a 100ms window -> 110/s.
        expect(counter.getRate()).toBeCloseTo(110);
    });

    it('starts a fresh window after closing the previous one', () => {
        const counter = new RollingRateCounter(100);
        for (let t = 0; t <= 100; t += 10) {
            counter.tick(t);
        }
        const firstRate = counter.getRate();
        expect(firstRate).toBeGreaterThan(0);

        // A slower cadence in the next window should produce a lower rate.
        counter.tick(150);
        counter.tick(300);
        expect(counter.getRate()).toBeLessThan(firstRate);
    });
});

describe('formatRate', () => {
    it('renders sub-1000 values as plain rounded integers', () => {
        expect(formatRate(0)).toBe('0');
        expect(formatRate(42.4)).toBe('42');
        expect(formatRate(999.6)).toBe('1000');
    });

    it('uses K/M/B suffixes instead of scientific notation above 1000', () => {
        expect(formatRate(1500)).toBe('1.50K');
        expect(formatRate(2_340_000)).toBe('2.34M');
        expect(formatRate(5_000_000_000)).toBe('5.00B');
    });

    it('always renders exactly 3 significant digits, regardless of magnitude within a suffix', () => {
        expect(formatRate(4_567)).toBe('4.57K');
        expect(formatRate(45_678)).toBe('45.7K');
        expect(formatRate(123_456)).toBe('123K');
    });

    it('preserves sign', () => {
        expect(formatRate(-1500)).toBe('-1.50K');
    });
});
