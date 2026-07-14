import { describe, expect, it } from 'vitest';
import { detectPlatform, isGpuRandomAccessFriendly, type NavigatorLike } from '../device_environment.ts';

function nav(overrides: Partial<NavigatorLike>): NavigatorLike {
    return { userAgent: '', ...overrides };
}

describe('detectPlatform', () => {
    it('trusts User-Agent Client Hints when present (Chromium)', () => {
        expect(detectPlatform(nav({ userAgentData: { platform: 'Android', mobile: true } }))).toBe('android');
        expect(detectPlatform(nav({ userAgentData: { platform: 'iOS', mobile: true } }))).toBe('ios');
        expect(detectPlatform(nav({ userAgentData: { platform: 'Windows', mobile: false } }))).toBe('desktop');
        expect(detectPlatform(nav({ userAgentData: { platform: 'macOS', mobile: false } }))).toBe('desktop');
    });

    it('falls back to UA-string sniffing when Client Hints are absent (Safari/WebKit)', () => {
        const iphoneUa = 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15';
        expect(detectPlatform(nav({ userAgent: iphoneUa }))).toBe('ios');

        const androidUa = 'Mozilla/5.0 (Linux; Android 14; Pixel 10 Pro) AppleWebKit/537.36 Chrome/120.0';
        expect(detectPlatform(nav({ userAgent: androidUa }))).toBe('android');

        const macUa = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15';
        expect(detectPlatform(nav({ userAgent: macUa, maxTouchPoints: 0 }))).toBe('desktop');
    });

    it('distinguishes an iPad (touch-capable) from a real Mac via maxTouchPoints, since iPadOS reports a Mac-style UA', () => {
        const ipadMasqueradingUa = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15';
        expect(detectPlatform(nav({ userAgent: ipadMasqueradingUa, maxTouchPoints: 5 }))).toBe('ios');
    });

    it('defaults to desktop for an unrecognized UA (e.g. Node.js in tests)', () => {
        expect(detectPlatform(nav({ userAgent: 'Node.js/24' }))).toBe('desktop');
    });
});

describe('isGpuRandomAccessFriendly', () => {
    it('assumes desktop GPUs are friendly and mobile GPUs are not, by default', () => {
        expect(isGpuRandomAccessFriendly('desktop', '')).toBe(true);
        expect(isGpuRandomAccessFriendly('android', '')).toBe(false);
        expect(isGpuRandomAccessFriendly('ios', '')).toBe(false);
    });

    it('overrides the mobile default for known-friendly vendors (e.g. Apple Silicon on iOS)', () => {
        expect(isGpuRandomAccessFriendly('ios', 'apple')).toBe(true);
        expect(isGpuRandomAccessFriendly('ios', 'Apple')).toBe(true);
    });

    it('overrides the desktop default for known-unfriendly vendors (e.g. a Windows-on-ARM laptop)', () => {
        expect(isGpuRandomAccessFriendly('desktop', 'qualcomm')).toBe(false);
        expect(isGpuRandomAccessFriendly('desktop', 'imagination')).toBe(false);
        expect(isGpuRandomAccessFriendly('desktop', 'arm')).toBe(false);
    });

    it('matches known Android GPU vendors as unfriendly even though that already matches the platform default', () => {
        expect(isGpuRandomAccessFriendly('android', 'qualcomm')).toBe(false);
        expect(isGpuRandomAccessFriendly('android', 'imagination')).toBe(false);
    });
});
