// test_color_spectrum.js — the rainbow spectrum: a backend or a pattern gets a colour of a hue that differs clearly from the others'.
import { describe, it, expect } from 'vitest';
import { TraceDataProvider, COLOR_SPECTRUMS, BACKEND_ORDER, PATTERN_ORDER } from '../../script/TraceDataProvider.js';

const provider = colorSpectrum => new TraceDataProvider({ fetchGraph: () => Promise.reject(new Error('none')), settings: {}, colorSpectrum });
const rgb = css => css.match(/\d+/g).map(Number);
const hue = ([r, g, b]) => {
    const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
    const h = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
    return (h * 60 + 360) % 360;
};
const apart = (a, b) => { const d = Math.abs(a - b) % 360; return Math.min(d, 360 - d); };
const minGap = colors => {
    const hues = colors.map(c => hue(rgb(c)));
    let least = 360;
    hues.forEach((h, i) => hues.forEach((o, j) => { if (i < j) least = Math.min(least, apart(h, o)); }));
    return least;
};

describe('the rainbow spectrum', () => {
    const p = provider('rainbow');

    it('has the colours of visible light, violet to red, at hues that are the same distance apart', () => {
        const hues = COLOR_SPECTRUMS.rainbow.stops.map(c => hue(c));
        expect(hues[0]).toBeGreaterThan(260);
        expect(hues[hues.length - 1]).toBeLessThan(5);
        for (let i = 1; i < hues.length; i++) expect(Math.abs(hues[i - 1] - hues[i] - 40)).toBeLessThan(2);
    });

    it('gives every backend and every pattern a hue that differs from all the others', () => {
        expect(minGap(BACKEND_ORDER.map(b => p.backendColor(b)))).toBeGreaterThanOrEqual(38);
        expect(minGap(PATTERN_ORDER.map(b => p.patternColor(b)))).toBeGreaterThanOrEqual(38);
        expect(new Set(PATTERN_ORDER.map(b => p.patternColor(b))).size).toBe(PATTERN_ORDER.length);
    });

    it('puts the categories on the hues themselves, and keeps the colour of each whatever else there is', () => {
        for (const b of BACKEND_ORDER) expect(COLOR_SPECTRUMS.rainbow.stops.map(c => `rgb(${c.join(',')})`)).toContain(p.backendColor(b));
        expect(p.backendColor('CPU')).toBe(p.backendColor('CPU'));
        expect(p.backendColor('Nothing')).toBeNull();
        expect(p.patternColor('MapOverlap 3D')).toBe(p.patternColor('MapOverlap'));
    });

    it('does not change what the other spectra give', () => {
        const v = provider('viridis');
        expect(v.backendColor('OpenMP')).toBe(v.spectrumColor(0.25));
        expect(v.patternColor('Scan')).toBe(v.spectrumColor(3 / 7));
    });
});
