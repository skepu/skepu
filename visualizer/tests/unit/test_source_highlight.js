// test_source_highlight.js — the colour of the selected lines of the source (script/source_highlight.js): the accent as strongly as the text allows.
import { describe, it, expect } from 'vitest';
import { parseColor, mix, contrast, selectionBackground } from '../../script/source_highlight.js';

const accentLight = parseColor('#0055cc'), accentDark = parseColor('#e8720c');

describe('colours', () => {
    it('are read from the forms of CSS that the browser gives and the themes use', () => {
        expect(parseColor('rgb(46, 52, 64)')).toEqual({ r: 46, g: 52, b: 64 });
        expect(parseColor('#fff')).toEqual({ r: 255, g: 255, b: 255 });
        expect(parseColor('rgba(0, 0, 0, 0.5)', { r: 100, g: 100, b: 100 })).toEqual({ r: 50, g: 50, b: 50 });
        expect(parseColor('nonsense')).toBeNull();
    });

    it('have the contrast of WCAG', () => {
        expect(contrast(parseColor('#000'), parseColor('#fff'))).toBeCloseTo(21, 0);
        expect(contrast(parseColor('#777'), parseColor('#777'))).toBeCloseTo(1, 5);
    });
});

describe('the background of a selected line', () => {
    const themes = {
        'dark, quiet text': { bg: '#2e3440', texts: ['#d8dee9', '#81a1c1', '#a3be8c', '#616e88', '#88c0d0'], accent: accentDark },
        'light': { bg: '#ffffff', texts: ['#24292e', '#d73a49', '#032f62', '#6a737d', '#6f42c1'], accent: accentLight },
        'light theme, dark accent': { bg: '#fafafa', texts: ['#383a42', '#a626a4', '#50a14f', '#a0a1a7'], accent: accentDark },
        'text with little contrast': { bg: '#1a1b26', texts: ['#a9b1d6', '#565f89', '#9ece6a'], accent: accentDark },
    };

    for (const [name, t] of Object.entries(themes)) {
        it(`keeps the base text readable and the syntax colours legible on ${name}`, () => {
            const bg = parseColor(t.bg), texts = t.texts.map(c => parseColor(c));
            const { color, fraction } = selectionBackground(bg, texts, t.accent);
            expect(fraction).toBeGreaterThanOrEqual(0.06);
            expect(contrast(texts[0], color)).toBeGreaterThanOrEqual(Math.min(contrast(texts[0], bg), 4.5) * 0.9 - 1e-9);
            texts.slice(1).forEach(c => expect(contrast(c, color)).toBeGreaterThanOrEqual(Math.min(contrast(c, bg), 3) * 0.55 - 1e-9));
        });
    }

    it('is stronger where the text has contrast to spare', () => {
        const bg = parseColor('#ffffff');
        const strong = selectionBackground(bg, [parseColor('#000000')], accentLight).fraction;
        const weak = selectionBackground(bg, [parseColor('#000000'), parseColor('#8a8a8a')], accentLight).fraction;
        expect(strong).toBeGreaterThan(weak);
    });

    it('is the accent over the background', () => {
        const bg = parseColor('#000000');
        const { color, fraction } = selectionBackground(bg, [parseColor('#ffffff')], accentDark);
        expect(color).toEqual(mix(accentDark, bg, fraction));
    });
});
