// test_shared_ui.js — what SkeVU and the designer share (script/): the colours and themes that the legend and the menus use, the text of the
// tooltips, and the files that the pages load.
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { formatDuration } from '../../script/format.js';
import { nodeTooltipText, edgeTooltipText } from '../../script/tooltip.js';
import { COLORS, regionColor } from '../../script/graph_style.js';
import { HLJS_THEMES, HLJS_THEME_DEFAULT_LIGHT, HLJS_THEME_DEFAULT_DARK, isDarkTheme, knownTheme } from '../../script/appearance.js';
import { COLOR_SPECTRUMS } from '../../script/TraceDataProvider.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '../..');

describe('formatting', () => {
    it('writes durations in the unit that fits', () => {
        expect(formatDuration(12)).toBe('12 μs');
        expect(formatDuration(3400)).toBe('3.4 ms');
        expect(formatDuration(1.5e6)).toBe('1.5 s');
        expect(formatDuration(null)).toBeNull();
        expect(formatDuration(-1)).toBeNull();
    });
});

describe('tooltips', () => {
    it('say what a node is: the label, time, size, backend, pattern and place', () => {
        expect(nodeTooltipText({ label: 'scale', duration: 2500, elements: [32, 48], backend: 'OpenMP', pattern: 'Map', file: '/a/b/prog.cpp', line: 17 }))
            .toBe('scale\ntime     2.5 ms\nsize     32 × 48\nbackend  OpenMP\npattern  Map\nprog.cpp:17');
        expect(nodeTooltipText({ label: 'x', file: '', line: -1 })).toBe('x');
    });
    it('say what an edge is', () => {
        expect(edgeTooltipText({ type: 'forward-dep', label: 'v', access_mode: 'elwise' })).toBe('Data dependence\nv\nmode  elwise');
        expect(edgeTooltipText({ type: 'prng-dep', prng_label: 'noise' })).toBe('PRNG dep: noise');
    });
});

describe('colours', () => {
    it('are the same for the graph and the legend: regions get darker blue by depth, brown in the dark mode', () => {
        expect(regionColor(0, false)).toBe('rgb(225,240,255)');
        expect(regionColor(0, true)).toBe('rgb(42,31,16)');
        expect(regionColor(9, false)).toBe('rgb(20,185,255)');                 // (the limits)
        expect(COLORS.data.fill).toBe('lightgray');
        expect(COLORS.edge).toHaveLength(2);
    });
});

describe('appearance', () => {
    it('has a theme for each of the files that the pages can load, and the defaults are among them', () => {
        for (const theme of HLJS_THEMES) expect(existsSync(join(root, 'static/highlight/styles', `${theme.id}.css`)), theme.id).toBe(true);
        expect(HLJS_THEMES.some(t => t.id === HLJS_THEME_DEFAULT_LIGHT && !t.dark)).toBe(true);
        expect(HLJS_THEMES.some(t => t.id === HLJS_THEME_DEFAULT_DARK && t.dark)).toBe(true);
    });
    it('knows which themes are dark, and falls back to the default for one that is not offered (a remembered one may have been taken away)', () => {
        expect(isDarkTheme('monokai')).toBe(true);
        expect(isDarkTheme('github')).toBe(false);
        expect(knownTheme('nord')).toBe('nord');
        expect(knownTheme('base16/zenburn')).toBe(HLJS_THEME_DEFAULT_LIGHT);
        expect(knownTheme(undefined)).toBe(HLJS_THEME_DEFAULT_LIGHT);
    });
    it('has the spectra that the Colors menu is made of', () => {
        expect(Object.keys(COLOR_SPECTRUMS)).toContain('default');
        for (const spectrum of Object.values(COLOR_SPECTRUMS)) expect(spectrum.stops.length).toBeGreaterThanOrEqual(2);
    });
});

describe('the pages', () => {
    // every script, stylesheet and image that a page names must be there (a library that was taken away is found here)
    for (const page of ['index.html', 'designer.html', 'embed-test.html']) {
        it(`${page} loads files that exist`, () => {
            const html = readFileSync(join(root, page), 'utf8');
            const files = [...html.matchAll(/(?:src|href)="(\.\/[^"#?]+)"/g)].map(m => m[1]).filter(f => !f.endsWith('.html'));
            expect(files.length).toBeGreaterThan(3);
            for (const f of files) expect(existsSync(join(root, f)), f).toBe(true);
        });
    }
    it('do not load jQuery any more, and no script uses it', () => {
        for (const page of ['index.html', 'designer.html', 'embed-test.html']) expect(readFileSync(join(root, page), 'utf8')).not.toMatch(/jquery/i);
        for (const file of ['SourceCodeView.js', 'InfoView.js', 'GraphView.js', 'TimelineView.js', 'ui.js', 'graph.js']) expect(readFileSync(join(root, 'script', file), 'utf8'), file).not.toMatch(/jQuery|(^|[^.\w])\$\(|\)\.scrollTo\(/m);          // (cy.$() is Cytoscape's)
    });
});
