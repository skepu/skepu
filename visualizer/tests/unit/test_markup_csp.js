// test_markup_csp.js — the page has no inline handlers, and no script that is not a file, in the part that an editor's webview shares
// (a content security policy with a nonce for the scripts does not run them): a control says what it does with data-click / data-input
// (script/actions.js), and every action that the markup names is there.
import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { ACTIONS } from '../../script/actions.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const page = readFileSync(join(ROOT, 'index.html'), 'utf8');

function scripts(dir) {
    return readdirSync(dir).flatMap(f => {
        const p = join(dir, f);
        if (f === 'static') return [];
        return statSync(p).isDirectory() ? scripts(p) : p.endsWith('.js') ? [p] : [];
    });
}
const sources = scripts(join(ROOT, 'script'));

describe('the markup of the page', () => {
    it('has no inline event handlers', () => {
        expect(page.match(/\son[a-z]+\s*=\s*["']/gi) ?? []).toEqual([]);
    });

    it('has inline scripts only for the browser (data-variant="web")', () => {
        const inline = [...page.matchAll(/<script\b([^>]*)>/g)].filter(m => !/\bsrc=/.test(m[1])).map(m => m[1]);
        for (const attrs of inline) expect(attrs).toContain('data-variant="web"');
    });

    it('has the module scripts for the browser only (data-variant="web"): the extension has a bundle', () => {
        for (const m of page.matchAll(/<script\b([^>]*type="module"[^>]*)>/g)) expect(m[1]).toContain('data-variant="web"');
    });

    it('names actions that exist', () => {
        const pageFunctions = new Set();
        for (const file of sources)
            for (const m of readFileSync(file, 'utf8').matchAll(/\bwindow\.(\w+)\s*=/g)) pageFunctions.add(m[1]);
        const named = [...page.matchAll(/data-(?:click|input)="([^"]*)"/g)].flatMap(m => m[1].split(/\s+/).filter(Boolean));
        expect(named.length).toBeGreaterThan(50);
        for (const name of new Set(named))
            expect(name in ACTIONS || pageFunctions.has(name), `an action ${name} that nothing makes`).toBe(true);
    });
});

describe('the markup that the scripts make', () => {
    it('has no inline event handlers', () => {
        const found = [];
        for (const file of sources)
            for (const m of readFileSync(file, 'utf8').matchAll(/[\s"']on(?:click|input|change|mouse\w+|key\w+|load|error)=[\\"']/g))
                found.push(file.slice(ROOT.length + 1) + ': ' + m[0].trim());
        expect(found).toEqual([]);
    });
});
