// test_webview_html.js — the page of an editor's webview is index.html with what an editor needs (src/webviewHtml.ts): the content security
// policy, the folder of the extension as the base of the URLs, a nonce on every script, the script of the viewer, and without the parts of the page
// that are for a browser.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildWebviewHtml, contentSecurityPolicy } from '../../src/webviewHtml.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const index = readFileSync(join(ROOT, 'index.html'), 'utf8');
const page = { baseUri: 'https://file+.vscode-resource.vscode-cdn.net/ext', cspSource: 'https://*.vscode-cdn.net', nonce: 'N0nce',
               entryScript: 'dist/skevu.js', skevu: { dagreWorker: 'https://x/dagre-worker.js', elkWorker: 'https://x/elk.js' } };
const html = buildWebviewHtml(index, page);
const count = (re, s = html) => (s.match(re) ?? []).length;

describe('the page of the webview', () => {
    it('has the policy and the base before everything with a URL', () => {
        const csp = html.indexOf('Content-Security-Policy'), base = html.indexOf('<base '), first = html.search(/<(script|link)\b/);
        expect(csp).toBeGreaterThan(0);
        expect(csp).toBeLessThan(first);
        expect(base).toBeLessThan(first);
        expect(html).toContain('<base href="https://file+.vscode-resource.vscode-cdn.net/ext/">');
    });

    it('lets scripts run only by the nonce, and has no inline handlers', () => {
        expect(contentSecurityPolicy('S', 'N')).toContain("script-src 'nonce-N';");
        expect(contentSecurityPolicy('S', 'N')).not.toMatch(/script-src[^;]*unsafe/);
        expect(contentSecurityPolicy('S', 'N')).toContain("default-src 'none'");
        expect(contentSecurityPolicy('S', 'N')).not.toMatch(/(img|connect|font|worker)-src[^;]*unsafe/);
        expect(count(/<script\b/g)).toBeGreaterThan(5);
        expect(count(/<script\b(?![^>]*nonce="N0nce")/g)).toBe(0);
        expect(html.match(/\son[a-z]+\s*=\s*["']/gi) ?? []).toEqual([]);
    });

    it('has none of the parts for a browser, and the script of the viewer last', () => {
        expect(html).not.toContain('data-variant="web"');
        expect(html).not.toContain('worker-loader-web');
        expect(html).not.toContain('main_web.js');
        expect(html).not.toContain('designer.html');
        expect(html).not.toContain('href="."');
        const scripts = [...html.matchAll(/<script\b[^>]*src="([^"]*)"/g)].map(m => m[1]);
        expect(scripts[scripts.length - 1]).toBe('dist/skevu.js');
        expect(scripts).toContain('./static/cytoscape/cytoscape.min.js');
    });

    it('tells the viewer where the extension has its parts, before the viewer runs', () => {
        const config = html.match(/window\.skevu = (\{.*?\});<\/script>/)[1];
        expect(JSON.parse(config)).toEqual({ traceLineRanges: [], dagreWorker: 'https://x/dagre-worker.js', elkWorker: 'https://x/elk.js' });
        expect(html.indexOf('window.skevu =')).toBeLessThan(html.indexOf('src="dist/skevu.js"'));
    });

    it('keeps the markup of the viewer: balanced, with the controls and the panes', () => {
        expect(count(/<div\b/g)).toBe(count(/<\/div>/g));
        expect(count(/data-click=/g)).toBeGreaterThan(50);
        for (const id of ['cy', 'legend', 'gantt-pane', 'info_window', 'cpp_tab_bar', 'menu-check-definitions'])
            expect(html).toContain(`id="${id}"`);
    });
});
