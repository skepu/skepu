// test_guide.js — the links to the user guide (src/guideLinks.ts): the address of a link, the page of the guide in a webview, and the anchors that the
// hover of a skeleton links to (src/guideHover.ts) against the sections that the sources of the guide (../user-guide) have.
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { GUIDE_ANCHORS, GUIDE_SECTIONS, SKELETON_ANCHORS, guideAnchor, guideTarget, splitTarget, resolveGuideLink, classifyLink, parseGuideUri, buildGuideHtml, skeletonAnchor, skeletonOfDetail, afterSkepuNamespace, guideHoverMarkdown } from '../../src/guideLinks.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const GUIDE_SOURCES = join(ROOT, '..', 'user-guide');

describe('the address of a link to the guide', () => {
    it('has the anchor, if the path is the guide', () => {
        expect(parseGuideUri('/guide', 'anchor=map')).toEqual({ anchor: 'map' });
        expect(parseGuideUri('/guide/', 'anchor=mappool')).toEqual({ anchor: 'mappool' });
        expect(parseGuideUri('/guide', '')).toEqual({ anchor: undefined });
    });

    it('is not for another path', () => {
        expect(parseGuideUri('/node', 'anchor=map')).toBeNull();
        expect(parseGuideUri('', '')).toBeNull();
    });

    it('does not pass on an anchor that is not a name (it goes into the page)', () => {
        expect(guideAnchor('map')).toBe('map');
        expect(guideAnchor('Map')).toBeUndefined();
        expect(guideAnchor('"><script>')).toBeUndefined();
        expect(guideAnchor('')).toBeUndefined();
        expect(parseGuideUri('/guide', 'anchor=%22%3E%3Cscript%3E')).toEqual({ anchor: undefined });
    });
});

describe('a page of the book in a webview', () => {
    const source = '<!DOCTYPE html>\n<html><head><title>t</title></head><body><a href="#x">x</a><p id="x">hi</p></body></html>';
    const page = buildGuideHtml(source, { cspSource: 'https://x.test', baseUri: 'https://x.test/guide/src', rootUri: 'https://x.test/guide', page: 'src/lsp.html', fragment: 'sec-lsp' });

    it('is the page, with a policy, a base for its files, and the script that takes the links', () => {
        expect(page).toContain('<title>t</title>');
        expect(page).toContain('<base href="https://x.test/guide/src/">');
        expect(page.indexOf('Content-Security-Policy')).toBeLessThan(page.indexOf('<base'));
        expect(page.indexOf('<base')).toBeLessThan(page.indexOf('<title>'));
        expect(page).toContain("default-src 'none'");
        expect(page).toContain("script-src https://x.test 'unsafe-inline'");
        expect(page).toContain("type: 'link'");
        expect(page.indexOf("addEventListener('error'")).toBeLessThan(page.indexOf('<title>'));   // (the reports come first, before the scripts of the book)
        expect(page).toContain("'securitypolicyviolation'");
        expect(page).toContain('event.stopPropagation();');   // (the webview opens the links that it sees in the browser)
        expect(page).toContain("hasAttribute('data-bs-toggle')");
        expect(page).toContain('var page = "src/lsp.html";');
        expect(page.indexOf('<script>')).toBeLessThan(page.indexOf('</body>'));
    });

    it('goes to the section at load, and when the extension asks for another in the same page', () => {
        expect(page).toContain('scrollTo("sec-lsp")');
        expect(page).toContain("event.data.type === 'scroll'");
    });

    it('does not wait for the web: no polyfill script, no fonts of Google, only MathJax may be fetched', () => {
        const withPolyfill = '<html><head><title>t</title>\n<script src="https://cdnjs.cloudflare.com/polyfill/v3/polyfill.min.js?features=es6"></script>\n<script defer src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-chtml.js"></script></head><body></body></html>';
        const built = buildGuideHtml(withPolyfill, { cspSource: 'c', baseUri: 'b', rootUri: 'r', page: 'p', fragment: '' });
        expect(built).not.toContain('polyfill');
        expect(built).toContain('mathjax@4/tex-chtml.js');
        expect(built).not.toContain('fonts.googleapis.com');
        expect(built).not.toContain('cdnjs.cloudflare.com');
        expect(built).toContain('https://cdn.jsdelivr.net');
    });

    it('does not let a name end the script', () => {
        const tricky = buildGuideHtml(source, { cspSource: 'c', baseUri: 'b', rootUri: 'r', page: 'p', fragment: '</script><b>' });
        expect(tricky).not.toContain('</script><b>');
    });
});

describe('a link in a page of the book', () => {
    it('goes to another page of the book, from the page that has it', () => {
        expect(resolveGuideLink('src/lsp.html', 'programming.html#sec-map')).toEqual({ kind: 'page', page: 'src/programming.html', fragment: 'sec-map' });
        expect(resolveGuideLink('src/lsp.html', '../index.html')).toEqual({ kind: 'page', page: 'index.html', fragment: '' });
        expect(resolveGuideLink('index.html', 'src/lsp.html#sec-lsp-setup')).toEqual({ kind: 'page', page: 'src/lsp.html', fragment: 'sec-lsp-setup' });
        expect(resolveGuideLink('src/lsp.html', 'programming.html')).toEqual({ kind: 'page', page: 'src/programming.html', fragment: '' });
        expect(resolveGuideLink('src/lsp.html', '../')).toEqual({ kind: 'page', page: 'index.html', fragment: '' });
        expect(resolveGuideLink('index.html', 'src/')).toEqual({ kind: 'page', page: 'src/index.html', fragment: '' });
    });

    it('goes to the web only by https', () => {
        expect(resolveGuideLink('index.html', 'https://quarto.org')).toEqual({ kind: 'external', url: 'https://quarto.org' });
        expect(resolveGuideLink('index.html', 'http://quarto.org').kind).toBe('none');
        expect(resolveGuideLink('index.html', 'https://file+.vscode-resource.vscode-cdn.net/x/guide/src/lsp.html').kind).toBe('none');   // (the webview's own address)
        expect(resolveGuideLink('index.html', 'https://abc.vscode-cdn.net/index.html').kind).toBe('none');
        expect(resolveGuideLink('index.html', 'javascript:alert(1)').kind).toBe('none');
        expect(resolveGuideLink('index.html', 'command:skevu.openGuide').kind).toBe('none');
        expect(resolveGuideLink('index.html', '//evil.test/x.html').kind).toBe('none');
    });

    it('does not leave the book, or open what is not a page', () => {
        expect(resolveGuideLink('src/lsp.html', '../../../etc/passwd.html')).toEqual({ kind: 'page', page: 'etc/passwd.html', fragment: '' });   // (still inside the book's directory: a URL stops at the root)
        expect(resolveGuideLink('index.html', 'search.json').kind).toBe('none');
        expect(resolveGuideLink('index.html', 'a b.html').kind).toBe('none');
        expect(resolveGuideLink('index.html', '').kind).toBe('none');   // (an empty link is the page itself)
        expect(resolveGuideLink('index.html', '#x').kind).toBe('none');
    });

    it('is split from a target of the sections', () => {
        expect(splitTarget('src/programming.html#sec-map')).toEqual({ page: 'src/programming.html', fragment: 'sec-map' });
        expect(splitTarget('src/skevu.html')).toEqual({ page: 'src/skevu.html', fragment: '' });
    });
});

describe('where a link in a page points (classifyLink, which is put into the page)', () => {
    const root = 'https://file+.vscode-resource.vscode-cdn.net/Users/me/ext/guide/';   // (with the slash at the end, as the page has it)
    const base = root + 'src/';
    const here = 'https://abc.vscode-cdn.net/index.html?id=1';
    const classify = (href, resolved) => classifyLink(href, resolved ?? new URL(href, base).href, base, here, root);

    it('is in the page for a link to a section, written as Quarto writes them or not', () => {
        expect(classify('#sec-map')).toEqual({ kind: 'anchor', id: 'sec-map' });
        expect(classify(base + '#sec-reduce')).toEqual({ kind: 'anchor', id: 'sec-reduce' });   // ("On this page" of Quarto)
        expect(classify('https://file%2B.vscode-resource.vscode-cdn.net/Users/me/ext/guide/src/#sec-scan')).toEqual({ kind: 'anchor', id: 'sec-scan' });   // (the same address written another way)
    });

    it('is to a page of the book for a link under its directory, with the path from the root of the book', () => {
        expect(classify(base + 'installation.html')).toEqual({ kind: 'page', rel: 'src/installation.html' });
        expect(classify('../index.html#sec-intro')).toEqual({ kind: 'page', rel: 'index.html#sec-intro' });
        expect(classify('https://file%2B.vscode-resource.vscode-cdn.net/Users/me/ext/guide/src/lsp.html')).toEqual({ kind: 'page', rel: 'src/lsp.html' });
    });

    it('is the first page for a link to the directory of the book (as Quarto writes the links to the introduction and the title)', () => {
        expect(classify(root)).toEqual({ kind: 'page', rel: 'index.html' });
        expect(classify('../')).toEqual({ kind: 'page', rel: 'index.html' });
        expect(classify('https://file%2B.vscode-resource.vscode-cdn.net/Users/me/ext/guide/')).toEqual({ kind: 'page', rel: 'index.html' });
    });

    it('is the web for anything else', () => {
        expect(classify('https://doi.org/10.1/x')).toEqual({ kind: 'web' });
    });

    it('is a function that stands by itself, as it is put into the page', () => {
        const alone = new Function('return ' + classifyLink.toString())();
        expect(alone(base + '#a', base + '#a', base, here, root)).toEqual({ kind: 'anchor', id: 'a' });
    });
});

describe('the sections of the guide', () => {
    it('are all in the sources of the guide: the page, and the id of the section in it', () => {
        for (const [anchor, target] of Object.entries(GUIDE_SECTIONS)) {
            const [page, id] = target.split('#');
            const source = join(GUIDE_SOURCES, page.replace(/\.html$/, '.qmd'));
            expect(existsSync(source), `${anchor}: ${source}`).toBe(true);
            if (id) expect(readFileSync(source, 'utf8'), `${anchor}: {#${id}}`).toContain(`{#${id}}`);
        }
    });

    it('are named by the anchors, with the first page for any other name', () => {
        expect(guideTarget('map')).toBe('src/programming.html#sec-map');
        expect(guideTarget('lsp')).toBe('src/lsp.html#sec-lsp');
        expect(guideTarget(undefined)).toBe('index.html');
        expect(guideTarget('nothing')).toBe('index.html');
        expect(guideTarget('toString')).toBe('index.html');
    });

    it('have an anchor for each skeleton', () => {
        for (const anchor of SKELETON_ANCHORS) expect(GUIDE_ANCHORS).toContain(anchor);
    });
});

describe('the anchors', () => {
    it('are made from the name of a skeleton, in lower case and without the dimension', () => {
        const names = ['Map', 'Reduce', 'MapReduce', 'Scan', 'MapOverlap1D', 'MapOverlap2D', 'MapPairs', 'MapPairsReduce', 'MapPool3D'];
        expect(names.map(skeletonAnchor)).toEqual(['map', 'reduce', 'mapreduce', 'scan', 'mapoverlap', 'mapoverlap', 'mappairs', 'mappairsreduce', 'mappool']);
        expect(skeletonAnchor('Banana')).toBeUndefined();
        expect(skeletonAnchor('lsp')).toBeUndefined();
        expect(skeletonAnchor(undefined)).toBeUndefined();
    });
});

describe('the hover of a skeleton', () => {
    it('names the skeleton of an instance from the detail of the outline', () => {
        expect(skeletonOfDetail('MapReduce(float, float)')).toBe('MapReduce');
        expect(skeletonOfDetail('Map(float)')).toBe('Map');
        expect(skeletonOfDetail('MapOverlap2D(int)')).toBe('MapOverlap2D');
        expect(skeletonOfDetail('template T (T, T)')).toBeUndefined();
        expect(skeletonOfDetail('Banana(int)')).toBeUndefined();
        expect(skeletonOfDetail(undefined)).toBeUndefined();
    });

    it('is for a name after skepu::', () => {
        expect(afterSkepuNamespace('\tauto dotprod = skepu::')).toBe(true);
        expect(afterSkepuNamespace('skepu:: ')).toBe(true);
        expect(afterSkepuNamespace('\tauto dotprod = ')).toBe(false);
        expect(afterSkepuNamespace('myskepu::')).toBe(false);
    });

    it('has links that are commands of the extension, with the anchor as the argument', () => {
        const md = guideHoverMarkdown('mapreduce', 'MapReduce');
        expect(md).toContain('[SkePU guide: MapReduce](command:skevu.openGuide?%5B%22mapreduce%22%5D)');
        expect(md).toContain('command:skevu.openGuideInWindow?%5B%22mapreduce%22%5D');
    });
});
