// test_embed_styles.js — the views that the user guide embeds (TimelineView, SourceCodeView, GraphView) are styled by the shared sheets, not by main.css, which the book does not load:
// a class that a view creates must not be styled only in main.css.
import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = f => readFileSync(join(ROOT, f), 'utf8');

/** The classes in the selectors of a style sheet. */
function styledClasses(css) {
    const selectors = css.replace(/\/\*[\s\S]*?\*\//g, '').match(/[^{}]+(?=\{)/g) || [];
    return new Set(selectors.join(' ').match(/\.[A-Za-z_][\w-]*/g).map(c => c.slice(1)));
}

/** The classes that a script puts on elements. */
function createdClasses(js) {
    const found = new Set();
    const add = list => list.split(/\s+/).filter(Boolean).forEach(c => found.add(c));
    for (const m of js.matchAll(/class(?:Name)?\s*[=:]\s*["'`]([^"'`$]+)["'`]/g)) add(m[1]);
    for (const m of js.matchAll(/class="([\w\- ]+)/g)) add(m[1]);
    for (const m of js.matchAll(/classList\.(?:add|toggle)\(\s*'([^']+)'/g)) add(m[1]);
    return found;
}

// states of the page, not elements of a view
const PAGE_STATES = new Set(['dark-mode']);

describe('the sheets of the embedded views', () => {
    const views = ['script/TimelineView.js', 'script/SourceCodeView.js', 'script/GraphView.js', 'script/TraceDataProvider.js', 'script/badges.js']
        .concat(readdirSync(join(ROOT, 'script/timeline')).map(f => 'script/timeline/' + f));
    const main = styledClasses(read('style/main.css'));
    const shared = new Set([...styledClasses(read('style/timeline.css')), ...styledClasses(read('style/source.css'))]);

    it('style every class of the views that main.css styles', () => {
        const missing = [];
        for (const file of views)
            for (const c of createdClasses(read(file)))
                if (main.has(c) && !shared.has(c) && !PAGE_STATES.has(c)) missing.push(`${c} (${file})`);
        expect(missing).toEqual([]);
    });

    it('find classes at all', () => {
        expect(createdClasses(read('script/timeline/draw.js')).has('gantt-bar')).toBe(true);
    });
});
