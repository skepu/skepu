// state.js — what the pieces of the designer share: the design and everything that is made from it (S), the helpers for the page, and `changed`,
// which says that the design changed. No piece imports another for this: the others register what they do about a change in `hooks`.

import { emptyDesign, sampleDesign, lower, effectiveContainers, transfers } from './model.js';
import { History } from './history.js';

export const CODE = 'skevu-designer-code';

export const STORAGE = 'skevu-designer-v1', LAYOUT = 'skevu-designer-layout';

export const $ = (selector, root = document) => root.querySelector(selector);

export const el = (tag, props = {}, ...children) => {
    const node = Object.assign(document.createElement(tag), props);
    for (const c of children) node.append(c);
    return node;
};

export const stored = (key, fallback = null) => { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } };

export const store = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private window: it is only not remembered */ } };

export const BACKEND_NAME = { cpu: 'CPU', openmp: 'OpenMP', opencl: 'OpenCL' };

// ── State ────────────────────────────────────────────────────────────────────

export function normalize(d) {
    const design = { ...emptyDesign(), ...d };
    for (const key of ['containers', 'skeletons', 'edges', 'loops']) if (!Array.isArray(design[key])) design[key] = [];
    design.positions = design.positions ?? {};
    return design;
}

export function restore() {
    try { const saved = localStorage.getItem(STORAGE); if (saved) return normalize(JSON.parse(saved)); } catch { /* private window, or not a design */ }
    return null;
}
             // { cpp, python }: what the emitters made, { text, lines, sites }; and `trace`: { text, name, events, ids, ranges } (the trace that it could have)
export const codeOptions = { userFunctions: 'functions', containerPlacement: 'grouped', skeletonPlacement: 'grouped', whitespace: 'balanced', ...stored(CODE, {}) };

export const allIds = () => [...S.design.containers, ...S.design.skeletons, ...S.design.loops, ...S.design.edges].map(x => x.id);

export const nextId = prefix => {
    const used = new Set(allIds());
    for (let n = 1; ; n++) if (!used.has(prefix + n)) return prefix + n;
};

export const find = id => S.design.containers.find(c => c.id === id) ?? S.design.skeletons.find(s => s.id === id) ?? S.design.loops.find(l => l.id === id);

export const kindOf = id => S.design.containers.some(c => c.id === id) ? 'container' : S.design.skeletons.some(s => s.id === id) ? 'skeleton' : S.design.loops.some(l => l.id === id) ? 'loop' : null;

export const effective = id => effectiveContainers(S.design).find(c => c.id === id);

/** The design is remembered by the page (and is what undo goes back to): the text of it. */
export function save() { const text = JSON.stringify(S.design); try { localStorage.setItem(STORAGE, text); } catch { /* private window: it is only not remembered */ } return text; }

export const shownTransfers = () => transfers((S.result.ok ? S.result : S.lastGood)?.program ?? { steps: [] });

export const descendantOf = (id, ancestor) => { for (let p = find(id)?.parent; p; p = find(p)?.parent) if (p === ancestor) return true; return false; };

/** A message in the menu bar, for a few seconds. */
export function say(message) {
    $('#message').textContent = message;
    clearTimeout(say.timer);
    say.timer = setTimeout(() => { $('#message').textContent = ''; }, 5000);
}

// ── The menus (those of SkeVU) ───────────────────────────────────────────────

export const setChecked = (button, on) => { const icon = $('.menu-icon', button); if (icon) icon.textContent = on ? '✓' : ''; };

export function download(name, content, type) {
    const a = el('a', { href: URL.createObjectURL(new Blob([content], { type })), download: name });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/** The state: the design, what is made from it (the result of lowering it, the programs), the pieces of the page, and what the editor is doing. */
export const S = {
    design: restore() ?? sampleDesign(),
    result: null,                // lower(design): { ok, errors, warnings, program, dataRegion }
    lastGood: null,              // the last result that had no errors
    cy: null, legend: null, source: null,                    // the graph, its legend, the view of the code
    connecting: null,            // null (not connecting), or { source: id | null }
    adding: null,                // null (not adding), or the type of node that a click on the canvas adds
    lastAdd: stored('skevu-designer-add', 'Map'),       // the type that was added last: what `A` goes back to
    inspected: null,             // what the properties were made for
    programs: null,              // { cpp, python }: what the emitters made, { text, lines, sites }; and `trace`: { text, name, events, ids, ranges } (the trace that it could have)
    /**
     * Where the selection that is being made comes from, for what follows it (as in SkeVU): 'graph' (a click on the graph, or an edit: the code
     * scrolls to what is selected; the graph stays where it is), 'code' (a click on a line, a badge or a problem: the graph pans and zooms to what
     * is selected; the code stays, it is where the pointer is) or 'restore' (the selection is put back after the graph was made again: nothing moves).
     */
    selectionFrom: 'graph',
    lastLanguage: 'cpp',         // the language of the program tab that was shown last (the bodies of functions are edited in it)
    moveKey: '',                 // the transfers that the graph shows, to see when they change
    files: { cpp: 'design.cpp', py: 'design.py' },
};
S.result = lower(S.design);
S.lastGood = S.result.ok ? S.result : null;

/** The steps of undo and redo: the designs that a person made (history.js). */
export const history = new History(JSON.stringify(S.design));

/** What each piece does when the design changed (they register it when they are loaded): rebuild, syncData, markProblems, renderCode, renderProblems, renderStatus, updateMenu, renderInspector. */
export const hooks = {};

/**
 * The design changed: check it, make the program, and show both. `structure` says that nodes or edges were added, removed or moved to a loop.
 * The change is a step of undo (history.js); `merge` joins it to the last when that was the same kind of change a moment ago (the keys of a text,
 * the nodes of a drag), and `record: false` is for undo and redo themselves.
 */
export function changed({ structure = false, inspector = false, merge = null, record = true } = {}) {
    S.result = lower(S.design);
    if (S.result.ok) S.lastGood = S.result;
    if (structure || JSON.stringify(shownTransfers()) !== S.moveKey) hooks.rebuild(); else { hooks.syncData(); hooks.markProblems(); }
    hooks.renderCode();
    hooks.renderProblems();
    hooks.renderStatus();
    if (inspector || structure) hooks.renderInspector(true);
    const text = save();                   // (after the places that the graph gave to new nodes)
    if (record) history.commit(text, { merge, window: merge === 'move' ? 400 : 1000 });
    else history.present = text;
    hooks.updateMenu();                    // (what can be undone)
}
