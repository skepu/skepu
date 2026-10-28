// state.js — shared designer state: the design and what is derived from it (S), page helpers, and `changed`. Modules register their reactions in `hooks`.

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
export const codeOptions = { userFunctions: 'functions', skeletonInstancing: 'named', containerPlacement: 'grouped', skeletonPlacement: 'grouped', whitespace: 'balanced', ...stored(CODE, {}) };

export const allIds = () => [...S.design.containers, ...S.design.skeletons, ...S.design.loops, ...S.design.edges].map(x => x.id);

export const nextId = prefix => {
    const used = new Set(allIds());
    for (let n = 1; ; n++) if (!used.has(prefix + n)) return prefix + n;
};

export const find = id => S.design.containers.find(c => c.id === id) ?? S.design.skeletons.find(s => s.id === id) ?? S.design.loops.find(l => l.id === id);

export const kindOf = id => S.design.containers.some(c => c.id === id) ? 'container' : S.design.skeletons.some(s => s.id === id) ? 'skeleton' : S.design.loops.some(l => l.id === id) ? 'loop' : null;

export const effective = id => effectiveContainers(S.design).find(c => c.id === id);

/** Persists the design (JSON) in the page storage. */
export function save() { const text = JSON.stringify(S.design); try { localStorage.setItem(STORAGE, text); } catch { /* private window: it is only not remembered */ } return text; }

export const shownTransfers = () => transfers((S.result.ok ? S.result : S.lastGood)?.program ?? { steps: [], containers: [], calls: [] });

export const descendantOf = (id, ancestor) => { for (let p = find(id)?.parent; p; p = find(p)?.parent) if (p === ancestor) return true; return false; };

/** Shows a message in the menu bar for a few seconds. */
export function say(message) {
    $('#message').textContent = message;
    clearTimeout(say.timer);
    say.timer = setTimeout(() => { $('#message').textContent = ''; }, 5000);
}

// ── Menus ────────────────────────────────────────────────────────────────────

export const setChecked = (button, on) => { const icon = $('.menu-icon', button); if (icon) icon.textContent = on ? '✓' : ''; };

export function download(name, content, type) {
    const a = el('a', { href: URL.createObjectURL(new Blob([content], { type })), download: name });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/** The state: the design, derived data (lowering result, programs), page objects and editor mode. */
export const S = {
    design: restore() ?? sampleDesign(),
    result: null,                // lower(design): { ok, errors, warnings, program, dataRegion }
    lastGood: null,              // the last error-free result
    cy: null, legend: null, source: null,                    // graph, legend, code view
    connecting: null,            // null, or { source: id | null }
    adding: null,                // null, or the type a canvas click adds
    lastAdd: stored('skevu-designer-add', 'Map'),       // the last added type (what `A` returns to)
    inspected: null,             // what the inspector shows
    programs: null,              // { cpp, python }: what the emitters made, { text, lines, sites }; and `trace`: { text, name, events, ids, ranges } (the trace that it could have)
    /**
     * The origin of the current selection: 'graph' (click or edit: the code scrolls), 'code' (line, badge or problem: the graph pans and zooms) or
     * 'restore' (reselection after a rebuild: nothing moves).
     */
    selectionFrom: 'graph',
    lastLanguage: 'cpp',         // the language of the last shown program tab (function bodies are edited in it)
    moveKey: '',                 // the transfers shown, to detect changes
    files: { cpp: 'design.cpp', py: 'design.py' },
};
S.result = lower(S.design);
S.lastGood = S.result.ok ? S.result : null;

/** Undo and redo steps (history.js). */
export const history = new History(JSON.stringify(S.design));

/** The reactions to a design change, registered by the modules: rebuild, syncData, markProblems, renderCode, renderProblems, renderStatus, updateMenu, renderInspector. */
export const hooks = {};

/**
 * The design changed: lower it, make the programs and refresh the views. `structure`: nodes or edges changed. The change is an undo step (history.js);
 * `merge` joins it to a recent change of the same kind, `record: false` is for undo and redo.
 */
export function changed({ structure = false, inspector = false, merge = null, record = true } = {}) {
    S.result = lower(S.design);
    if (S.result.ok) S.lastGood = S.result;
    if (structure || JSON.stringify(shownTransfers()) !== S.moveKey) hooks.rebuild(); else { hooks.syncData(); hooks.markProblems(); }
    hooks.renderCode();
    hooks.renderProblems();
    hooks.renderStatus();
    if (inspector || structure) hooks.renderInspector(true);
    const text = save();                   // after the graph has placed new nodes
    if (record) history.commit(text, { merge, window: merge === 'move' ? 400 : 1000 });
    else history.present = text;
    hooks.updateMenu();                    // undo and redo availability
}
