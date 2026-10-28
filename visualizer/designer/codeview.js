// codeview.js — the programs (C++, Python) and the synthesized trace in the source view of SkeVU: tabs, selected-node lines, badges, line clicks.

import { identifier } from './model.js';
import { EMITTERS } from './codegen.js';
import { synthesizeTrace } from './trace.js';
import { computeTraceLineRanges } from '../script/backend/trace.js';
import { readAppearance, writeAppearance } from '../script/appearance.js';
import { SourceCodeView } from '../script/SourceCodeView.js';
import { $, S, codeOptions, download, hooks, restore, say } from './state.js';
import { renderInspector, text } from './inspector.js';
import { selectById } from './actions.js';

/** The shown file: 'cpp', 'py' or 'json' (the trace). */
export const activeFile = () => { const f = $('#cpp_tab_panes .cpp-tab-pane.active')?.dataset.filename ?? ''; return f.endsWith('.py') ? 'py' : f.endsWith('.json') ? 'json' : 'cpp'; };

/** The language of the last shown program (while the trace is shown, the one before). */
export const activeLanguage = () => { const f = activeFile(); if (f !== 'json') S.lastLanguage = f; return S.lastLanguage; };

export const TAB = { cpp: 0, py: 1, json: 2 };

/**
 * The programs and the trace depend on the design without positions, the call order and the code options. If none changed (a node was only moved),
 * the files are kept: the code view keeps its scroll.
 */
let madeFor = null;
const programKey = shown => JSON.stringify([S.design, shown?.program.calls.map(c => c.skeleton.id), S.result.ok, codeOptions], (key, value) => key === 'positions' ? undefined : value);

export function renderCode() {
    const shown = S.result.ok ? S.result : S.lastGood;
    const key = programKey(shown);
    if (key === madeFor) return;
    madeFor = key;
    const base = identifier(S.design.name, new Set(), 'design');
    S.files = { cpp: `${base}.cpp`, py: `${base}.py` };
    S.programs = shown ? { cpp: EMITTERS.cpp(shown.program, codeOptions), python: EMITTERS.python(shown.program, codeOptions) } : null;
    if (S.programs) {
        const made = synthesizeTrace(shown.program, { dir: '/designer', file: S.files.cpp, lines: S.programs.cpp.lines, sites: S.programs.cpp.sites, containerPlacement: codeOptions.containerPlacement, skeletonInstancing: codeOptions.skeletonInstancing });
        const text = JSON.stringify(made.events, null, 2) + '\n';
        S.programs.trace = { ...made, text, ranges: computeTraceLineRanges(text),
                           name: made.truncated ? `trace_${base}_first_${made.truncated}_iterations.json` : `trace_${base}.json` };
    }

    if (S.cy) hooks.syncData();                                   // anonymous skeleton names include the call line
    const shownFile = activeFile();
    activeLanguage();
    const top = $('#cpp_tab_panes .cpp-tab-pane.active')?.scrollTop ?? 0;
    S.source.rebuild(S.programs
        ? { [S.files.cpp]: S.programs.cpp.text, [S.files.py]: S.programs.python.text }
        : { [S.files.cpp]: '// No program: the design has errors.\n',
            [S.files.py]: '# No program: the design has errors.\n' },
        S.programs ? S.programs.trace.text : null, S.programs ? S.programs.trace.name : '', S.programs ? S.programs.trace.ranges : []);
    if (shownFile !== 'cpp' && $('#cpp_tab_bar').children[TAB[shownFile]]) S.source.switchTab($('#cpp_tab_bar').children[TAB[shownFile]], TAB[shownFile]);
    const restore = () => { const pane = $('#cpp_tab_panes .cpp-tab-pane.active'); if (pane) pane.scrollTop = top; markLines(); };
    restore();
    setTimeout(restore, 40);

    // gutter badges: run counts and code paths
    if (S.programs) {
        const nodes = [];
        for (const [file, program] of [[S.files.cpp, S.programs.cpp], [S.files.py, S.programs.python]])
            for (const s of program.sites)
                nodes.push({ data: { id: s.id, type: s.type, file, line: s.line,
                                     total_orders: Array.from({ length: Math.min(999, Math.max(1, s.count)) }, (_, i) => s.order + i / 1000) } });
        // and a badge for each trace event
        S.programs.trace.ids.forEach((ids, i) => nodes.push({ data: { id: `${ids[0]}#trace`, type: 'trace', trace_index: i } }));
        S.source.placeBadges(nodes);
    }
}

export const linesOf = (program, ids) => (program?.lines ?? []).flatMap((line, i) => line.some(id => ids.has(id)) ? [i + 1] : []);

/** The trace lines of the events about the nodes with these ids. */
export const traceLines = ids => {
    const trace = S.programs?.trace, lines = [];
    trace?.ids.forEach((nodes, i) => { if (nodes.some(id => ids.has(id))) for (let l = trace.ranges[i]?.[0]; l <= trace.ranges[i]?.[1]; l++) lines.push(l); });
    return lines;
};

/** The ids of the selected nodes and of the ends of the selected edges. */
export function selectedIds() {
    const ids = new Set();
    const callOf = id => id.startsWith('r:') ? id.slice(2) : id;               // a scalar maps to its call
    S.cy?.$(':selected').forEach(x => {
        if (x.isEdge()) { ids.add(callOf(x.data('source'))); ids.add(callOf(x.data('target'))); } else ids.add(callOf(x.id()));
    });
    return ids;
}

export function markLines({ scroll = false } = {}) {
    const ids = selectedIds();
    const json = activeFile() === 'json';
    const program = activeLanguage() === 'py' ? S.programs?.python : S.programs?.cpp;
    const site = json ? null : program?.sites.find(s => ids.has(s.id));
    S.source.highlightLines(json ? traceLines(ids) : linesOf(program, ids), { scroll, target: site?.line });
}

/** Creates the code view and binds tabs, line clicks and buttons. */
export function initCode() {
    S.source = new SourceCodeView({ tabBarEl: $('#cpp_tab_bar'), tabPanesEl: $('#cpp_tab_panes'), hljsThemeLinkEl: $('#hljs-theme'), initialTheme: readAppearance().hljsTheme,
                                    onThemeChange: theme => writeAppearance({ hljsTheme: theme }),
                                    onBadgeClick: id => selectById(id.replace(/#trace$/, ''), 'code') });
    S.source.restoreTheme(readAppearance().hljsTheme);

    // a click on a line selects its node
    $('#cpp_tab_bar').addEventListener('click', () => setTimeout(() => { markLines({ scroll: true }); if (S.inspected && S.inspected !== 'none' && S.inspected !== 'many') renderInspector(true); }, 0));
    $('#cpp_tab_panes').addEventListener('click', e => {
        const code = e.target.closest('.hljs-ln-code[data-line-number]');
        if (!code) return;
        const line = +code.dataset.lineNumber;
        const trace = S.programs?.trace;
        const id = activeFile() === 'json' ? trace?.ids[trace.ranges.findIndex(r => r[0] <= line && line <= r[1])]?.[0]
            : (activeLanguage() === 'py' ? S.programs?.python : S.programs?.cpp)?.lines[line - 1]?.[0];
        if (id) selectById(id, 'code');
    });
    // name and text of the shown tab
    const shownText = () => { const f = activeFile(); return { name: f === 'json' ? S.programs?.trace.name : S.files[f], text: (f === 'py' ? S.programs?.python : f === 'json' ? S.programs?.trace : S.programs?.cpp)?.text ?? '' }; };
    $('#copy-code').onclick = async () => {
        try { await navigator.clipboard.writeText(shownText().text); say('Copied.'); } catch { say('Copy failed.'); }
    };
    $('#download-code').onclick = () => { const { name, text } = shownText(); download(name ?? 'design.txt', text, activeFile() === 'json' ? 'application/json' : 'text/plain'); };
}

hooks.renderCode = renderCode;
