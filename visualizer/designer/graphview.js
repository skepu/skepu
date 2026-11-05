// graphview.js — the graph of the design: elements, style, legend, hover, selection and pointer handling, using the style of SkeVU (script/).

import { PATTERNS, hasScalar } from './model.js';
import { anonymousLabel, isAnonymous } from './codegen.js';
import { TraceDataProvider } from '../script/TraceDataProvider.js';
import { appSettings } from '../script/settings_model.js';
import { buildGraphStyle } from '../script/graph_style.js';
import { PAN_ZOOM_OPTIONS, bindPanZoom, ZOOM_FIT_PAD, roomAround, animateFit } from '../script/panzoom.js';
import { readAppearance } from '../script/appearance.js';
import { LegendView } from '../script/LegendView.js';
import { showTooltip, hideTooltip, positionTooltip, nodeTooltipText, edgeTooltipText } from '../script/tooltip.js';
import { highlightSelection, bindHover } from '../script/graph_interaction.js';
import { $, BACKEND_NAME, S, changed, codeOptions, effective, find, hooks, kindOf, shownTransfers, stored } from './state.js';
import { markLines } from './codeview.js';
import { renderInspector } from './inspector.js';
import { updateMenu } from './actions.js';

// ── The view object the SkeVU style and legend read ──────────────────────────

export const darkNow = () => document.documentElement.classList.contains('dark-mode');

export const settings = appSettings({ 'show_regions': true, 'node-color-call': stored('skevu-designer-color', 'pattern') });

export const provider = new TraceDataProvider({ fetchGraph: () => Promise.reject(new Error('the designer has no trace')), settings,
                                         colorSpectrum: readAppearance().colorSpectrum });

export const view = {
    settings, _opaqueNodes: [], _groupedEdgeReps: new Set(), _provider: provider,
    get _darkMode() { return darkNow(); },
    _highlightColor: () => darkNow() ? '#e8720c' : '#0055cc',
    _spectrumColor: t => provider.spectrumColor(t),
    _spectrumColorDark: (t, f) => provider.spectrumColorDark(t, f),
    _normProp: (key, ele, minmax) => provider.normFromEle(key, ele, minmax),
    hasElements: selector => !!S.cy && S.cy.$(selector).length > 0,
    // backends or patterns of the calls, for the legend
    callCategories: mode => [...new Set(S.design.skeletons.filter(s => !PATTERNS[s.pattern]?.host).map(s => mode === 'backend' ? BACKEND_NAME[s.backend] : s.pattern).filter(Boolean))],
};

// ── The graph ────────────────────────────────────────────────────────────────

export const midpoint = (a, b) => {
    const pa = S.design.positions[a] ?? { x: 0, y: 0 }, pb = S.design.positions[b] ?? pa;
    return { x: (pa.x + pb.x) / 2, y: (pa.y + pb.y) / 2 };
};

export const iterationsOf = l => l.scope ? 1 : Math.max(1, +l.count || 1);

/** The region of a data node: that of its writer. */
export const regionOfData = id => { const r = S.result.dataRegion?.[id]; return r && S.design.loops.some(l => l.id === r) ? r : undefined; };

export const isEmpty = id => !S.design.skeletons.some(s => s.parent === id) && !S.design.loops.some(l => l.parent === id);

export const loopDepth = id => { let d = 0; for (let l = S.design.loops.find(x => x.id === find(id)?.parent); l; l = S.design.loops.find(x => x.id === l.parent)) d++; return d; };

/** The edge label: the data carried (V or H for vertical and horizontal edges). */
const edgeLabel = e => {
    if (e.role === 'vertical') return 'V';
    if (e.role === 'horizontal') return 'H';
    const data = e.role === 'out' ? find(e.to) : find(e.from);
    return data?.label ?? '';
};

/** The graph label of a skeleton: its name, or the call-site name if anonymous (once the program exists). */
export const labelOf = s => isAnonymous(s, codeOptions) ? anonymousLabel(s, (S.result.ok ? S.result : S.lastGood)?.program.calls.find(c => c.skeleton.id === s.id), S.files?.cpp ?? 'design.cpp', S.programs?.cpp.sites.find(x => x.id === s.id)?.line) : s.label;

/** The id of the derived scalar node of a Reduce or MapReduce. */
export const scalarOf = id => `r:${id}`;

export function elements() {
    view._groupedEdgeReps.clear();
    const nodes = [];
    // unwritten data is an allocation; written data is a data node
    const written = new Set(S.design.edges.filter(e => e.role === 'out').map(e => e.to));
    // updates carry their version number (none for a single node)
    for (const c of S.design.containers) {
        const v = S.result.versions?.[c.id] ?? { version: 1, count: 1 };
        nodes.push({ data: { id: c.id, type: c.kind === 'PRNG' || !written.has(c.id) ? 'allocation' : 'container_update', is_prng: c.kind === 'PRNG' || undefined, label: c.label, version: v.version, version_count: v.count, iteration_count: 1,
                             parent: regionOfData(c.id) },
                    position: S.design.positions[c.id] });
    }
    for (const s of S.design.skeletons)
        nodes.push({ data: { id: s.id, type: PATTERNS[s.pattern]?.host ? 'external' : 'skeleton_call', label: labelOf(s), pattern: s.pattern, backend: BACKEND_NAME[s.backend] ?? '', iteration_count: 1,
                             parent: s.parent || undefined }, position: S.design.positions[s.id] });
    // the result of a Reduce or MapReduce is a scalar node, read by the I/O block that prints it
    const scalars = S.design.skeletons.filter(s => hasScalar(S.design, s));
    for (const s of scalars)
        nodes.push({ data: { id: scalarOf(s.id), type: 'scalar', label: 'scalar', scalar: true, parent: s.parent || undefined },
                    position: S.design.positions[scalarOf(s.id)] ?? { x: (S.design.positions[s.id]?.x ?? 0) + 60, y: (S.design.positions[s.id]?.y ?? 0) + 60 } });
    for (const l of S.design.loops)
        nodes.push({ data: { id: l.id, type: 'region', label: l.label, iteration_count: iterationsOf(l), region_depth: loopDepth(l.id), parent: l.parent || undefined },
                    ...(isEmpty(l.id) ? { position: S.design.positions[l.id] } : {}) });       // an empty region has its own position
    // transfers are derived from the design, between the data and the call that needs it moved
    const moves = shownTransfers();
    S.moveKey = JSON.stringify(moves);
    const move = (from, to) => moves.find(m => m.container === from && m.consumers.includes(to));       // a transfer is the source for each call reading its copy
    const edges = scalars.map(s => ({ data: { id: `${s.id}~scalar`, source: s.id, target: scalarOf(s.id), type: 'forward-dep', access_mode: 'scalar', derived: true } }));
    const mode = e => e.role === 'proxy' || e.role === 'read' ? 'proxy' : e.role === 'scalar' ? 'scalar' : e.role === 'random' ? undefined : 'elwise';
    const made = new Set(), kept = new Set();
    for (const e of S.design.edges.filter(e => find(e.from) && find(e.to))) {
        const label = edgeLabel(e);
        if (e.role === 'vertical' || e.role === 'horizontal') view._groupedEdgeReps.add(e.id);
        const m = e.role === 'out' || e.role === 'random' ? null : move(e.from, e.to);
        const from = e.role === 'scalar' ? scalarOf(e.from) : e.from;        // a result is read from its scalar node
        if (!m) { edges.push({ data: { id: e.id, source: from, target: e.to, type: e.role === 'random' ? 'prng-dep' : 'forward-dep', label, access_mode: mode(e) } }); continue; }
        const t = `t:${m.container}:${m.consumer}`;
        if (!made.has(t)) {
            made.add(t);
            // a transfer starts between its ends, then keeps its position
            kept.add(t);
            nodes.push({ data: { id: t, type: 'transfer', label: find(m.container).label, direction: m.direction, transfer: true, parent: find(m.consumer)?.parent || undefined }, position: S.design.positions[t] ?? midpoint(m.container, m.consumer) });
            edges.push({ data: { id: `${e.id}~in`, source: e.from, target: t, type: 'forward-dep', access_mode: mode(e), derived: true } });
        }
        edges.push({ data: { id: e.id, source: t, target: e.to, type: 'forward-dep', label, access_mode: mode(e) } });     // keeps the design edge's id
    }
    for (const id of Object.keys(S.design.positions)) if (id.startsWith('t:') && !kept.has(id) || id.startsWith('r:') && !scalars.some(s => scalarOf(s.id) === id)) delete S.design.positions[id];       // drop positions of vanished transfers and scalars
    return [...nodes, ...edges];
}

export function stylesheet() {
    return [
        // Problems are underlays, first in the list: selection and hover borders stay visible, and the legend mark (also an underlay) draws above them
        { selector: 'node.has-warning', style: { 'underlay-color': '#e8b400', 'underlay-padding': 6, 'underlay-opacity': 0.7, 'underlay-shape': 'round-rectangle' } },
        { selector: 'node.has-error', style: { 'underlay-color': '#e53935', 'underlay-padding': 6, 'underlay-opacity': 0.7, 'underlay-shape': 'round-rectangle' } },
        { selector: 'edge.has-warning', style: { 'underlay-color': '#e8b400', 'underlay-padding': 4, 'underlay-opacity': 0.7 } },
        { selector: 'edge.has-error', style: { 'underlay-color': '#e53935', 'underlay-padding': 4, 'underlay-opacity': 0.7 } },
        ...buildGraphStyle(view, {}),
        { selector: 'node[type="region"]', style: { 'text-valign': 'top', 'text-halign': 'center' } },
        { selector: 'node[type="region"]:childless', style: { width: 110, height: 60, 'text-valign': 'center' } },
        { selector: '.connect-source', style: { 'border-width': 5, 'border-color': view._highlightColor() } },
    ];
}

export function layoutPositions() {
    // new nodes go below all others (a new skeleton is the last call)
    let y = Math.max(-110, ...Object.values(S.design.positions).map(p => p.y)) + 110;
    for (const n of [...S.design.containers, ...S.design.skeletons, ...S.design.loops.filter(l => isEmpty(l.id))])
        if (!S.design.positions[n.id]) { S.design.positions[n.id] = { x: kindOf(n.id) === 'container' ? -120 : 40, y }; y += 110; }
}

export function rebuild() {
    const keep = S.cy.$(':selected').map(n => n.id());
    layoutPositions();
    S.selectionFrom = 'restore';
    S.cy.batch(() => {
        S.cy.elements().remove();
        S.cy.add(elements());
        keep.forEach(id => S.cy.$id(id).select());
    });
    S.selectionFrom = 'graph';
    markProblems();
    renderLegend();
}

/** Updates node and edge labels after a change that keeps the structure. */
export function syncData() {
    S.cy.batch(() => {
        for (const c of S.design.containers) { const v = S.result.versions?.[c.id] ?? { version: 1, count: 1 }; S.cy.$id(c.id).data({ label: c.label, version: v.version, version_count: v.count }); }
        for (const s of S.design.skeletons) S.cy.$id(s.id).data({ label: labelOf(s), pattern: s.pattern, backend: BACKEND_NAME[s.backend] ?? '' });
        for (const l of S.design.loops) S.cy.$id(l.id).data({ label: l.label, iteration_count: iterationsOf(l) });
        for (const e of S.design.edges) S.cy.$id(e.id).data('label', edgeLabel(e));
    });
}

export function markProblems() {
    S.cy.elements().removeClass('has-error has-warning');
    for (const w of S.result.warnings) S.cy.$id(w.id).addClass('has-warning');
    for (const e of S.result.errors) S.cy.$id(e.id).addClass('has-error');
}

export function storePositions() {
    S.cy.nodes('[type != "region"]').forEach(n => { S.design.positions[n.id()] = { ...n.position() }; });
    S.cy.nodes('[type = "region"]:childless').forEach(n => { S.design.positions[n.id()] = { ...n.position() }; });
}

// ── Legend ───────────────────────────────────────────────────────────────────

export function renderLegend() { S.legend.render(darkNow()); }

/**
 * Marks the code lines of the selected nodes in the shown file. With `scroll`, the code scrolls to the first (for a call: its call line).
 */
/** The tooltip data of a node: label, kind, size, backend and C++ location. */
export function tooltipData(ele) {
    const id = ele.id(), node = find(id), line = S.programs?.cpp.sites.find(s => s.id === id)?.line;
    const declared = (S.programs?.cpp.lines.findIndex(ids => ids.includes(id)) ?? -1) + 1;
    const place = { file: S.files.cpp, line: line ?? (declared || undefined) };
    switch (kindOf(id)) {
        case 'container': { const c = effective(id); return node.kind === 'PRNG' ? { label: node.label, pattern: 'random stream', ...place } : { label: c.label, pattern: `${c.kind}<${c.type}>`, elements: c.size, ...place }; }
        case 'skeleton': return { label: labelOf(node), pattern: PATTERNS[node.pattern]?.title ?? node.pattern, backend: BACKEND_NAME[node.backend] ?? '', ...place };
        case 'loop': return { label: node.label, pattern: node.scope ? 'region' : `loop, ${node.count} iterations` };
        default: return { label: ele.data('label'), ...(ele.data('direction') ? { pattern: ele.data('direction') } : {}) };
    }
}

export function focusGraph() {
    let eles = S.cy.$(':selected').nodes();
    S.cy.$(':selected').edges().forEach(e => { eles = eles.union(e.connectedNodes()); });
    if (!eles.length) return;
    animateFit(S.cy, eles, eles.length === 1 && eles[0].data('type') !== 'region' ? roomAround(S.cy) : ZOOM_FIT_PAD);
}

export function initGraph() {
    S.cy = window.cytoscape({ container: $('#cy'), elements: [], style: stylesheet(), ...PAN_ZOOM_OPTIONS, boxSelectionEnabled: true });
    bindPanZoom(S.cy, $('#cy'));                              // wheel pans, pinch zooms
    S.legend = new LegendView($('#legend'), view, { labels: { container_update: 'Data' }, hoverGraph: () => S.cy, clearHighlight: () => S.cy.elements().unselect(),
        onSelect: callback => S.cy.on('select', () => { if (S.selectionFrom !== 'restore') callback(); }), onSpectrumClick: (x, y) => hooks.spectrumMenu?.(x, y), onColorByClick: (x, y) => hooks.colorByMenu?.(x, y) });
    S.legend.settings = settings;
    layoutPositions();
    S.cy.add(elements());
    S.cy.fit(undefined, 40);

    S.cy.on('select unselect', () => {
        highlightSelection(S.cy, S.cy.nodes(':selected'), S.cy.edges(':selected'));
        markLines({ scroll: S.selectionFrom === 'graph' });
        if (S.selectionFrom === 'code') focusGraph();
        renderInspector();
        updateMenu();
    });
    // hover: tooltip, legend entry, code badge (graph_interaction.js)
    bindHover(S.cy, $('#cy'), {
        over: (ele, event) => {
            const text = ele.isNode() ? nodeTooltipText(tooltipData(ele)) : edgeTooltipText(ele.data());
            if (text && event.originalEvent) showTooltip(text, event.originalEvent.clientX, event.originalEvent.clientY);
            S.legend.highlightFor(ele);
            if (ele.isNode()) S.source.highlightBadgeNode(ele.id());
        },
        move: event => event.originalEvent && positionTooltip(event.originalEvent.clientX, event.originalEvent.clientY),
        out: () => { hideTooltip(); S.legend.clearHighlight(); S.source.clearBadgeHighlight(); },
        reset: hideTooltip,
        leave: () => { hideTooltip(); S.legend.clearHighlight(); S.source.clearBadgeHighlight(); },
    });
    $('#cy').addEventListener('contextmenu', e => e.preventDefault());                // the context menus are the designer's (menus.js)
    S.cy.on('dragfree', 'node', () => { storePositions(); changed({ merge: 'move' }); });          // one undo step per drag
    // fit once the page is laid out
    requestAnimationFrame(() => { S.cy.resize(); S.cy.fit(undefined, 40); });
    new ResizeObserver(() => S.cy.resize()).observe($('#cy'));
}

hooks.rebuild = rebuild;
hooks.syncData = syncData;
hooks.markProblems = markProblems;
