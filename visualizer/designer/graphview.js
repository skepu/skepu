// graphview.js — the graph of the design, drawn with the style, the legend, the hover and the selection of SkeVU (script/): what the nodes and
// edges are made from the design, how they look, where they are, and what the pointer does to them.

import { PATTERNS } from './model.js';
import { TraceDataProvider } from '../script/TraceDataProvider.js';
import { buildGraphStyle } from '../script/graph_style.js';
import { PAN_ZOOM_OPTIONS, bindPanZoom, ZOOM_FIT_PAD, roomAround, animateFit } from '../script/panzoom.js';
import { readAppearance } from '../script/appearance.js';
import { LegendView } from '../script/LegendView.js';
import { showTooltip, hideTooltip, positionTooltip, nodeTooltipText, edgeTooltipText } from '../script/tooltip.js';
import { highlightSelection, bindHover } from '../script/graph_interaction.js';
import { $, BACKEND_NAME, S, changed, effective, find, hooks, kindOf, shownTransfers, stored } from './state.js';
import { markLines } from './codeview.js';
import { renderInspector } from './inspector.js';
import { updateMenu } from './actions.js';

// ── The view of the graph: what GraphView is to the style of SkeVU ──────────

export const darkNow = () => document.documentElement.classList.contains('dark-mode');

export const settings = { 'node-labels': true, 'edge-labels': false, 'show_regions': true, 'node-color-call': stored('skevu-designer-color', 'pattern'), 'node-size-call': 'fixed',
                   'edge-width': 'fixed', 'edge-opacity': 'fixed' };

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
    // the backends or patterns that the calls have, for the legend
    callCategories: mode => [...new Set(S.design.skeletons.filter(s => !PATTERNS[s.pattern]?.host).map(s => mode === 'backend' ? BACKEND_NAME[s.backend] : s.pattern).filter(Boolean))],
};

// ── The graph ────────────────────────────────────────────────────────────────

export const midpoint = (a, b) => {
    const pa = S.design.positions[a] ?? { x: 0, y: 0 }, pb = S.design.positions[b] ?? pa;
    return { x: (pa.x + pb.x) / 2, y: (pa.y + pb.y) / 2 };
};

export const iterationsOf = l => l.scope ? 1 : Math.max(1, +l.count || 1);

/** The region that the data is in: that of the call that writes it. */
export const regionOfData = id => { const r = S.result.dataRegion?.[id]; return r && S.design.loops.some(l => l.id === r) ? r : undefined; };

/** A region with nothing in it. */
export const isEmpty = id => !S.design.skeletons.some(s => s.parent === id) && !S.design.loops.some(l => l.parent === id);

export const loopDepth = id => { let d = 0; for (let l = S.design.loops.find(x => x.id === find(id)?.parent); l; l = S.design.loops.find(x => x.id === l.parent)) d++; return d; };

export function elements() {
    view._groupedEdgeReps.clear();
    const nodes = [];
    // data that nothing writes is an allocation (its first use is what it is made for); data that is written is a data node
    const written = new Set(S.design.edges.filter(e => e.role === 'out').map(e => e.to));
    // (an update has the number of its version among the updates of its object, as in SkeVU: no number for the only one)
    for (const c of S.design.containers) {
        const v = S.result.versions?.[c.id] ?? { version: 1, count: 1 };
        nodes.push({ data: { id: c.id, type: c.kind === 'PRNG' || !written.has(c.id) ? 'allocation' : 'container_update', is_prng: c.kind === 'PRNG' || undefined, label: c.label, version: v.version, version_count: v.count, iteration_count: 1,
                             parent: regionOfData(c.id) },        // (the data that a call writes is in the region of the call)
                    position: S.design.positions[c.id] });
    }
    for (const s of S.design.skeletons)
        nodes.push({ data: { id: s.id, type: PATTERNS[s.pattern]?.host ? 'external' : 'skeleton_call', label: s.label, pattern: s.pattern, backend: BACKEND_NAME[s.backend] ?? '', iteration_count: 1,
                             parent: s.parent || undefined }, position: S.design.positions[s.id] });
    for (const l of S.design.loops)
        nodes.push({ data: { id: l.id, type: 'region', label: l.label, iteration_count: iterationsOf(l), region_depth: loopDepth(l.id), parent: l.parent || undefined },
                    ...(isEmpty(l.id) ? { position: S.design.positions[l.id] } : {}) });       // (a region with nothing in it has a place of its own)
    // a transfer is between the data and the call that needs it moved: it is made from the design, not part of it
    const moves = shownTransfers();
    S.moveKey = JSON.stringify(moves);
    const move = (from, to) => moves.find(m => m.container === from && m.consumers.includes(to));       // (a transfer is the source of every call that reads its copy)
    const edges = [];
    const mode = e => e.role === 'proxy' || e.role === 'read' ? 'proxy' : e.role === 'scalar' ? 'scalar' : e.role === 'out' || e.role === 'random' ? undefined : 'elwise';
    const made = new Set(), kept = new Set();
    for (const e of S.design.edges.filter(e => find(e.from) && find(e.to))) {
        const label = e.role === 'vertical' ? 'V' : e.role === 'horizontal' ? 'H' : '';
        if (label) view._groupedEdgeReps.add(e.id);
        const m = e.role === 'out' || e.role === 'random' ? null : move(e.from, e.to);
        if (!m) { edges.push({ data: { id: e.id, source: e.from, target: e.to, type: e.role === 'random' ? 'prng-dep' : 'forward-dep', label, access_mode: mode(e) } }); continue; }
        const t = `t:${m.container}:${m.consumer}`;
        if (!made.has(t)) {
            made.add(t);
            // (a transfer is a node like the others, as in SkeVU: it starts between its ends, and is where it was put or laid out after that)
            kept.add(t);
            nodes.push({ data: { id: t, type: 'transfer', label: find(m.container).label, direction: m.direction, transfer: true, parent: find(m.consumer)?.parent || undefined }, position: S.design.positions[t] ?? midpoint(m.container, m.consumer) });
            edges.push({ data: { id: `${e.id}~in`, source: e.from, target: t, type: 'forward-dep', access_mode: mode(e), derived: true } });
        }
        edges.push({ data: { id: e.id, source: t, target: e.to, type: 'forward-dep', label, access_mode: mode(e) } });     // (keeps the id of the edge: it is the one to select)
    }
    for (const id of Object.keys(S.design.positions)) if (id.startsWith('t:') && !kept.has(id)) delete S.design.positions[id];       // (the transfers that are gone)
    return [...nodes, ...edges];
}

export function stylesheet() {
    return [
        ...buildGraphStyle(view, {}),
        { selector: 'node[type="region"]', style: { 'text-valign': 'top', 'text-halign': 'center' } },
        { selector: 'node[type="region"]:childless', style: { width: 110, height: 60, 'text-valign': 'center' } },
        { selector: '.has-warning', style: { 'border-color': '#d9a400', 'border-width': 4 } },
        { selector: '.has-error', style: { 'border-color': '#d33', 'border-width': 4 } },
        { selector: 'edge.has-error', style: { 'line-color': '#d33', 'target-arrow-color': '#d33' } },
        { selector: '.connect-source', style: { 'border-width': 5, 'border-color': view._highlightColor() } },
    ];
}

export function layoutPositions() {
    // new nodes have no place: below everything that there is, so that a new skeleton is the last call
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

/** Properties that changed but not the structure: the text of the nodes. */
export function syncData() {
    S.cy.batch(() => {
        for (const c of S.design.containers) { const v = S.result.versions?.[c.id] ?? { version: 1, count: 1 }; S.cy.$id(c.id).data({ label: c.label, version: v.version, version_count: v.count }); }
        for (const s of S.design.skeletons) S.cy.$id(s.id).data({ label: s.label, pattern: s.pattern, backend: BACKEND_NAME[s.backend] ?? '' });
        for (const l of S.design.loops) S.cy.$id(l.id).data({ label: l.label, iteration_count: iterationsOf(l) });
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

// ── The legend: that of SkeVU, with the colours of what the designer colours by ──

/** The legend is SkeVU's; the designer adds what only it has: a random stream, and the name of the data. */
export function renderLegend() { S.legend.render(darkNow()); }

/**
 * Marks the lines that are for the selected nodes (and the ends of the selected edges), in the file that is shown. With `scroll`, the code
 * scrolls, smoothly, to put the place of the first of them at the top, as the code of SkeVU does: for a call, the line where it is called.
 */
/** What the tooltip says of a node: the label, what it is, its size, backend and place in the C++ program. */
export function tooltipData(ele) {
    const id = ele.id(), node = find(id), line = S.programs?.cpp.sites.find(s => s.id === id)?.line;
    const declared = (S.programs?.cpp.lines.findIndex(ids => ids.includes(id)) ?? -1) + 1;
    const place = { file: S.files.cpp, line: line ?? (declared || undefined) };
    switch (kindOf(id)) {
        case 'container': { const c = effective(id); return node.kind === 'PRNG' ? { label: node.label, pattern: 'random stream', ...place } : { label: c.label, pattern: `${c.kind}<${c.type}>`, elements: c.size, ...place }; }
        case 'skeleton': return { label: node.label, pattern: PATTERNS[node.pattern]?.title ?? node.pattern, backend: BACKEND_NAME[node.backend] ?? '', ...place };
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

/** Make the graph (and its legend) for the design, and what the pointer and the selection do to it. */
export function initGraph() {
    S.cy = window.cytoscape({ container: $('#cy'), elements: [], style: stylesheet(), ...PAN_ZOOM_OPTIONS, boxSelectionEnabled: true });
    bindPanZoom(S.cy, $('#cy'));                              // the wheel pans and the pinch zooms, as in SkeVU
    S.legend = new LegendView($('#legend'), view, { labels: { container_update: 'Data' } });
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
    // the hover: a tooltip, the entry of the legend, and the badge of the node in the code (graph_interaction.js, as in SkeVU)
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
    S.cy.on('dragfree', 'node', () => { storePositions(); changed({ merge: 'move' }); });          // (a drag is one step of undo, whatever the number of nodes in it)
    // (the size of the view is known when the page is laid out)
    requestAnimationFrame(() => { S.cy.resize(); S.cy.fit(undefined, 40); });
    new ResizeObserver(() => S.cy.resize()).observe($('#cy'));
}

hooks.rebuild = rebuild;
hooks.syncData = syncData;
hooks.markProblems = markProblems;
