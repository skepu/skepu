// actions.js — editing actions: add, connect, group, duplicate, delete, arrange; the modes (explore, add, connect), selection, Edit menu state and keys.

import { PATTERNS, hasScalar, CONTAINER_KINDS, defaultSkeleton, defaultContainer, defaultPrng, defaultSize, duplicateNodes, inferOutput, joinObjects, randomTail, dimsOf } from './model.js';
import { enableMenuItem } from '../script/menu.js';
import { $, S, changed, descendantOf, effective, find, history, hooks, kindOf, nextId, normalize, say, setChecked, store } from './state.js';
import { storePositions } from './graphview.js';
import { LAYOUT_NODE_SEP, LAYOUT_RANK_SEP } from '../script/layout_options.js';
import { startLayout, layoutWorkerUrls } from '../script/layout.js';

// ── Editing the graph ────────────────────────────────────────────────────────

export function add(node, list, at) {
    if (at) S.design.positions[node.id] = { x: Math.round(at.x), y: Math.round(at.y) };
    list.push(node);
    changed({ structure: true });
    selectById(node.id);
}

export function addContainer(kind, at) {
    const labels = new Set(S.design.containers.map(c => c.label));
    let n = S.design.containers.length + 1, label;
    const base = kind === 'PRNG' ? 'random' : 'data';
    do label = `${base}${n++}`; while (labels.has(label));
    add(kind === 'PRNG' ? defaultPrng(nextId('c'), label) : defaultContainer(nextId('c'), label, kind), S.design.containers, at);
}

export function addSkeleton(pattern, at, parent) {
    const n = S.design.skeletons.filter(s => s.pattern === pattern).length + 1;
    const label = pattern === 'IO' ? (n === 1 ? 'I/O' : `I/O ${n}`) : `${pattern.toLowerCase()}${n}`;
    add({ ...defaultSkeleton(pattern, nextId('s'), label), ...(parent ? { parent } : {}) }, S.design.skeletons, at);
}

/** The types the Add menu offers: data kinds, skeletons and regions. */
export const ADDABLE = [...CONTAINER_KINDS, 'PRNG', ...Object.keys(PATTERNS), 'Region'];

export function addNode(type, at, region = null) {
    if (CONTAINER_KINDS.includes(type) || type === 'PRNG') addContainer(type, at);
    else if (type === 'Region') addRegion(at, region);
    else addSkeleton(type, at, region);
}

/** The skeletons that take a random stream, as a list for messages. */
const DRAWS = Object.entries(PATTERNS).filter(([, p]) => p.roles.random).map(([name]) => name).join(', ').replace(/, ([^,]*)$/, ' or $1');

export function defaultRole(target) {
    if (PATTERNS[target.pattern]?.host) return 'read';
    if (target.pattern === 'MapPairs' || target.pattern === 'MapPairsReduce')
        return S.design.edges.some(e => e.to === target.id && e.role === 'vertical') ? 'horizontal' : 'vertical';
    return 'elwise';
}

/** Skeleton → skeleton: the data written by the first (created if it has none) is the input of the second. */
export function connectCalls(a, b) {
    const from = find(a), to = find(b);
    if (PATTERNS[from.pattern]?.io) return say(`${from.label}: I/O blocks write no data.`);
    const scalar = hasScalar(S.design, from);                    // a Reduce writing a vector has data
    if (scalar && PATTERNS[to.pattern]?.io) {              // result of a reduction, printed
        if (S.design.edges.some(e => e.from === a && e.to === b && e.role === 'scalar')) return say('Edge exists.');
        const edge = { id: nextId('e'), from: a, to: b, role: 'scalar' };
        S.design.edges.push(edge);
        changed({ structure: true });
        selectById(edge.id);
        return say(`${from.label} → ${to.label} (result)`);
    }
    if (scalar) return say(`${from.label} has a scalar result: connect it to an I/O block.`);
    let out = S.design.edges.find(e => e.from === a && e.role === 'out');
    let made = null;
    if (!out) {
        const inputs = {};
        for (const e of S.design.edges) if (e.to === a && e.role !== 'out' && find(e.from)) (inputs[e.role] ??= []).push(effective(e.from));
        const format = inferOutput(from, inputs) ?? (() => { const kind = from.pattern === 'MapPairs' ? 'Matrix' : from.pattern === 'MapPairsReduce' ? 'Vector' : from.pattern === 'MapOverlap' || from.pattern === 'MapPool' ? CONTAINER_KINDS[dimsOf(from) - 1] : 'Vector'; return { kind, type: 'float', size: defaultSize(kind) }; })();
        const labels = new Set(S.design.containers.map(c => c.label));
        let label = `${from.label}_out`;
        for (let n = 2; labels.has(label); n++) label = `${from.label}_out${n}`;
        made = { ...defaultContainer(nextId('c'), label, format.kind), ...format, auto: true };
        S.design.containers.push(made);
        const pa = S.design.positions[a] ?? { x: 0, y: 0 }, pb = S.design.positions[b] ?? pa;
        S.design.positions[made.id] = { x: (pa.x + pb.x) / 2 - 60, y: (pa.y + pb.y) / 2 };
        out = { id: nextId('e'), from: a, to: made.id, role: 'out' };
        S.design.edges.push(out);
    }
    const role = defaultRole(to);
    if (S.design.edges.some(e => e.from === out.to && e.to === b && e.role === role)) return say('Edge exists.');
    const edge = { id: nextId('e'), from: out.to, to: b, role };
    S.design.edges.push(edge);
    changed({ structure: true });
    selectById(edge.id);
    say(`${from.label} → ${find(out.to).label} → ${to.label}${made ? ' (data created)' : ''}`);
}

/**
 * Data → data: the two become updates of one object with the name and format of the first.
 */
export function connectData(a, b) {
    const was = find(b).label;
    const result = joinObjects(S.design, find(a), find(b));
    if (result.refused) return say(result.refused);
    changed({ structure: true, inspector: true });
    selectById(b);
    say(`${result.merged}: updates of one object`);
}

export function connect(a, b) {
    const from = find(a), to = find(b), kinds = [kindOf(a), kindOf(b)];
    let edge;
    if (kinds.includes('loop')) return connectRegion(a, b);
    if (kinds[0] === 'skeleton' && kinds[1] === 'skeleton') return connectCalls(a, b);
    if (kinds[0] === 'container' && kinds[1] === 'container') return connectData(a, b);
    if (kinds[0] === 'container' && kinds[1] === 'skeleton' && from.kind === 'PRNG') {
        if (!PATTERNS[to.pattern]?.roles.random) return say(`Random streams connect to ${DRAWS}.`);
        if (S.design.edges.some(e => e.to === b && e.role === 'random')) return say(`${to.label} already has a random stream.`);
        // connect from the last call of the stream's chain
        const tail = randomTail(S.design, a);
        edge = { from: tail, to: b, role: 'random' };
        if (tail !== a) say(`${find(tail).label} → ${to.label} (stream chained)`);
    }
    else if (kinds[0] === 'container' && kinds[1] === 'skeleton') edge = { from: a, to: b, role: defaultRole(to) };
    else if (kinds[0] === 'skeleton' && kinds[1] === 'container' && to.kind === 'PRNG') return say('Calls do not write random streams.');
    else if (kinds[0] === 'skeleton' && kinds[1] === 'container') edge = { from: a, to: b, role: 'out' };
    else return say('Connect skeleton → skeleton, data → skeleton, skeleton → data or data → data.');
    if (S.design.edges.some(e => e.from === edge.from && e.to === edge.to && e.role === edge.role)) return say('Edge exists.');
    edge.id = nextId('e');
    S.design.edges.push(edge);
    changed({ structure: true });
    selectById(edge.id);
    if (edge.role !== 'random' || edge.from === a) say(`${from.label} → ${to.label}`);
}

/** Region ↔ call: the call is in the region. Region ↔ region: the second is in the first. */
export function connectRegion(a, b) {
    const region = kindOf(a) === 'loop' ? a : b, member = region === a ? b : a;
    if (kindOf(member) === 'container') return say('Data belongs to the region of its call.');
    if (kindOf(member) === 'loop' && (member === region || descendantOf(region, member))) return say('A region cannot contain itself.');
    if (find(member).parent === region) return say(`${find(member).label} is already in ${find(region).label}.`);
    find(member).parent = region;
    changed({ structure: true });
    selectById(region);
    say(`${find(member).label} in ${find(region).label}`);
}

export function deleteSelected() {
    const ids = new Set(S.cy.$(':selected').map(x => x.id()));
    if (!ids.size) return;
    const loops = S.design.loops.filter(l => ids.has(l.id));
    for (const l of loops) dissolve(l);
    // deleting a call from a random chain reconnects the chain
    for (const id of ids) {
        const edges = S.design.edges, into = edges.find(e => e.to === id && e.role === 'random'), out = edges.find(e => e.from === id && e.role === 'random');
        if (into && out) { S.design.edges = edges.filter(e => e !== into && e !== out); S.design.edges.push({ id: nextId('e'), from: into.from, to: out.to, role: 'random' }); }
    }
    S.design.containers = S.design.containers.filter(c => !ids.has(c.id));
    S.design.skeletons = S.design.skeletons.filter(s => !ids.has(s.id));
    S.design.edges = S.design.edges.filter(e => !ids.has(e.id) && !ids.has(e.from) && !ids.has(e.to));
    for (const id of ids) delete S.design.positions[id];
    changed({ structure: true });
}

/** Removes a loop; its contents move to the enclosing loop. */
export function dissolve(loop) {
    for (const x of [...S.design.skeletons, ...S.design.loops]) if (x.parent === loop.id) x.parent = loop.parent;
    S.design.loops = S.design.loops.filter(l => l.id !== loop.id);
}

/** The selected nodes of a kind (transfers are not design nodes). */
export const selectedOf = kind => S.cy.$(':selected').nodes().map(n => n.id()).filter(id => kindOf(id) === kind);

/** The selected calls and regions that can be grouped: those outside selected regions, if all share one parent. */
export function groupable() {
    const picked = [...selectedOf('skeleton'), ...selectedOf('loop')];
    const members = picked.filter(id => !picked.some(other => other !== id && descendantOf(id, other)));
    return members.length && new Set(members.map(id => find(id).parent ?? null)).size === 1 ? { members, parent: find(members[0]).parent ?? null } : null;
}

/** A region (scope) or loop around the selected calls and regions. */
export function groupNodes(scope) {
    const group = groupable();
    if (!group) return say(selectedOf('skeleton').length + selectedOf('loop').length ? 'Selection spans several regions.' : 'Select calls to group.');
    const labels = new Set(S.design.loops.map(l => l.label)), base = scope ? 'Region' : 'Loop';
    let n = S.design.loops.filter(l => !!l.scope === scope).length + 1;
    while (labels.has(`${base} ${n}`)) n++;
    const loop = { id: nextId('l'), label: `${base} ${n}`, count: scope ? 1 : 10, scope, parent: group.parent };
    S.design.loops.push(loop);
    for (const id of group.members) find(id).parent = loop.id;
    changed({ structure: true });
    selectById(loop.id);
}

export const groupInLoop = () => groupNodes(false);

export function addRegion(at, parent = null) {
    const labels = new Set(S.design.loops.map(l => l.label));
    let n = S.design.loops.filter(l => l.scope).length + 1;
    while (labels.has(`Region ${n}`)) n++;
    add({ id: nextId('l'), label: `Region ${n}`, count: 1, scope: true, parent }, S.design.loops, at);
}

/** The selected design nodes (not edges or transfers). */
export const selectedNodes = () => S.cy.$(':selected').nodes().map(n => n.id()).filter(id => kindOf(id));

/** Edit > Duplicate: copies the selected nodes (a region with its contents) and selects the copies. */
export function duplicate() {
    const ids = selectedNodes();
    if (!ids.length) return say('No selection.');
    const made = duplicateNodes(S.design, ids);
    changed({ structure: true });
    S.cy.elements().unselect();
    made.forEach(id => S.cy.$id(id).select());
    say(`Duplicated ${made.length} node${made.length === 1 ? '' : 's'}.`);
}

export const canUngroup = () => selectedOf('loop').length > 0 || [...selectedOf('skeleton'), ...selectedOf('loop')].some(id => find(id).parent);

export function ungroup() {
    const selected = [...selectedOf('skeleton'), ...selectedOf('loop')];
    const loops = selected.filter(id => kindOf(id) === 'loop');
    if (!selected.length || (!loops.length && !canUngroup())) return say('Select a region, or calls in a region.');
    if (loops.length) loops.forEach(id => dissolve(find(id)));
    else for (const id of selected) { const x = find(id); x.parent = S.design.loops.find(l => l.id === x.parent)?.parent ?? null; }
    changed({ structure: true });
}

export function updateMenu() {
    const enable = enableMenuItem;
    const nodes = S.design.containers.length + S.design.skeletons.length + S.design.loops.length;
    const selected = S.cy.$(':selected').filter(x => x.isEdge() ? S.design.edges.some(e => e.id === x.id()) : !!find(x.id()));
    enable('#connect', !!S.connecting || nodes >= 2);
    enable('#duplicate', selectedNodes().length > 0);
    enable('#group-loop', !!groupable());
    enable('#group-nodes', !!groupable());
    enable('#ungroup', canUngroup());
    enable('#delete', selected.length > 0);
    enable('#undo', history.canUndo());
    enable('#redo', history.canRedo());
    enable('#arrange', nodes > 0);
}

/** A signature of the laid-out graph: nodes, regions and edges. */
const shapeOf = () => JSON.stringify([S.cy.nodes().map(n => [n.id(), n.data('parent')]).sort(), S.cy.edges().map(e => [e.source().id(), e.target().id()]).sort()]);

/**
 * Lays out the graph with the layout of SkeVU (script/layout.js). Positions become design positions. They decide the call order, which decides the
 * data, transfers and regions: if that changes the graph, it is laid out again.
 */
export async function arrange() {
    for (let round = 0; round < 4; round++) {
        storePositions();
        const shape = shapeOf();
        const layout = startLayout(S.cy, { rankDir: 'TB', nodeSep: LAYOUT_NODE_SEP, rankSep: LAYOUT_RANK_SEP, nodeDimensionsIncludeLabels: true,
                                           ranker: 'longest-path', acyclicer: 'greedy', algorithm: 'layered' },
                                   layoutWorkerUrls(), S.cy.nodes(':parent').length > 0);
        let positions;
        try { positions = await layout.promise; } catch (err) { console.error(err); return say('Layout failed.'); }
        for (const [id, at] of Object.entries(positions)) if (S.cy.$id(id).length && !S.cy.$id(id).isParent()) S.design.positions[id] = at;
        changed({ structure: true, record: round === 0 });          // one undo step
        if (shapeOf() === shape) break;
    }
    S.cy.fit(undefined, 40);
}

export function selectById(id, from = 'graph') {
    S.selectionFrom = from;
    S.cy.elements().unselect();
    S.cy.$id(id).select();
    S.selectionFrom = 'graph';
}

export const MODES = {
    explore: { label: 'Explore', help: 'Select and move nodes. A: add, C: connect.' },
    connect: { label: 'Connect', help: 'Select a source, then a target. Esc: Explore.' },
    add:     { label: 'Add', help: 'Click the canvas or a region to add the checked type. Esc: Explore.' },
};

/** The editor mode: 'explore' (select, move), 'connect' (clicks join nodes) or 'add' (a click adds a node of `type`). */
export function setMode(mode, type) {
    S.connecting = mode === 'connect' ? { source: null } : null;
    S.adding = mode === 'add' ? (type ?? S.lastAdd) : null;
    if (S.adding) { S.lastAdd = S.adding; store('skevu-designer-add', S.adding); }
    S.cy.nodes().removeClass('connect-source');
    $('#cy').classList.toggle('connecting', !!S.connecting);
    $('#cy').classList.toggle('adding', !!S.adding);
    setChecked($('#connect'), !!S.connecting);
    for (const b of document.querySelectorAll('#add-menu [data-add]')) setChecked(b, b.dataset.add === S.adding);
    const badge = $('#mode-badge');
    badge.textContent = MODES[mode].label;
    badge.className = `mode-${mode}`;
    badge.title = (S.adding ? `${MODES[mode].help} (${S.adding})` : MODES[mode].help) + ' Click: next mode.';
    updateMenu();
}

export const setConnecting = on => setMode(on ? 'connect' : 'explore');

/** Sets the design to `text` (undo, redo): reselects the nodes that still exist. */
function goTo(text) {
    if (text === null) return;
    const had = new Set([...S.design.containers, ...S.design.skeletons, ...S.design.loops].map(n => n.id));
    S.design = normalize(JSON.parse(text));
    S.lastGood = null;
    $('#design-name').value = S.design.name;
    changed({ structure: true, inspector: true, record: false });
    const back = [...S.design.containers, ...S.design.skeletons, ...S.design.loops].map(n => n.id).filter(id => !had.has(id));
    if (back.length) { S.cy.elements().unselect(); back.forEach(id => S.cy.$id(id).select()); }
}

export const undo = () => goTo(history.undo());
export const redo = () => goTo(history.redo());

export function replace(next) {
    S.design = next;
    S.lastGood = null;
    $('#design-name').value = S.design.name;
    changed({ structure: true, inspector: true });
    S.cy.fit(undefined, 40);
}

export function initActions() {
    S.cy.on('tap', 'node', evt => {                         // connect mode: source, then target
        if (!S.connecting) return;
        const id = evt.target.id();
        if (!kindOf(id)) return;                    // transfers are not design nodes
        if (!S.connecting.source) { S.connecting.source = id; evt.target.addClass('connect-source'); return; }
        const from = S.connecting.source;
        S.cy.nodes().removeClass('connect-source');
        S.connecting.source = null;
        if (from !== id) connect(from, id);
    });
    S.cy.on('tap', evt => {                    // add mode: a click on the canvas or a region adds a node
        if (!S.adding) return;
        const t = evt.target, region = t !== S.cy && t.isNode() && kindOf(t.id()) === 'loop' ? t.id() : null;
        if (t !== S.cy && !region) return;
        addNode(S.adding, evt.position, region);
    });
    S.cy.on('tap', evt => { if (evt.target === S.cy && S.connecting?.source) { S.connecting.source = null; S.cy.nodes().removeClass('connect-source'); } });

    $('#group-loop').onclick = groupInLoop;
    $('#group-nodes').onclick = () => groupNodes(true);
    $('#duplicate').onclick = duplicate;
    $('#ungroup').onclick = ungroup;
    $('#connect').onclick = () => setMode(S.connecting ? 'explore' : 'connect');
    $('#mode-badge').onclick = () => setMode(S.adding ? 'connect' : S.connecting ? 'explore' : 'add');          // Explore → Add → Connect → Explore
    $('#delete').onclick = deleteSelected;
    $('#arrange').onclick = arrange;
    $('#fit').onclick = () => S.cy.fit(undefined, 40);
    const mac = /Mac|iPhone|iPad/.test(navigator.platform);
    $('#undo .menu-key').textContent = mac ? '⌘Z' : 'Ctrl+Z';
    $('#redo .menu-key').textContent = mac ? '⇧⌘Z' : 'Ctrl+Y';
    $('#undo').onclick = undo;
    $('#redo').onclick = redo;
    document.addEventListener('keydown', e => {
        const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
        if ((e.metaKey || e.ctrlKey) && !e.altKey && !typing) {                  // text fields keep the browser's undo
            const key = e.key.toLowerCase();
            if (key === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); }
            else if (key === 'y' && !e.metaKey) { e.preventDefault(); redo(); }
            return;
        }
        if (typing || e.metaKey || e.ctrlKey) return;
        if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); if (!$('#delete').disabled) deleteSelected(); }
        else if (e.key === 'Escape') setMode('explore');
        else if (e.key === 'c') setMode(S.connecting ? 'explore' : 'connect');
        else if (e.key === 'a') setMode(S.adding ? 'explore' : 'add');
        else if (e.key === 'l' && groupable()) groupInLoop();
        else if (e.key === 'g' && groupable()) groupNodes(true);
        else if (e.key === 'd' && selectedNodes().length) duplicate();
    });
}

hooks.updateMenu = updateMenu;
