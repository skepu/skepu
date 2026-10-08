// actions.js — what can be done to the design: add, connect, group, duplicate, delete, arrange; the modes (explore, add, connect), the selection, the
// items of the Edit menu (which are enabled when they would do something) and the keys.

import { PATTERNS, CONTAINER_KINDS, defaultSkeleton, defaultContainer, defaultPrng, defaultSize, keepLoopsTogether, duplicateNodes, inferOutput, joinObjects, randomTail, dimsOf } from './model.js';
import { enableMenuItem } from '../script/menu.js';
import { $, S, changed, descendantOf, effective, find, history, hooks, kindOf, nextId, normalize, say, setChecked, store } from './state.js';
import { storePositions } from './graphview.js';

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
    const label = pattern === 'IO' ? (n === 1 ? 'I/O' : `I/O ${n}`) : `${pattern.toLowerCase()}${n}`;           // (an I/O block is "I/O" in SkeVU)
    add({ ...defaultSkeleton(pattern, nextId('s'), label), ...(parent ? { parent } : {}) }, S.design.skeletons, at);
}

/** What the Add menu can add: the type is a kind of data, a skeleton, or a region. */
export const ADDABLE = [...CONTAINER_KINDS, 'PRNG', ...Object.keys(PATTERNS), 'Region'];

/** Adds a node of a type at a place of the canvas (in a region, if the place is in one). */
export function addNode(type, at, region = null) {
    if (CONTAINER_KINDS.includes(type) || type === 'PRNG') addContainer(type, at);
    else if (type === 'Region') addRegion(at, region);
    else addSkeleton(type, at, region);
}

/** The skeletons that can draw random numbers, for what is said of them. */
const DRAWS = Object.entries(PATTERNS).filter(([, p]) => p.roles.random).map(([name]) => name).join(', ').replace(/, ([^,]*)$/, ' or $1');

export function defaultRole(target) {
    if (PATTERNS[target.pattern]?.host) return 'read';
    if (target.pattern === 'MapPairs' || target.pattern === 'MapPairsReduce')
        return S.design.edges.some(e => e.to === target.id && e.role === 'vertical') ? 'horizontal' : 'vertical';
    return 'elwise';
}

/** The data that a call writes, from a call to a call: the data that the first writes (made if it writes none, in its format) is the input of the second. */
export function connectCalls(a, b) {
    const from = find(a), to = find(b);
    if (PATTERNS[from.pattern]?.io) return say(`${from.label} prints data: it writes nothing for a call to read.`);
    if (PATTERNS[from.pattern]?.scalar && PATTERNS[to.pattern]?.io) {              // the result of a reduction, printed
        if (S.design.edges.some(e => e.from === a && e.to === b && e.role === 'scalar')) return say('That edge is already there.');
        const edge = { id: nextId('e'), from: a, to: b, role: 'scalar' };
        S.design.edges.push(edge);
        changed({ structure: true });
        selectById(edge.id);
        return say(`${from.label}'s result is printed by ${to.label}`);
    }
    if (PATTERNS[from.pattern]?.scalar) return say(`${from.label} is a ${from.pattern}: its result is a scalar, not data for a call (an I/O block can print it).`);
    let out = S.design.edges.find(e => e.from === a && e.role === 'out');
    let made = null;
    if (!out) {
        const inputs = {};
        for (const e of S.design.edges) if (e.to === a && e.role !== 'out' && find(e.from)) (inputs[e.role] ??= []).push(effective(e.from));
        const format = inferOutput(from, inputs) ?? (() => { const kind = from.pattern === 'MapPairs' ? 'Matrix' : from.pattern === 'MapPairsReduce' ? 'Vector' : from.pattern === 'MapOverlap' ? CONTAINER_KINDS[dimsOf(from) - 1] : 'Vector'; return { kind, type: 'float', size: defaultSize(kind) }; })();
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
    if (S.design.edges.some(e => e.from === out.to && e.to === b && e.role === role)) return say('That edge is already there.');
    const edge = { id: nextId('e'), from: out.to, to: b, role };
    S.design.edges.push(edge);
    changed({ structure: true });
    selectById(edge.id);
    say(`${from.label} → ${find(out.to).label} → ${to.label}${made ? ' (the data is made)' : ''}`);
}

/**
 * Data to data: the two are updates of the same object, which has the name and format of the first. (The data flow orders the updates, so the
 * direction of the connection means nothing more.)
 */
export function connectData(a, b) {
    const was = find(b).label;
    const result = joinObjects(S.design, find(a), find(b));
    if (result.refused) return say(result.refused);
    changed({ structure: true, inspector: true });
    selectById(b);
    say(`${was === result.merged ? 'The nodes' : was} and ${result.merged} are updates of the same data now (Detach in the properties: a node of its own again)`);
}

export function connect(a, b) {
    const from = find(a), to = find(b), kinds = [kindOf(a), kindOf(b)];
    let edge;
    if (kinds.includes('loop')) return connectRegion(a, b);
    if (kinds[0] === 'skeleton' && kinds[1] === 'skeleton') return connectCalls(a, b);
    if (kinds[0] === 'container' && kinds[1] === 'container') return connectData(a, b);
    if (kinds[0] === 'container' && kinds[1] === 'skeleton' && from.kind === 'PRNG') {
        if (!PATTERNS[to.pattern]?.roles.random) return say(`A ${to.pattern} does not draw random numbers: a random stream goes to ${DRAWS}.`);
        if (S.design.edges.some(e => e.to === b && e.role === 'random')) return say(`${to.label} draws from a random stream already.`);
        // a stream has one edge out: the calls that draw from it are a chain, so the connection is made from the last call of it
        const tail = randomTail(S.design, a);
        edge = { from: tail, to: b, role: 'random' };
        if (tail !== a) say(`${find(tail).label} → ${to.label} (the random stream ${from.label} continues in a chain)`);
    }
    else if (kinds[0] === 'container' && kinds[1] === 'skeleton') edge = { from: a, to: b, role: defaultRole(to) };
    else if (kinds[0] === 'skeleton' && kinds[1] === 'container' && to.kind === 'PRNG') return say(`${to.label} is a random stream: a call does not write it.`);
    else if (kinds[0] === 'skeleton' && kinds[1] === 'container') edge = { from: a, to: b, role: 'out' };
    else return say('Connect a skeleton to a skeleton (the data between them is made), data to a skeleton, a skeleton to data, or data to data (updates of the same data).');
    if (S.design.edges.some(e => e.from === edge.from && e.to === edge.to && e.role === edge.role)) return say('That edge is already there.');
    edge.id = nextId('e');
    S.design.edges.push(edge);
    changed({ structure: true });
    selectById(edge.id);
    if (edge.role !== 'random' || edge.from === a) say(`${from.label} → ${to.label}`);
}

/** A region connected to a call (or to a region): the call is in the region. Of two regions, the second is in the first. */
export function connectRegion(a, b) {
    const region = kindOf(a) === 'loop' ? a : b, member = region === a ? b : a;
    if (kindOf(member) === 'container') return say(`${find(member).label}: the data that a call writes is in the region of the call. Put the call in the region.`);
    if (kindOf(member) === 'loop' && (member === region || descendantOf(region, member))) return say('A region cannot be in itself.');
    if (find(member).parent === region) return say(`${find(member).label} is in ${find(region).label} already.`);
    find(member).parent = region;
    changed({ structure: true });
    selectById(region);
    say(`${find(member).label} is in ${find(region).label}`);
}

export function deleteSelected() {
    const ids = new Set(S.cy.$(':selected').map(x => x.id()));
    if (!ids.size) return;
    const loops = S.design.loops.filter(l => ids.has(l.id));
    for (const l of loops) dissolve(l);
    // a call that is deleted from the middle of a chain of random edges: the chain goes on past it
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

/** Remove a loop, its contents go to the loop around it. */
export function dissolve(loop) {
    for (const x of [...S.design.skeletons, ...S.design.loops]) if (x.parent === loop.id) x.parent = loop.parent;
    S.design.loops = S.design.loops.filter(l => l.id !== loop.id);
}

/** The nodes that are selected, of a kind (the transfers are not in the design). */
export const selectedOf = kind => S.cy.$(':selected').nodes().map(n => n.id()).filter(id => kindOf(id) === kind);

/** The calls and regions that can be put in a region: the selected ones that are not in a selected region, if they are all in the same region. */
export function groupable() {
    const picked = [...selectedOf('skeleton'), ...selectedOf('loop')];
    const members = picked.filter(id => !picked.some(other => other !== id && descendantOf(id, other)));
    return members.length && new Set(members.map(id => find(id).parent ?? null)).size === 1 ? { members, parent: find(members[0]).parent ?? null } : null;
}

/** A region (a scope: a group of calls) or a loop around the selected calls and regions. */
export function groupNodes(scope) {
    const group = groupable();
    if (!group) return say(selectedOf('skeleton').length + selectedOf('loop').length ? 'The selection is in different regions.' : 'Select the calls to put in a region.');
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

/** Add > Region: a region with nothing in it; connect it to the calls that are in it. */
export function addRegion(at, parent = null) {
    const labels = new Set(S.design.loops.map(l => l.label));
    let n = S.design.loops.filter(l => l.scope).length + 1;
    while (labels.has(`Region ${n}`)) n++;
    add({ id: nextId('l'), label: `Region ${n}`, count: 1, scope: true, parent }, S.design.loops, at);
}

/** The selected nodes of the design (not edges, not transfers). */
export const selectedNodes = () => S.cy.$(':selected').nodes().map(n => n.id()).filter(id => kindOf(id));

/** Edit > Duplicate: copies of the selected nodes (a region with what is in it), selected when they are made. */
export function duplicate() {
    const ids = selectedNodes();
    if (!ids.length) return say('Select the nodes to duplicate.');
    const made = duplicateNodes(S.design, ids);
    changed({ structure: true });
    S.cy.elements().unselect();
    made.forEach(id => S.cy.$id(id).select());
    say(`${made.length} node${made.length === 1 ? '' : 's'} made`);
}

export const canUngroup = () => selectedOf('loop').length > 0 || [...selectedOf('skeleton'), ...selectedOf('loop')].some(id => find(id).parent);

export function ungroup() {
    const selected = [...selectedOf('skeleton'), ...selectedOf('loop')];
    const loops = selected.filter(id => kindOf(id) === 'loop');
    if (!selected.length || (!loops.length && !canUngroup())) return say('Select a region to remove, or the calls to take out of their region.');
    if (loops.length) loops.forEach(id => dissolve(find(id)));
    else for (const id of selected) { const x = find(id); x.parent = S.design.loops.find(l => l.id === x.parent)?.parent ?? null; }
    changed({ structure: true });
}

/** Which of the Edit menu would do something now. */
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

export function arrange() {
    storePositions();
    try {
        const layout = S.cy.layout({ name: 'dagre', rankDir: 'TB', nodeSep: 30, rankSep: 45, fit: true, padding: 30, animate: false });
        layout.one('layoutstop', () => { storePositions(); keepLoopsTogether(S.design); changed({ structure: true }); });
        layout.run();
    } catch { say('The layout is not available.'); }
}

export function selectById(id, from = 'graph') {
    S.selectionFrom = from;
    S.cy.elements().unselect();
    S.cy.$id(id).select();
    S.selectionFrom = 'graph';
}

export const MODES = {
    explore: { label: 'Explore', help: 'Select and move nodes. A: add nodes, C: connect them.' },
    connect: { label: 'Connect', help: 'Click a node, then another: the second reads what the first writes. Between two skeletons the data is made; between two data nodes they are updates of the same data (one name); a region and a call: the call is in the region. Esc: back.' },
    add:     { label: 'Add', help: 'Click on the canvas to add a node of the checked type; click in a region to add to it. A: back to explore.' },
};

/** The mode of the editor: 'explore' (select and move), 'connect' (the clicks join nodes), or 'add' (a click on the canvas adds a node of `type`). */
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
    badge.title = (S.adding ? `${MODES[mode].help} (${S.adding})` : MODES[mode].help) + ' Click: the next mode (Explore, Add, Connect).';
    updateMenu();
}

export const setConnecting = on => setMode(on ? 'connect' : 'explore');

/** The design goes to `text` (undo or redo, from history.js): what comes back is selected, and what was selected and is gone is not. */
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

/** What the clicks on the graph, the buttons of the Edit menu and the keys do. */
export function initActions() {
    S.cy.on('tap', 'node', evt => {                         // the Connect mode: the first node, then the second
        if (!S.connecting) return;
        const id = evt.target.id();
        if (!kindOf(id)) return;                    // (a transfer is not part of the design)
        if (!S.connecting.source) { S.connecting.source = id; evt.target.addClass('connect-source'); return; }
        const from = S.connecting.source;
        S.cy.nodes().removeClass('connect-source');
        S.connecting.source = null;
        if (from !== id) connect(from, id);
    });
    S.cy.on('tap', evt => {                    // the Add mode: a click on the canvas, or in a region, adds a node
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
    $('#mode-badge').onclick = () => setMode(S.adding ? 'connect' : S.connecting ? 'explore' : 'add');          // Explore, Add, Connect, and round again
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
        if ((e.metaKey || e.ctrlKey) && !e.altKey && !typing) {                  // (in a text field the keys are the browser's: its own undo)
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
