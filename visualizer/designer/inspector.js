// inspector.js — the properties of what is selected, in the pane below the code: the forms of a container, a skeleton, a region and an edge.
import { PATTERNS, ELEMENT_TYPES, BACKENDS, EDGES, CONTAINER_KINDS, defaultSkeleton, defaultSize, renameObject, syncObject, detachNode, membersOf, objectKey, dimsOf } from './model.js';
import { $, S, changed, effective, el, find, hooks, kindOf, say } from './state.js';
import { activeLanguage } from './codeview.js';
import { add } from './actions.js';

// ── The properties of what is selected ───────────────────────────────────────

export const field = (label, ...controls) => el('div', { className: 'info-row' }, el('span', { className: 'info-key', textContent: label }), el('div', { className: 'info-value' }, ...controls));

export const text = (value, onchange, props = {}) => {
    const input = el('input', { type: 'text', value: value ?? '', ...props });
    input.onchange = () => onchange(input.value);
    return input;
};

export const number = (value, onchange, props = {}) => {
    const input = el('input', { type: 'number', value, min: 1, step: 1, ...props });
    input.onchange = () => onchange(Math.max(+input.min || 0, Math.floor(+input.value) || +input.min || 1));
    return input;
};

export const select = (value, options, onchange, props = {}) => {
    const s = el('select', props);
    for (const o of options) s.append(el('option', { value: Array.isArray(o) ? o[0] : o, textContent: Array.isArray(o) ? o[1] : (o || 'as the program is started') }));
    s.value = value;
    s.onchange = () => onchange(s.value);
    return s;
};

export const checkbox = (value, onchange, props = {}) => {
    const c = el('input', { type: 'checkbox', checked: !!value, ...props });
    c.onchange = () => onchange(c.checked);
    return c;
};

export const heading = t => el('div', { className: 'info-section', textContent: t });

export const note = t => el('div', { className: 'info-note', textContent: t });

export function renderInspector(force = false) {
    const pane = $('#info');
    const sel = S.cy.$(':selected');
    const key = sel.length === 1 ? sel.id() : sel.length ? 'many' : 'none';
    if (!force && key === S.inspected) return;
    S.inspected = key;
    pane.replaceChildren();
    $('#info-heading').textContent = 'Properties';
    if (sel.length !== 1) {
        pane.append(el('div', { className: 'info-empty', textContent: sel.length ? 'Several things are selected.' : 'Select a node or an edge to see its properties.' }));
        return;
    }
    const id = sel.id();
    if (sel.isEdge()) { $('#info-heading').textContent = 'Edge'; return inspectEdge(pane, S.design.edges.find(e => e.id === id)); }
    switch (kindOf(id)) {
        case 'container': $('#info-heading').textContent = find(id).kind === 'PRNG' ? 'Random stream' : sel.data('type') === 'allocation' ? 'Allocation' : 'Data'; return inspectContainer(pane, find(id));
        case 'skeleton':  $('#info-heading').textContent = find(id).pattern === 'IO' ? 'I/O' : find(id).pattern === 'External' ? 'External' : 'Skeleton'; return inspectSkeleton(pane, find(id));
        case 'loop':      $('#info-heading').textContent = find(id).scope ? 'Region' : 'Loop'; return inspectLoop(pane, find(id));
        default:
            if (sel.data('transfer')) {
                $('#info-heading').textContent = 'Transfer';
                const up = sel.data('direction') === 'host-to-device';
                pane.append(note(`${sel.data('label')} moves ${up ? 'from the host to the device' : 'from the device to the host'} before the call that reads it: the calls are on different explicit backends. SkePU does this by itself; it is shown, not designed.`));
            }
    }
}

/**
 * The name of a data node is the name of its object. A name that another object has makes the nodes updates of that object (joining it); a
 * random stream and data are never one object, and say why not.
 */
function rename(c, name) {
    const result = renameObject(S.design, c, name);
    if (result.refused) say(result.refused);
    else if (result.merged) say(`${result.merged}: the nodes are updates of the same data now (Detach: a node of its own again)`);
    changed({ structure: true, inspector: true });
}

/** What the properties of a node that is an update say about the object: which update it is, and how to make it a node of its own. */
function objectNotes(pane, c) {
    const members = membersOf(S.design, objectKey(c));
    if (members.length < 2) return;
    const v = S.result.versions?.[c.id];
    const detach = el('button', { textContent: 'Detach', title: 'Make this node data of its own again: a new object with its own name' });
    detach.onclick = () => { const name = detachNode(S.design, c); if (name) say(`${name}: data of its own`); changed({ structure: true, inspector: true }); };
    pane.append(note(`${v ? `Update ${v.version} of ${v.count}` : 'An allocation'} of ${c.label}: ${members.length} nodes are the same data (the name makes them one). The calls decide the order of the updates. The name, format and initial value are those of the data.`),
        el('div', { className: 'info-buttons' }, detach));
}

export function inspectContainer(pane, c) {
    if (c.kind === 'PRNG') {
        pane.append(field('Name', text(c.label, v => rename(c, v))),
            field('Seed', text(c.init, v => { c.init = v; changed(); }, { placeholder: 'the default seed' })),
            note('A random stream (skepu::PRNG). Connect it to a call that draws random numbers from it (a Map, MapReduce, MapOverlap, MapPairs or MapPairsReduce). It has one edge out: the calls that share it are a chain, each continuing the stream from the one before, so connecting it again connects from the last call of the chain.'));
        return;
    }
    const eff = effective(c.id);
    const inferred = c.auto && eff !== c;
    const producer = S.design.edges.find(e => e.to === c.id && e.role === 'out');
    const fixed = inferred ? { disabled: true } : {};
    const edited = () => { syncObject(S.design, c); changed({ inspector: false }); };            // (the nodes of an object have one format)
    const sizeInputs = () => eff.size.map((n, i) => number(n, v => { c.size[i] = v; edited(); }, { style: 'width: 6.5em', ...fixed }));
    pane.append(
        field('Name', text(c.label, v => rename(c, v))),
        field('Format', checkbox(c.auto, v => {
            c.auto = v;
            if (!v) Object.assign(c, { kind: eff.kind, type: eff.type, size: [...eff.size] });       // by hand: what it is now
            syncObject(S.design, c);
            changed({ structure: true, inspector: true });
        }), el('span', { textContent: 'from the call that writes it' })),
        field('Kind', select(eff.kind, CONTAINER_KINDS, v => { c.kind = v; c.size = v === 'Vector' ? [c.size.reduce((a, b) => a * b, 1)] : defaultSize(v); syncObject(S.design, c); changed({ inspector: true }); }, fixed)),
        field('Element type', select(eff.type, ELEMENT_TYPES, v => { c.type = v; edited(); }, fixed)),
        field({ Vector: 'Size', Matrix: 'Rows × columns', Tensor3: 'Size i × j × k', Tensor4: 'Size i × j × k × l' }[eff.kind] ?? 'Size', ...sizeInputs()),
        field('Initial value', text(c.init, v => { c.init = v; edited(); })),
        note(inferred ? `${eff.kind}<${eff.type}> ${eff.size.join(' × ')}, from ${find(producer.from)?.label}.`
            : c.auto ? (producer ? 'The call that writes it does not say its format yet (it needs its inputs): by hand until it does.' : 'No call writes it: by hand.')
            : 'By hand. Every call that reads or writes the data is an edge: connect it to a skeleton.'),
        note(S.design.edges.some(e => e.to === c.id && e.role === 'out') ? '' : 'Nothing writes it, so it is an allocation: it has its initial value when it is first used.'));
    objectNotes(pane, c);
}

export function inspectLoop(pane, l) {
    pane.append(
        field('Name', text(l.label, v => { l.label = v; changed(); })),
        field('Loop', checkbox(!l.scope, v => { l.scope = !v; if (v && !(+l.count > 1)) l.count = 10; changed({ structure: true, inspector: true }); }), el('span', { textContent: 'repeat the calls' })));
    if (!l.scope) pane.append(field('Iterations', number(l.count, v => { l.count = v; changed(); })));
    pane.append(note(`The name is the trace region (a SKEPU_TRACE_SCOPE) that ${l.scope ? 'the calls are in' : 'each iteration is in'}. The calls of a region run together: a call that depends on them and that they depend on cannot be between them. The data that a call writes is in the region of the call.`));
}

export function inspectEdge(pane, e) {
    if (!e) return;
    const target = find(e.to), from = find(e.from);
    pane.append(note(`${from?.label} → ${target?.label}`));
    if (e.role === 'out') { pane.append(note('The output of the call: the skeleton writes the data.')); return; }
    if (e.role === 'scalar') { pane.append(note('The result of the call, which the I/O block prints. A result that no I/O block prints is printed by the call itself.')); return; }
    if (e.role === 'random') { pane.append(note('The random stream of the Map: its function gets a Random parameter that draws from it. Between two Maps: the second continues the stream after the first.')); return; }
    const roles = Object.keys(PATTERNS[target.pattern]?.roles ?? {});
    pane.append(field('Used as', select(e.role, roles, v => { e.role = v; changed({ structure: true, inspector: true }); })),
        note(e.role === 'proxy' ? 'A proxy: the function can read any element (a Vec or Mat parameter).' :
             e.role === 'elwise' ? 'Element by element: a parameter of the function, in the order of the edges.' : ''));
    const same = S.design.edges.filter(x => x.to === e.to && x.role === e.role);
    if (same.length > 1) {
        const move = delta => {
            const i = S.design.edges.indexOf(e), j = S.design.edges.indexOf(same[same.indexOf(e) + delta]);
            if (j < 0) return;
            [S.design.edges[i], S.design.edges[j]] = [S.design.edges[j], S.design.edges[i]];
            changed({ structure: true, inspector: true });
        };
        const earlier = el('button', { textContent: 'Earlier', disabled: same.indexOf(e) === 0 }), later = el('button', { textContent: 'Later', disabled: same.indexOf(e) === same.length - 1 });
        earlier.onclick = () => move(-1);
        later.onclick = () => move(1);
        pane.append(field('Parameter order', earlier, later));
    }
}

export function inspectSkeleton(pane, s) {
    const spec = PATTERNS[s.pattern];
    const language = activeLanguage();
    pane.append(
        field('Name', text(s.label, v => { s.label = v; changed(); })),
        field('Pattern', select(s.pattern, Object.entries(PATTERNS).map(([key, p]) => [key, p.title]), v => {
            const fresh = defaultSkeleton(v, s.id, s.label);
            Object.assign(s, { pattern: v, params: fresh.params, indexed: false });
            changed({ structure: true, inspector: true });
        })),
        note(spec?.doc ?? ''));
    if (!spec?.host) pane.append(field('Backend', select(s.backend, BACKENDS, v => { s.backend = v; changed(); })));

    const call = S.result.program?.calls.find(c => c.skeleton.id === s.id);
    if (s.pattern === 'Map')
        pane.append(field('Index', checkbox(s.indexed || call?.elwise.length === 0, v => { s.indexed = v; changed({ inspector: true }); }, { disabled: call?.elwise.length === 0 })));
    if (spec?.roles.random)
        pane.append(note(call?.random ? `Random numbers from ${call.random.label}.` : 'Random numbers: connect a random stream (Add › Random Stream) to it.'));
    if (s.pattern === 'MapOverlap') {
        const p = s.params;
        pane.append(heading('Overlap'),
            field('Dimensions', select(String(dimsOf(s)), ['1', '2', '3', '4'], v => { p.dims = +v; changed({ inspector: true }); })),
            field('Radius', ...Array.from({ length: dimsOf(s) }, (_, i) => number(p.overlap[i] ?? 1, v => { p.overlap[i] = v; changed(); }, { style: 'width: 4.2em' }))),
            field('Edge', select(p.edge, EDGES, v => { p.edge = v; changed({ inspector: true }); })));
        if (p.edge === 'Pad') pane.append(field('Pad value', text(p.pad, v => { p.pad = v; changed(); })));
    }
    if (s.pattern === 'MapPairsReduce') {
        pane.append(heading('Reduction'), field('Along', select(s.params.mode ?? 'RowWise', [['RowWise', 'the rows (a vector as long as the vertical one)'], ['ColWise', 'the columns (as long as the horizontal one)']], v => { s.params.mode = v; changed({ structure: true, inspector: true }); })),
            field('Start value', text(s.params.start, v => { s.params.start = v; changed(); })));
    }
    if (s.pattern === 'Scan') {
        pane.append(heading('Scan'), field('Mode', select(s.params.mode, ['Inclusive', 'Exclusive'], v => { s.params.mode = v; changed(); })),
            field('Start value', text(s.params.start, v => { s.params.start = v; changed(); })));
    }

    if (!spec?.host) pane.append(heading('Uniform arguments'));
    const table = el('table', { className: 'uniforms' });
    s.uniforms.forEach((u, i) => {
        const remove = el('button', { textContent: '×', title: 'Remove' });
        remove.onclick = () => { s.uniforms.splice(i, 1); changed({ inspector: true }); };
        table.append(el('tr', {}, el('td', {}, text(u.name, v => { u.name = v; changed(); }, { placeholder: 'name' })),
            el('td', {}, select(u.type, ELEMENT_TYPES, v => { u.type = v; changed(); })),
            el('td', {}, text(u.value, v => { u.value = v; changed(); }, { placeholder: 'value' })), el('td', {}, remove)));
    });
    const add = el('button', { textContent: '+ Uniform' });
    add.onclick = () => { s.uniforms.push({ name: `u${s.uniforms.length + 1}`, type: 'float', value: '1' }); changed({ inspector: true }); };
    if (!spec?.host) pane.append(table, el('div', { className: 'info-buttons' }, add), note('Scalars that every call passes to the function, after the data.'));

    const functions = (S.result.ok ? S.result : S.lastGood)?.program.functions.filter(f => f.call.skeleton.id === s.id) ?? [];
    if (functions.length) pane.append(heading(`${spec?.host ? 'Host code' : `Function${functions.length > 1 ? 's' : ''}`} (${language === 'cpp' ? 'C++' : 'Python'}: the tab of the program that is shown)`));
    for (const f of functions) {
        const area = el('textarea', { className: 'body', value: f.bodies?.[language] ?? '', placeholder: f.stub[language], spellcheck: false });
        area.oninput = () => {
            s.bodies = s.bodies ?? {};
            s.bodies[f.suffix] = { ...(s.bodies[f.suffix] ?? {}), [language]: area.value };
            changed({ merge: `body:${s.id}:${f.suffix}:${language}` });          // (the keys of a text are one step of undo)
        };
        if (!f.host) pane.append(note(`${f.name}(${f.params.map(p => p.name).join(', ')})`));
        pane.append(area);
    }
    if (functions.length) pane.append(note(spec?.host ? 'Empty: the stub that is shown. What you write is the body of the block: it can use the data that it reads and writes by name.' : 'Empty: the stub that is shown. What you write is the body of the function in the program.'));
}

hooks.renderInspector = renderInspector;
