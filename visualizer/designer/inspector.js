// inspector.js — properties of the selection: forms for data, skeletons, regions and edges.
import { PATTERNS, ELEMENT_TYPES, BACKENDS, EDGES, UPDATE_MODES, CONTAINER_KINDS, defaultSkeleton, defaultSize, renameObject, syncObject, detachNode, membersOf, objectKey, dimsOf, poolOf, strideOf, stridesOf } from './model.js';
import { $, S, changed, codeOptions, effective, el, find, hooks, kindOf, say } from './state.js';
import { activeLanguage } from './codeview.js';
import { isLambda, isAnonymous } from './codegen.js';
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
    for (const o of options) s.append(el('option', { value: Array.isArray(o) ? o[0] : o, textContent: Array.isArray(o) ? o[1] : (o || 'Default') }));
    s.value = value;
    s.onchange = () => onchange(s.value);
    return s;
};

/** The options of a select as [value, label], the label of the default one marked. */
export const withDefault = (options, standard) => options.map(o => { const [value, label] = Array.isArray(o) ? o : [o, o]; return [value, value === standard ? `${label} (Default)` : label]; });

export const checkbox = (value, onchange, props = {}) => {
    const c = el('input', { type: 'checkbox', checked: !!value, ...props });
    c.onchange = () => onchange(c.checked);
    return c;
};

export const heading = t => el('div', { className: 'info-section', textContent: t });

export const note = t => el('div', { className: 'info-note', textContent: t });

export const tip = (text, node) => { node.title = text; return node; };

export function renderInspector(force = false) {
    const pane = $('#info');
    const sel = S.cy.$(':selected');
    const key = sel.length === 1 ? sel.id() : sel.length ? 'many' : 'none';
    if (!force && key === S.inspected) return;
    S.inspected = key;
    // the pane opens for a single selection and collapses otherwise
    $('#info_window').style.flex = sel.length === 1 ? `0 0 ${S.infoHeight ?? Math.round(innerHeight * 0.4)}px` : '0 0 0';
    pane.replaceChildren();
    $('#info-heading').textContent = 'Properties';
    if (sel.length !== 1) {
        pane.append(el('div', { className: 'info-empty', textContent: sel.length ? 'Multiple selection.' : 'No selection.' }));
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
                pane.append(note(`${sel.data('label')}: ${up ? 'host → device' : 'device → host'}`));
            }
    }
}

/**
 * The name of a data node is the name of its object. An existing name joins that object; a random stream and data cannot share one.
 */
function rename(c, name) {
    const result = renameObject(S.design, c, name);
    if (result.refused) say(result.refused);
    else if (result.merged) say(`${result.merged}: updates of one object`);
    changed({ structure: true, inspector: true });
}

/** The update position of a data node in its object, with a button to detach it. */
function objectNotes(pane, c) {
    const members = membersOf(S.design, objectKey(c));
    if (members.length < 2) return;
    const v = S.result.versions?.[c.id];
    const detach = el('button', { textContent: 'Detach', title: 'New object with its own name' });
    detach.onclick = () => { const name = detachNode(S.design, c); if (name) say(`${name}: detached`); changed({ structure: true, inspector: true }); };
    pane.append(note(`${v ? `Update ${v.version} of ${v.count}` : 'Allocation'} (${members.length} nodes)`),
        el('div', { className: 'info-buttons' }, detach));
}

export function inspectContainer(pane, c) {
    if (c.kind === 'PRNG') {
        pane.append(tip('skepu::PRNG. Calls that share a stream form a chain.', field('Name', text(c.label, v => rename(c, v)))),
            field('Seed', text(c.init, v => { c.init = v; changed(); }, { placeholder: 'default' })));
        return;
    }
    const eff = effective(c.id);
    const inferred = c.auto && eff !== c;
    const producer = S.design.edges.find(e => e.to === c.id && e.role === 'out');
    const fixed = inferred ? { disabled: true } : {};
    const edited = () => { syncObject(S.design, c); changed({ inspector: false }); };            // the nodes of an object share one format
    const sizeInputs = () => eff.size.map((n, i) => number(n, v => { c.size[i] = v; edited(); }, { style: 'width: 6.5em', ...fixed }));
    pane.append(
        field('Name', text(c.label, v => rename(c, v))),
        tip('Take kind, type and size from the call that writes the data', field('Format', checkbox(c.auto, v => {
            c.auto = v;
            if (!v) Object.assign(c, { kind: eff.kind, type: eff.type, size: [...eff.size] });       // manual: keep the current format
            syncObject(S.design, c);
            changed({ structure: true, inspector: true });
        }), el('span', { textContent: 'inferred' }))),
        field('Kind', select(eff.kind, CONTAINER_KINDS, v => { c.kind = v; c.size = v === 'Vector' ? [c.size.reduce((a, b) => a * b, 1)] : defaultSize(v); syncObject(S.design, c); changed({ inspector: true }); }, fixed)),
        field('Element type', select(eff.type, ELEMENT_TYPES, v => { c.type = v; edited(); }, fixed)),
        field({ Vector: 'Size', Matrix: 'Rows × columns', Tensor3: 'Size i × j × k', Tensor4: 'Size i × j × k × l' }[eff.kind] ?? 'Size', ...sizeInputs()),
        field('Initial value', text(c.init, v => { c.init = v; edited(); })),
        ...(c.auto && !inferred && producer ? [note('Format not inferable.')] : []));
    objectNotes(pane, c);
}

export function inspectLoop(pane, l) {
    pane.append(
        tip('Trace region (SKEPU_TRACE_SCOPE)', field('Name', text(l.label, v => { l.label = v; changed(); }))),
        field('Loop', checkbox(!l.scope, v => { l.scope = !v; if (v && !(+l.count > 1)) l.count = 10; changed({ structure: true, inspector: true }); })));
    if (!l.scope) pane.append(field('Iterations', number(l.count, v => { l.count = v; changed(); })));
}

export function inspectEdge(pane, e) {
    if (!e) return;
    const target = find(e.to), from = find(e.from);
    pane.append(note(`${from?.label} → ${target?.label}`));
    if (e.role === 'out' || e.role === 'scalar' || e.role === 'random') { pane.append(note({ out: 'Output', scalar: 'Result', random: 'Random stream' }[e.role])); return; }
    const roles = Object.keys(PATTERNS[target.pattern]?.roles ?? {});
    pane.append(tip(e.role === 'proxy' ? 'Random access to the container' : e.role === 'elwise' ? 'Elementwise parameter, in edge order' : '',
        field('Used as', select(e.role, roles, v => { e.role = v; changed({ structure: true, inspector: true }); }))));
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
    // an anonymous skeleton is named by the precompiler from its call
    if (!spec?.host) pane.append(tip('Created at the call, C++ only: skepu::Map<1>(f)(out, in). Traced as Type@file:line.',
        field('Anonymous', checkbox(isAnonymous(s, codeOptions), v => { s.anonymous = v; changed({ structure: true, inspector: true }); }))));
    pane.append(
        ...(isAnonymous(s, codeOptions) ? [] : [field('Name', text(s.label, v => { s.label = v; changed(); }))]),
        tip(spec?.doc ?? '', field('Pattern', select(s.pattern, Object.entries(PATTERNS).map(([key, p]) => [key, p.title]), v => {
            const fresh = defaultSkeleton(v, s.id, s.label);
            Object.assign(s, { pattern: v, params: fresh.params, indexed: false });
            changed({ structure: true, inspector: true });
        }))));
    if (!spec?.host) pane.append(field('Backend', select(s.backend, BACKENDS, v => { s.backend = v; changed(); })));

    const call = S.result.program?.calls.find(c => c.skeleton.id === s.id);
    if (s.pattern === 'Map')
        pane.append(field('Index', checkbox(s.indexed || call?.elwise.length === 0, v => { s.indexed = v; changed({ inspector: true }); }, { disabled: call?.elwise.length === 0 })));
    if (spec?.roles.random)
        if (call?.random) pane.append(field('Random stream', el('span', { textContent: call.random.label })));
    if (s.pattern === 'MapOverlap') {
        const p = s.params;
        pane.append(heading('Overlap'),
            field('Dimensions', select(String(dimsOf(s)), ['1', '2', '3', '4'], v => { p.dims = +v; changed({ inspector: true }); })),
            field('Radius', ...Array.from({ length: dimsOf(s) }, (_, i) => number(p.overlap[i] ?? 1, v => { p.overlap[i] = v; changed(); }, { style: 'width: 4.2em' }))),
            field('Edge', select(p.edge, withDefault(EDGES, 'Duplicate'), v => { p.edge = v; changed({ inspector: true }); })));
        if (p.edge === 'Pad') pane.append(field('Pad value', text(p.pad, v => { p.pad = v; changed(); }, { placeholder: '0' })));
        pane.append(field('Update mode', select(p.update ?? 'Normal', withDefault(UPDATE_MODES, 'Normal'), v => { p.update = v; changed(); })));
    }
    if (s.pattern === 'Map' || s.pattern === 'MapReduce') {
        const inputs = call?.elwise.length ?? S.design.edges.filter(e => e.to === s.id && e.role === 'elwise').length, n = (s.pattern === 'Map' ? 1 : 0) + inputs;
        const strides = stridesOf(s, n);
        if (n) pane.append(tip(s.pattern === 'Map' ? 'Per container: the output, then the inputs in edge order. Default 1.' : 'Per input, in edge order. Default 1.',
            field('Strides', ...strides.map((x, i) => number(x, v => { const next = [...strides]; next[i] = v; s.params.strides = next; changed({ structure: true, inspector: true }); }, { style: 'width: 4.2em' })))));
        if (s.pattern === 'MapReduce') pane.append(field('Start value', text(s.params.start ?? '', v => { s.params.start = v; changed(); }, { placeholder: '0' })));
    }
    if (s.pattern === 'MapPool') {
        const p = s.params;
        const pool = poolOf(s), stride = strideOf(s);
        pane.append(heading('Pool'),
            field('Dimensions', select(String(dimsOf(s)), ['2', '3', '4'], v => { p.dims = +v; changed({ structure: true, inspector: true }); })),
            field('Pool size', ...pool.map((n, i) => number(n, v => { p.pool = [...pool.map((x, j) => j === i ? Math.max(1, v) : x), ...(p.pool ?? []).slice(pool.length)]; changed({ inspector: true }); }, { style: 'width: 4.2em' }))),
            tip('Output size: (input − pool) / stride + 1', field('Stride', ...stride.map((n, i) => number(n, v => { p.stride = [...stride.map((x, j) => j === i ? Math.max(1, v) : x), ...(p.stride ?? []).slice(stride.length)]; changed({ inspector: true }); }, { style: 'width: 4.2em' })))));
    }
    if (s.pattern === 'Reduce') {
        const first = call?.elwise[0], matrix = first?.kind === 'Matrix';
        pane.append(heading('Reduction'));
        if (matrix) {
            pane.append(tip(call?.scalar ? '' : 'The rows or columns are reduced to the output vector (not supported by SkePY)',
                field('Along', select(s.params.mode ?? 'RowWise', withDefault([['RowWise', 'Rows'], ['ColWise', 'Columns']], 'RowWise'), v => { s.params.mode = v; changed({ structure: true, inspector: true }); }))));
            if (call?.scalar) pane.append(tip('Rows (or columns) with the first function, their results with the second',
                field('Two functions', checkbox(!!s.params.second, v => { s.params.second = v; changed({ structure: true, inspector: true }); }))));
        }
        pane.append(field('Start value', text(s.params.start ?? '', v => { s.params.start = v; changed(); })));
    }
    if (s.pattern === 'MapPairsReduce') {
        pane.append(heading('Reduction'), field('Along', select(s.params.mode ?? 'RowWise', withDefault([['RowWise', 'Rows'], ['ColWise', 'Columns']], 'RowWise'), v => { s.params.mode = v; changed({ structure: true, inspector: true }); })),
            field('Start value', text(s.params.start, v => { s.params.start = v; changed(); })));
    }
    if (s.pattern === 'Scan') {
        pane.append(heading('Scan'), field('Mode', select(s.params.mode, withDefault(['Inclusive', 'Exclusive'], 'Inclusive'), v => { s.params.mode = v; changed(); })),
            field('Start value', text(s.params.start, v => { s.params.start = v; changed(); })));
    }

    if (!spec?.host) pane.append(tip('Passed to every call after the data', heading('Uniform arguments')));
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
    if (!spec?.host) pane.append(table, el('div', { className: 'info-buttons' }, add));

    const functions = (S.result.ok ? S.result : S.lastGood)?.program.functions.filter(f => f.call.skeleton.id === s.id) ?? [];
    if (functions.length) pane.append(heading(`${spec?.host ? 'Host code' : `Function${functions.length > 1 ? 's' : ''}`} (${language === 'cpp' ? 'C++' : 'Python'})`));
    for (const f of functions) {
        const area = el('textarea', { className: 'body', value: f.bodies?.[language] ?? '', placeholder: f.stub[language], spellcheck: false });
        area.oninput = () => {
            s.bodies = s.bodies ?? {};
            s.bodies[f.suffix] = { ...(s.bodies[f.suffix] ?? {}), [language]: area.value };
            changed({ merge: `body:${s.id}:${f.suffix}:${language}` });          // keystrokes in a body are one undo step
        };
        if (!f.host) {
            const lambda = isLambda(f, codeOptions);
            pane.append(tip('Python: single-line bodies only', field('Lambda', checkbox(lambda, v => { s.lambda = { ...s.lambda, [f.suffix]: v }; changed({ inspector: true }); }))),
                        note(`${f.name}(${f.params.map(p => p.name).join(', ')})`));
        }
        pane.append(area);
    }
}

hooks.renderInspector = renderInspector;
