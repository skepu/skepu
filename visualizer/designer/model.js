// model.js — the design of a SkePU program as a graph, and its lowering to a program (validated, ordered, named). No DOM.
//
// The design is JSON:
//   containers  { id, label, kind: 'Vector'|'Matrix'|'Tensor3'|'Tensor4', type, size, init, auto }, or a random stream { id, label, kind: 'PRNG', init: seed }.
//               `auto`: kind, type and size are inferred from the writing call (inferOutput). A container node is an update of an object: nodes with
//               the same `object` key (default: the id) share one declaration, name and format, and their writers order them. A node that nothing
//               writes is the allocation of its object. A node has at most one writer.
//   skeletons   { id, label, pattern, backend, indexed, uniforms: [{ name, type, value }], params, bodies, anonymous, lambda }
//   edges       { id, from, to, role }
//               container -> skeleton: 'elwise' | 'proxy' | 'vertical' | 'horizontal' | 'read' (external)
//               random stream -> skeleton, or skeleton -> skeleton within a chain: 'random' (the stream and each call of its chain have one edge out)
//               skeleton -> container: 'out'; skeleton -> I/O block: 'scalar'
//   loops       { id, label, count, scope, parent }: a trace region (SKEPU_TRACE_SCOPE) around skeletons and regions. A loop repeats its calls `count`
//               times; a scope (`scope: true`) runs them once. Data belongs to the region of its writer (`dataRegion` of lower()).
//   positions   { id: { x, y } }: node places in the editor. The calls follow the data flow; the vertical position decides where it leaves a choice.
'use strict';

export const ELEMENT_TYPES = ['float', 'int', 'double'];
export const BACKENDS = ['', 'cpu', 'openmp', 'opencl'];     // '' is the one that the program is started with
export const EDGES = ['None', 'Cyclic', 'Duplicate', 'Pad'];

/**
 * The skeletons of a design. `roles`: what a container can be to one, as (min, max); `out`: how many containers it writes.
 */
export const PATTERNS = {
    Map:       { title: 'Map',       roles: { elwise: [0, 9], proxy: [0, 9], random: [0, 1] }, out: 1,
                 doc: 'f(a, b, …) per element, or per index without input.' },
    MapReduce: { title: 'MapReduce', roles: { elwise: [1, 9], proxy: [0, 9], random: [0, 1] }, out: 0, scalar: true,
                 doc: 'Map and reduction in one call; scalar result.' },
    Reduce:    { title: 'Reduce',    roles: { elwise: [1, 1] }, out: [0, 1], scalar: true, partial: true,
                 doc: 'Reduction to a scalar. A matrix can also be reduced to a vector (rows or columns) or with two functions.' },
    Scan:      { title: 'Scan',      roles: { elwise: [1, 1] }, out: 1,
                 doc: 'Prefix scan.' },
    MapOverlap: { title: 'MapOverlap', roles: { elwise: [1, 1], proxy: [0, 9], random: [0, 1] }, out: 1,
                 doc: 'f(region) per element (stencil).' },
    MapPool:   { title: 'MapPool',   roles: { elwise: [1, 1], proxy: [0, 9] }, out: 1,
                 doc: 'f(pool) per window; the output is smaller (pooling).' },
    External:  { title: 'External',  roles: { read: [0, 99] }, out: [0, 99], host: true,
                 doc: 'Host code with the connected data as input and output (skepu::external).' },
    MapPairs:  { title: 'MapPairs',  roles: { vertical: [1, 1], horizontal: [1, 1], proxy: [0, 9], random: [0, 1] }, out: 1,
                 doc: 'f(v, h) per pair of vertical and horizontal elements; matrix output.' },
    MapPairsReduce: { title: 'MapPairsReduce', roles: { vertical: [1, 1], horizontal: [1, 1], proxy: [0, 9], random: [0, 1] }, out: 1,
                 doc: 'MapPairs reduced along rows or columns by g(a, b); vector output.' },
    IO:        { title: 'I/O', roles: { read: [0, 99] }, out: [0, 0], host: true, io: true,
                 doc: 'Prints the connected data and scalar results (skepu::io::cout).' },
};

/** Container kinds by rank (1D–4D). A MapOverlap or MapPool of `dims` dimensions takes the kind of that rank. */
export const CONTAINER_KINDS = ['Vector', 'Matrix', 'Tensor3', 'Tensor4'];
export const rankOf = kind => CONTAINER_KINDS.indexOf(kind) + 1;
export const defaultSize = kind => ({ Vector: [1024], Matrix: [32, 32], Tensor3: [16, 16, 16], Tensor4: [8, 8, 8, 8] })[kind] ?? [1024];
export const INDEX_TYPES = ['Index1D', 'Index2D', 'Index3D', 'Index4D'];
/** MapOverlap dimensions (1–4) and the radius in each (default 1). */
export const dimsOf = s => [1, 2, 3, 4].includes(+s.params?.dims) ? +s.params.dims : 1;
export const overlapOf = s => Array.from({ length: dimsOf(s) }, (_, i) => s.params.overlap?.[i] ?? 1);
/** MapPool size and stride per dimension (the stride defaults to the pool size). */
/** The strides of a Map (the output, then each input) or a MapReduce (each input), `n` of them; 1 where not set. */
export const stridesOf = (s, n) => Array.from({ length: n }, (_, i) => Math.max(1, Math.floor(+s.params?.strides?.[i]) || 1));
export const UPDATE_MODES = ['Normal', 'RedBlack', 'Red', 'Black'];
export const poolOf = s => Array.from({ length: dimsOf(s) }, (_, i) => Math.max(1, +s.params.pool?.[i] || 2));
export const strideOf = s => Array.from({ length: dimsOf(s) }, (_, i) => Math.max(1, +s.params.stride?.[i] || poolOf(s)[i]));
/** The output size of a MapPool for an input of `size`; undefined if no window fits. */
export const pooledSize = (s, size) => { const out = size.map((n, i) => Math.floor((n - poolOf(s)[i]) / strideOf(s)[i]) + 1); return out.every(n => n >= 1) ? out : undefined; };
/** Whether a call has a scalar result: a MapReduce, or a Reduce without an output container. */
export const hasScalar = (design, s) => { const p = PATTERNS[s.pattern]; return !!p?.scalar && !(p.partial && design.edges.some(e => e.from === s.id && e.role === 'out')); };

export const defaultSkeleton = (pattern, id, label) => ({
    id, label, pattern, backend: '', indexed: false, uniforms: [],
    params: pattern === 'MapOverlap' ? { dims: 1, overlap: [1, 1, 1, 1], edge: 'Duplicate', pad: '0', update: 'Normal' }
          : pattern === 'MapReduce' ? { start: '' }
          : pattern === 'MapPool' ? { dims: 2, pool: [2, 2, 2, 2], stride: [2, 2, 2, 2] }
          : pattern === 'Reduce' ? { mode: 'RowWise', start: '', second: false }
          : pattern === 'Scan' ? { mode: 'Inclusive', start: '' }
          : pattern === 'MapPairsReduce' ? { mode: 'RowWise', start: '' } : {},
    bodies: {},
});

export const defaultContainer = (id, label, kind = 'Vector') => ({
    id, label, kind, type: 'float', size: defaultSize(kind), init: '0',
});

/** A random stream (skepu::PRNG); `init` is the seed. */
export const defaultPrng = (id, label) => ({ id, label, kind: 'PRNG', type: '', size: [], init: '' });

export const emptyDesign = () => ({ version: 1, name: 'design', containers: [], skeletons: [], edges: [], loops: [], positions: {} });

// ── Objects ──────────────────────────────────────────────────────────────────

/** The object key of a container node. */
export const objectKey = c => c.object ?? c.id;

/** The nodes of an object, in design order. */
export const membersOf = (design, key) => design.containers.filter(c => objectKey(c) === key);

/** The objects of a design: Map(key -> nodes), in design order. */
export function objectsOf(design) {
    const objects = new Map();
    for (const c of design.containers) (objects.get(objectKey(c)) ?? objects.set(objectKey(c), []).get(objectKey(c))).push(c);
    return objects;
}

const freshKey = design => {
    const used = new Set(design.containers.map(objectKey));
    for (let n = 1; ; n++) if (!used.has(`o${n}`)) return `o${n}`;
};

/** A label not used by another object (`except` is ignored): `label`, `label_2`, … */
export function freeLabel(design, label, except = null) {
    const taken = new Set(design.containers.filter(c => objectKey(c) !== except).map(c => c.label));
    let name = label;
    for (let n = 2; taken.has(name); n++) name = `${label}_${n}`;
    return name;
}

/** Gives the nodes of the object of `node` the name and format of `node`. */
export function syncObject(design, node) {
    for (const c of membersOf(design, objectKey(node)))
        if (c !== node) Object.assign(c, { label: node.label, kind: node.kind, type: node.type, size: [...node.size], init: node.init });
}

/** Makes `node` and its object updates of the object `target`. */
function join(design, node, target) {
    const key = objectKey(target), mine = objectKey(node);
    const format = effectiveContainers(design).find(c => c.id === target.id);
    const moved = membersOf(design, mine);
    for (const c of moved) {
        const own = effectiveContainers(design).find(x => x.id === c.id);
        c.object = key; c.label = target.label;
        if (JSON.stringify([own.kind, own.type, own.size]) !== JSON.stringify([format.kind, format.type, format.size]))
            Object.assign(c, { kind: format.kind, type: format.type, size: [...format.size], auto: false });
        c.init = target.init;
    }
}

/**
 * Renames the object of `node`. An existing name joins that object. Returns { merged } or { refused }.
 */
export function renameObject(design, node, label) {
    label = String(label);
    const other = design.containers.find(c => c.label === label && objectKey(c) !== objectKey(node));
    if (!other) { for (const c of membersOf(design, objectKey(node))) c.label = label; return {}; }
    if ((other.kind === 'PRNG') !== (node.kind === 'PRNG')) return { refused: 'A random stream and data cannot be one object.' };
    join(design, node, other);
    return { merged: label };
}

/** Makes `a` and `b` updates of one object with the name and format of `a`. Returns { merged } or { refused }. */
export function joinObjects(design, a, b) {
    if (objectKey(a) === objectKey(b)) return { refused: `${b.label} is already an update of ${a.label}.` };
    if ((a.kind === 'PRNG') !== (b.kind === 'PRNG')) return { refused: 'A random stream and data cannot be one object.' };
    if (a.kind === 'PRNG') return { refused: 'A random stream has one node.' };
    join(design, b, a);
    return { merged: a.label };
}

/** Makes `node` an object of its own with a unique name. Returns the name (null if it already was). */
export function detachNode(design, node) {
    if (membersOf(design, objectKey(node)).length < 2) return null;
    node.label = freeLabel(design, node.label);
    node.object = freshKey(design);
    return node.label;
}

/** The last call of the random chain from `id` (a stream or a call): where the next call connects from. */
export function randomTail(design, id) {
    const seen = new Set();
    for (let cur = id; !seen.has(cur);) {
        seen.add(cur);
        const next = design.edges.find(e => e.role === 'random' && e.from === cur);
        if (!next) return cur;
        cur = next.to;
    }
    return id;
}

// ── Names ────────────────────────────────────────────────────────────────────

const RESERVED = new Set(['main', 'int', 'float', 'double', 'long', 'short', 'char', 'auto', 'class', 'struct', 'for', 'while', 'if', 'else',
    'return', 'skepu', 'std', 'def', 'import', 'from', 'in', 'is', 'not', 'and', 'or', 'lambda', 'None', 'True', 'False', 'print', 'sys', 'range']);

/** A label as a C++ and Python identifier distinct from `used` (which receives it). */
export function identifier(label, used, fallback = 'x') {
    let base = String(label || fallback).trim().replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '') || fallback;
    if (/^[0-9]/.test(base)) base = '_' + base;
    if (RESERVED.has(base)) base += '_';
    let name = base;
    for (let n = 2; used.has(name); n++) name = `${base}_${n}`;
    used.add(name);
    return name;
}

export const sizeOf = c => c.size.reduce((a, b) => a * b, 1);
const sameSize = (a, b) => JSON.stringify(a.size) === JSON.stringify(b.size);
const showSize = c => c.size.join(' × ');


// ── What a skeleton writes ───────────────────────────────────────────────────

/**
 * The container a skeleton writes: { kind, type, size }, or null if unknown. `inputs`: the read containers by role.
 */
export function inferOutput(skeleton, inputs) {
    const first = (inputs.elwise ?? [])[0];
    const copy = c => ({ kind: c.kind, type: c.type, size: [...c.size] });
    switch (skeleton.pattern) {
        case 'Map': {
            if (!first) return null;
            const [out, ...ins] = stridesOf(skeleton, 1 + (inputs.elwise ?? []).length);
            // strided: the output has as many strided elements as the input (a vector only: other kinds keep the size of the input)
            return first.kind === 'Vector' && (out !== 1 || ins[0] !== 1) ? { ...copy(first), size: [Math.floor(first.size[0] / ins[0]) * out] } : copy(first);
        }
        case 'Scan': return first ? copy(first) : null;
        case 'MapOverlap': {
            if (!first) return null;
            const out = copy(first);
            // MapOverlap: the output has the input's kind; with edge mode None it shrinks by the overlap on each side
            if (skeleton.params?.edge === 'None' && first.size.length === dimsOf(skeleton)) {
                const size = first.size.map((n, i) => n - 2 * overlapOf(skeleton)[i]);
                if (size.every(n => n >= 1)) out.size = size;
            }
            return out;
        }
        case 'MapPool': {
            if (!first || first.size.length !== dimsOf(skeleton)) return first ? copy(first) : null;
            const size = pooledSize(skeleton, first.size);
            return { ...copy(first), ...(size ? { size } : {}) };
        }
        case 'Reduce':               // rows or columns of a matrix to a vector
            return first?.kind === 'Matrix' ? { kind: 'Vector', type: first.type, size: [first.size[skeleton.params?.mode === 'ColWise' ? 1 : 0]] } : null;
        case 'MapPairs': {
            const v = (inputs.vertical ?? [])[0], h = (inputs.horizontal ?? [])[0];
            return v && h && v.kind === 'Vector' && h.kind === 'Vector' ? { kind: 'Matrix', type: v.type, size: [v.size[0], h.size[0]] } : null;
        }
        case 'MapPairsReduce': {
            const v = (inputs.vertical ?? [])[0], h = (inputs.horizontal ?? [])[0];
            return v && h && v.kind === 'Vector' && h.kind === 'Vector' ? { kind: 'Vector', type: v.type, size: [(skeleton.params?.mode === 'ColWise' ? h : v).size[0]] } : null;
        }
        default: return null;
    }
}

/**
 * The containers with the kind, type and size of the `auto` ones inferred from the writing call (the first edge in).
 */
export function effectiveContainers(design) {
    const raw = new Map(design.containers.map(c => [c.id, c]));
    const skeletons = new Map(design.skeletons.map(s => [s.id, s]));
    const done = new Map(), open = new Set();
    const resolve = id => {
        const c = raw.get(id);
        if (!c || !c.auto) return c;
        if (done.has(id)) return done.get(id);
        if (open.has(id)) return c;
        open.add(id);
        let result = c;
        const written = design.edges.find(e => e.to === id && e.role === 'out' && skeletons.has(e.from));
        if (written) {
            const inputs = {};
            for (const e of design.edges)
                if (e.to === written.from && e.role !== 'out' && e.from !== id && raw.has(e.from)) (inputs[e.role] ??= []).push(resolve(e.from));
            const inferred = inferOutput(skeletons.get(written.from), inputs);
            if (inferred) result = { ...c, ...inferred };
        }
        open.delete(id);
        done.set(id, result);
        return result;
    };
    return design.containers.map(c => resolve(c.id));
}

// ── Lowering ─────────────────────────────────────────────────────────────────

/**
 * Checks a design and makes its program (containers, skeletons, user functions, ordered steps). `errors` prevent a program,
 * `warnings` do not. Each has the `id` of its node or edge.
 */
export function lower(design) {
    const errors = [], warnings = [];
    const err = (id, msg) => errors.push({ id, msg });
    const warn = (id, msg) => warnings.push({ id, msg });

    const all = effectiveContainers(design);
    const byId = Object.fromEntries([...all, ...design.skeletons, ...design.loops].map(n => [n.id, n]));
    const containers = Object.fromEntries(all.map(c => [c.id, c]));
    const used = new Set();
    const names = {};
    // one variable per object
    const objects = new Map();                           // key -> the (effective) nodes of the object, in the order of the design
    for (const c of all) (objects.get(objectKey(c)) ?? objects.set(objectKey(c), []).get(objectKey(c))).push(c);
    for (const [key, nodes] of objects) { const name = identifier(nodes[0].label, used, 'data'); for (const c of nodes) names[c.id] = name; }
    for (const s of design.skeletons) names[s.id] = identifier(s.label, used, s.pattern.toLowerCase());

    // what each skeleton reads and writes
    const ins = {}, outs = {};
    for (const s of design.skeletons) { ins[s.id] = []; outs[s.id] = []; }
    // random chains: a stream and each call of its chain have one outgoing random edge
    const randomIn = {}, randomOut = {};
    const scalarIn = {};                 // an I/O block -> the edges from the calls whose scalar results it prints
    design.edges.forEach((e, i) => {
        const s = byId[e.to], c = byId[e.from];
        if (e.role === 'scalar') {
            if (!s || !PATTERNS[s.pattern]?.io || !c || !PATTERNS[c.pattern]?.scalar) return err(e.id, 'Scalar edges go from a Reduce or MapReduce to an I/O block.');
            (scalarIn[s.id] ??= []).push(e);
            return;
        }
        if (e.role === 'random') {
            if (!s || !ins[s.id]) return err(e.id, 'Random edges go to calls that draw random numbers.');
            if (!c || !(containers[c.id] ? c.kind === 'PRNG' : !!ins[c.id] && !!PATTERNS[c.pattern]?.roles.random)) return err(e.id, containers[c?.id] ? `${c.label} is data, not a random stream.` : 'Random edges start at a random stream or at the previous call of its chain.');
            (randomIn[s.id] ??= []).push(e);
            (randomOut[c.id] ??= []).push(e);
            return;
        }
        if (e.role === 'out') {
            const sk = byId[e.from], co = byId[e.to];
            if (!sk || !co || !outs[sk.id] || !containers[co.id]) return err(e.id, 'Output edges go from a skeleton to data.');
            if (co.kind === 'PRNG') return err(e.id, `${co.label}: calls do not write random streams.`);
            outs[sk.id].push({ container: co, edge: e, order: i });
        } else {
            if (!c || !s || !containers[c.id] || !ins[s.id]) return err(e.id, 'Input edges go from data to a skeleton.');
            ins[s.id].push({ container: c, role: e.role, edge: e, order: i });
        }
    });

    for (const [id, list] of Object.entries(randomOut))
        if (list.length > 1 && containers[id]) err(id, `${byId[id].label}: ${list.length} outgoing random edges (one allowed).`);

    const calls = {};
    for (const s of design.skeletons) {
        const spec = PATTERNS[s.pattern];
        if (!spec) { err(s.id, `${s.label}: unknown pattern ${s.pattern}.`); continue; }
        const mine = ins[s.id].sort((a, b) => a.order - b.order);
        const nm = c => c && ({ ...c, name: names[c.id], object: objectKey(c) });
        const of = role => mine.filter(i => i.role === role).map(i => nm(i.container));
        for (const [role, [min, max]] of Object.entries(spec.roles)) {
            if (role === 'random') continue;   
            const n = of(role).length;
            if (n < min || n > max) err(s.id, `${s.label}: ${n} ${role === 'elwise' ? '' : role + ' '}inputs, expected ${min === max ? min : `${min}–${max}`}.`);
        }
        for (const e of randomIn[s.id] ?? []) if (!spec.roles.random) err(e.id, `${s.label}: ${s.pattern} takes no random stream.`);
        if ((randomIn[s.id] ?? []).length > 1) err(s.id, `${s.label}: ${randomIn[s.id].length} incoming random edges (one allowed).`);
        if ((randomOut[s.id] ?? []).length > 1) err(s.id, `${s.label}: ${randomOut[s.id].length} outgoing random edges (one allowed).`);
        for (const i of mine) if (!(i.role in spec.roles)) err(i.edge.id, `${s.label}: ${s.pattern} has no ${i.role} input.`);
        for (const i of mine) if (i.container.kind === 'PRNG' && i.role in spec.roles)
            err(i.edge.id, `${i.container.label}: a random stream is not an input; use a random edge.`);
        const o = outs[s.id].map(x => nm(x.container));
        const [omin, omax] = Array.isArray(spec.out) ? spec.out : [spec.out, spec.out];
        if (o.length < omin || o.length > omax) err(s.id, spec.partial ? `${s.label}: at most one output.`
            : spec.scalar ? `${s.label}: scalar result, no output data.`
            : `${s.label}: ${o.length} outputs, expected ${omin}.`);

        const call = { skeleton: s, name: names[s.id], pattern: s.pattern, elwise: of('elwise'), proxies: of('proxy'),
            vertical: of('vertical')[0], horizontal: of('horizontal')[0], out: spec.host ? null : o[0] ?? null };
        if (spec.host) Object.assign(call, { host: true, reads: of('read'), writes: o });
        if (spec.io) call.io = true;
        call.scalar = hasScalar(design, s);                    // MapReduce, or a Reduce without output
        call.partial = !!spec.partial && !call.scalar;         // Reduce to a vector
        call.second = s.pattern === 'Reduce' && !!s.params.second && call.scalar;      // Reduce with two functions
        calls[s.id] = call;
        const isIndexed = s.pattern === 'Map' && (s.indexed || call.elwise.length === 0);
        call.indexed = isIndexed;
        // the stream at the start of this call's random chain, and the previous call of the chain
        call.random = null; call.randomAfter = null;
        if (spec.roles.random && randomIn[s.id]?.length) {
            const seen = new Set([s.id]);
            for (let e = randomIn[s.id][0]; e;) {
                const from = byId[e.from];
                if (containers[from.id]) { call.random = nm(from); break; }
                if (seen.has(from.id)) { err(e.id, `${s.label}: random edges form a cycle.`); break; }
                seen.add(from.id);
                if (e === randomIn[s.id][0]) call.randomAfter = from.id;
                e = randomIn[from.id]?.[0];
                if (!e) err(s.id, `${s.label}: ${from.label} has no random stream.`);
            }
        }

        // sizes and types
        const first = call.elwise[0];
        if (s.pattern === 'Map' || s.pattern === 'Scan' || s.pattern === 'MapOverlap') {
            const ref = call.out;
            const shrinks = s.pattern === 'MapOverlap' && s.params.edge === 'None';          // edge mode None: the output shrinks
            const strides = s.pattern === 'Map' ? stridesOf(s, 1 + call.elwise.length) : null;          // output, then inputs
            for (const [i, x] of call.elwise.entries()) if (ref && !shrinks && Math.floor(sizeOf(x) / (strides?.[1 + i] ?? 1)) !== Math.floor(sizeOf(ref) / (strides?.[0] ?? 1)))
                err(s.id, `${s.label}: input ${x.label} (${showSize(x)}) and output ${ref.label} (${showSize(ref)}) differ in size.`);
            if (s.pattern === 'MapOverlap' && ref && first && s.params.edge !== 'None' && !sameSize(first, ref))
                err(s.id, `${s.label}: edge mode ${s.params.edge} requires an output of the input size.`);
            if (shrinks && ref && first && first.kind === ref.kind) {
                const want = first.size.map((n, i) => n - 2 * overlapOf(s)[i]);
                if (want.some(n => n < 1)) err(s.id, `${s.label}: overlap exceeds ${first.label} (${showSize(first)}).`);
                else if (JSON.stringify(want) !== JSON.stringify(ref.size))
                    err(s.id, `${s.label}: edge mode None requires an output of ${want.join(' × ')} for ${first.label} (${showSize(first)}), not ${showSize(ref)}.`);
            }
            if (s.pattern === 'MapOverlap' && first && first.kind !== CONTAINER_KINDS[dimsOf(s) - 1])
                err(s.id, `${s.label}: ${dimsOf(s)}D MapOverlap requires a ${CONTAINER_KINDS[dimsOf(s) - 1]}, not a ${first.kind}.`);
            if (s.pattern === 'Scan' && first && first.kind !== 'Vector') warn(s.id, `${s.label}: Scan treats a ${first.kind} as one sequence.`);
        }
        if (s.pattern === 'MapPool' && first) {
            const dims = dimsOf(s);
            if (first.kind !== CONTAINER_KINDS[dims - 1]) err(s.id, `${s.label}: ${dims}D MapPool requires a ${CONTAINER_KINDS[dims - 1]}, not a ${first.kind}.`);
            else {
                const want = pooledSize(s, first.size);
                if (!want) err(s.id, `${s.label}: pool ${poolOf(s).join(' × ')} exceeds ${first.label} (${showSize(first)}).`);
                else if (call.out && call.out.kind === first.kind) {
                    const room = call.out.size.every((n, i) => first.size[i] >= poolOf(s)[i] + strideOf(s)[i] * (n - 1));
                    if (!room) err(s.id, `${s.label}: output ${showSize(call.out)} does not fit ${first.label} (${showSize(first)}); pool ${poolOf(s).join(' × ')} and stride ${strideOf(s).join(' × ')} give ${want.join(' × ')}.`);
                    else if (JSON.stringify(want) !== JSON.stringify(call.out.size)) warn(s.id, `${s.label}: pool and stride give ${want.join(' × ')}; output ${call.out.label} is ${showSize(call.out)}.`);
                }
            }
        }
        if (s.pattern === 'Reduce' && first) {
            if (call.partial || call.second) {
                if (first.kind !== 'Matrix') err(s.id, `${s.label}: ${call.partial ? 'Reduce to a vector' : 'Reduce with two functions'} requires a Matrix, not a ${first.kind}.`);
                else if (call.partial && call.out) {
                    const along = first.size[s.params.mode === 'ColWise' ? 1 : 0];
                    if (call.out.kind !== 'Vector' || call.out.size[0] !== along)
                        err(s.id, `${s.label}: reducing the ${s.params.mode === 'ColWise' ? 'columns' : 'rows'} of ${first.label} (${showSize(first)}) gives a Vector of ${along}, not ${call.out.kind} ${showSize(call.out)}.`);
                }
            }
            if (s.params.second && !call.scalar) warn(s.id, `${s.label}: second function unused (vector result).`);
        }
        if (s.pattern === 'MapReduce' || s.pattern === 'Map') {
            const inStrides = s.pattern === 'Map' ? stridesOf(s, 1 + call.elwise.length).slice(1) : stridesOf(s, call.elwise.length);
            for (const [i, x] of call.elwise.entries()) if (first && Math.floor(sizeOf(x) / inStrides[i]) !== Math.floor(sizeOf(first) / inStrides[0]))
                err(s.id, `${s.label}: inputs ${x.label} (${showSize(x)}) and ${first.label} (${showSize(first)}) differ in size.`);
        }
        if (s.pattern === 'MapPairs' && call.vertical && call.horizontal && call.out) {
            if (call.vertical.kind !== 'Vector' || call.horizontal.kind !== 'Vector') err(s.id, `${s.label}: MapPairs inputs must be vectors.`);
            else if (call.out.kind !== 'Matrix' || call.out.size[0] !== call.vertical.size[0] || call.out.size[1] !== call.horizontal.size[0])
                err(s.id, `${s.label}: output must be a ${call.vertical.size[0]} × ${call.horizontal.size[0]} Matrix (${call.vertical.label} × ${call.horizontal.label}).`);
        }
        if (s.pattern === 'MapPairsReduce' && call.vertical && call.horizontal && call.out) {
            const along = s.params.mode === 'ColWise' ? call.horizontal : call.vertical;
            if (call.vertical.kind !== 'Vector' || call.horizontal.kind !== 'Vector') err(s.id, `${s.label}: MapPairsReduce inputs must be vectors.`);
            else if (call.out.kind !== 'Vector' || call.out.size[0] !== along.size[0])
                err(s.id, `${s.label}: output must be a Vector of ${along.size[0]} (${along.label}).`);
        }
        if (s.pattern === 'Map' && call.indexed && !call.out) { /* already an error */ }
        if (s.pattern === 'Map' && call.indexed && call.out)
            call.indexType = INDEX_TYPES[Math.max(1, rankOf(call.out.kind)) - 1];
        call.elementType = (call.out ?? first ?? call.vertical)?.type ?? 'float';
        // a call reading and writing one object (an update reads its predecessor); in place is valid for a Map
        const written = writesOf(call);
        for (const x of readsOf(call)) if (written.some(w => w.id === x.id))
            err(s.id, `${s.label}: reads and writes ${x.label} as one update.`);
        if (call.out && s.pattern === 'Map' && call.proxies.some(x => x.object === call.out.object)) warn(s.id, `${s.label}: proxy reads ${call.out.label}, which it writes.`);
        else if (call.out && s.pattern !== 'Map' && [...call.elwise, ...call.proxies, call.vertical, call.horizontal].some(x => x && x.object === call.out.object))
            warn(s.id, `${s.label}: ${call.out.label} is read and written in place; ${s.pattern} is not elementwise.`);
    }

    // scalar results printed by I/O blocks
    for (const s of design.skeletons) if (calls[s.id]?.io) calls[s.id].scalars = (scalarIn[s.id] ?? []).filter(e => {
        if (calls[e.from]?.scalar === false) { err(e.id, `${calls[e.from].skeleton.label} has a vector result: connect its data to ${s.label}.`); return false; }
        return true;
    }).map(e => calls[e.from]).filter(Boolean);

    // an update has one writer
    for (const c of all) {
        const writers = design.skeletons.filter(s => calls[s.id] && writesOf(calls[s.id]).some(x => x.id === c.id));
        if (writers.length > 1) err(c.id, `${c.label}: written by ${writers.map(w => w.label).join(' and ')}; an update has one writer.`);
    }

    // call order: data flow first, then vertical position; a loop keeps its calls together
    const { order, cycles, conflicts, versions, unordered } = orderCalls(design, calls);
    for (const s of cycles) warn(s.id, `${s.label}: dependence cycle, ordered by position.`);
    // versions of an object: the written nodes in order; an unwritten node is the allocation
    const writerOf = new Map();                          // node id -> the call that writes it
    for (const s of design.skeletons) if (calls[s.id]) for (const c of writesOf(calls[s.id])) if (!writerOf.has(c.id)) writerOf.set(c.id, s);
    for (const { a, b } of unordered)
        warn(b, `${byId[b].label}: write order of ${writerOf.get(a).label} and ${writerOf.get(b).label} is undetermined; position decides.`);
    const versionOf = {};
    for (const list of versions.values()) {
        const first = byId[list[0]];
        list.forEach((id, i) => {
            const c = byId[id];
            if (i === 0) return;
            if (!writerOf.has(id) && !writerOf.has(list[0])) err(id, `${c.label}: several allocations of one object.`);
            if (JSON.stringify([c.kind, c.type, c.size]) !== JSON.stringify([first.kind, first.type, first.size]))
                err(id, `${c.label}: ${c.kind === 'PRNG' || first.kind === 'PRNG' ? 'a random stream and data cannot be one object' : `${first.kind}<${first.type}> ${showSize(first)} and ${c.kind}<${c.type}> ${showSize(c)} differ; updates of one object share a format`}.`);
        });
        const written = list.filter(id => writerOf.has(id));
        written.forEach((id, i) => { versionOf[id] = { version: i + 1, count: written.length }; });
    }
    for (const l of conflicts) err(l.id, `${l.label}: calls are not consecutive (dependence cycle).`);
    const chainOf = id => {
        const chain = [];
        for (let p = parentLoop(design, id); p; p = parentLoop(design, p.id)) chain.unshift(p);
        return chain;
    };
    const root = { steps: [] };
    const stack = [];                // the loops that are open: their steps
    const left = new Set();          // the loops that were open and are closed
    for (const s of order) {
        const chain = chainOf(s.id);
        let k = 0;
        while (k < stack.length && k < chain.length && stack[k].loop.id === chain[k].id) k++;
        for (const gone of stack.splice(k)) left.add(gone.loop.id);          // leave the loops that this call is not in
        for (; k < chain.length; k++) {
            if (left.has(chain[k].id)) err(chain[k].id, `${chain[k].label}: calls are not consecutive.`);
            const scope = !!chain[k].scope;
            const step = { type: 'loop', loop: chain[k], label: chain[k].label, scope, count: scope ? 1 : Math.max(1, Math.floor(+chain[k].count) || 1), depth: k,
                           loops: chain.slice(0, k).filter(l => !l.scope).length, steps: [] };
            (k === 0 ? root : stack[k - 1]).steps.push(step);
            stack.push(step);
        }
        (stack.length ? stack[stack.length - 1] : root).steps.push({ type: 'call', call: calls[s.id] });
    }

    // the data that a call writes is in the region of the call
    const dataRegion = {};
    for (const c of all) dataRegion[c.id] = writerOf.get(c.id)?.parent ?? null;
    for (const l of design.loops) if (!order.some(s => chainOf(s.id).some(x => x.id === l.id))) warn(l.id, `${l.label}: no calls.`);

    // what is read before it is written
    const written = new Set();
    for (const s of order) {
        for (const x of readsOf(calls[s.id]))
            if (!written.has(x.id) && design.skeletons.some(w => writesOf(calls[w.id]).some(y => y.id === x.id)))
                warn(s.id, `${s.label}: reads ${x.label} before it is written (initial value).`);
        for (const x of writesOf(calls[s.id])) written.add(x.id);
    }
    for (const c of design.containers)
        if (!design.edges.some(e => e.from === c.id || e.to === c.id)) warn(c.id, `${c.label}: unused.`);

    // the user functions
    const functions = [];
    for (const s of order) functions.push(...userFunctions(calls[s.id], names, used));
    // the objects as declared: the format of the first node, and its updates
    const program = {
        name: design.name,
        containers: [...objects.entries()].map(([key, nodes]) => {
            const list = versions.get(key) ?? nodes.map(c => c.id), first = byId[list[0]];
            return { ...first, name: names[first.id], object: key, nodes: [...list, ...nodes.map(c => c.id).filter(id => !list.includes(id))] };
        }),
        calls: order.map(s => calls[s.id]), steps: root.steps, functions,
    };
    return { ok: errors.length === 0, errors, warnings, program, dataRegion, versions: versionOf };
}

/**
 * The calls in program order: the data flow gives a partial order and the vertical position decides the rest. An update is written before it is
 * read; the updates of an object follow their writers; a read of an update precedes the next write. The calls of a loop stay together (the order
 * is made per level, a loop having the dependences of its calls).
 * Returns `cycles` (calls ordered by position because of a dependence cycle), `conflicts` (loops in such a cycle), `versions` (Map: object key ->
 * node ids, allocation then updates) and `unordered` ([{ a, b }]: consecutive updates the data flow does not order).
 */
export function orderCalls(design, calls) {
    const list = design.skeletons.filter(s => calls[s.id]);
    const y = new Map(list.map(s => [s.id, design.positions[s.id]?.y ?? 0]));
    const loopOf = id => design.skeletons.find(s => s.id === id)?.parent ?? design.loops.find(l => l.id === id)?.parent ?? null;
    const chainOf = id => { const chain = [id]; for (let p = loopOf(id); p; p = loopOf(p)) chain.unshift(p); return chain; };      // [outermost loop, ..., id]

    // the pairs (a before b) of calls: first what the data flow says (an update is written before it is read)
    const before = [];
    const writers = new Map(), readers = new Map();                  // node id -> the calls that write / read it
    for (const s of list) {
        for (const c of writesOf(calls[s.id])) (writers.get(c.id) ?? writers.set(c.id, []).get(c.id)).push(s.id);
        for (const c of readsOf(calls[s.id])) (readers.get(c.id) ?? readers.set(c.id, []).get(c.id)).push(s.id);
    }
    for (const [node, ws] of writers) for (const w of ws) for (const r of readers.get(node) ?? []) if (r !== w) before.push([w, r]);
    for (const s of list) if (calls[s.id].randomAfter && calls[calls[s.id].randomAfter]) before.push([calls[s.id].randomAfter, s.id]);
    for (const s of list) for (const source of calls[s.id].scalars ?? []) before.push([source.skeleton.id, s.id]);                     // a result is printed after it is made

    // updates of each object in data-flow order (a before b if the writer of a reaches the writer of b); otherwise by writer position, recorded in `unordered`
    const next = new Map();
    for (const [a, b] of before) (next.get(a) ?? next.set(a, []).get(a)).push(b);
    const reached = new Map();
    const reach = a => {
        if (!reached.has(a)) { const seen = new Set(), todo = [a]; while (todo.length) for (const t of next.get(todo.pop()) ?? []) if (!seen.has(t)) { seen.add(t); todo.push(t); } reached.set(a, seen); }
        return reached.get(a);
    };
    const writerOf = node => (writers.get(node) ?? [])[0];
    const earlier = (u, v) => !!writerOf(u) && !!writerOf(v) && reach(writerOf(u)).has(writerOf(v));       // (u is written before v)
    const versions = new Map(), unordered = [];
    const objects = new Map();
    for (const c of design.containers) (objects.get(c.object ?? c.id) ?? objects.set(c.object ?? c.id, []).get(c.object ?? c.id)).push(c.id);
    for (const [key, nodes] of objects) {
        const bare = nodes.filter(id => !writerOf(id)), written = nodes.filter(id => writerOf(id));
        const ordered = [];
        for (let rest = [...written]; rest.length;) {
            let minimal = rest.filter(u => !rest.some(v => v !== u && earlier(v, u)));
            if (!minimal.length) minimal = rest;
            minimal.sort((a, b) => y.get(writerOf(a)) - y.get(writerOf(b)));
            ordered.push(minimal[0]);
            rest = rest.filter(u => u !== minimal[0]);
        }
        ordered.slice(1).forEach((b, i) => { const a = ordered[i]; if (!earlier(a, b) && writerOf(a) !== writerOf(b)) unordered.push({ a, b }); });
        versions.set(key, [...bare, ...ordered]);
        // a read of an update precedes the next write
        const all = versions.get(key);
        for (let i = 0; i < all.length; i++) for (let j = i + 1; j < all.length; j++) {
            const later = writerOf(all[j]);
            if (!later) continue;
            for (const r of readers.get(all[i]) ?? []) if (r !== later) before.push([r, later]);
            const earliest = writerOf(all[i]);
            if (earliest && earliest !== later) before.push([earliest, later]);
        }
    }

    // per level: the items (calls and loops) and their precedence
    const levels = new Map();         // loop id (or null) -> { items: Set, edges: Map(item -> Set(item)) }
    const level = id => levels.get(id) ?? levels.set(id, { items: new Set(), edges: new Map() }).get(id);
    for (const s of list) { const chain = chainOf(s.id); chain.forEach((id, k) => level(k === 0 ? null : chain[k - 1]).items.add(id)); }
    for (const [a, b] of before) {
        const ca = chainOf(a), cb = chainOf(b);
        let k = 0;
        while (k < ca.length - 1 && k < cb.length - 1 && ca[k] === cb[k]) k++;
        if (ca[k] === cb[k]) continue;
        const l = level(k === 0 ? null : ca[k - 1]);
        (l.edges.get(ca[k]) ?? l.edges.set(ca[k], new Set()).get(ca[k])).add(cb[k]);
    }

    const members = new Map();        // item -> the calls in it
    const collect = id => members.get(id) ?? members.set(id, design.skeletons.some(s => s.id === id) ? [id] : [...(levels.get(id)?.items ?? [])].flatMap(collect)).get(id);
    const top = id => Math.min(...collect(id).map(c => y.get(c)));
    const cycles = [], conflicts = [];
    const order = [];
    const emit = loop => {
        const { items, edges } = level(loop);
        const waiting = new Map([...items].map(i => [i, 0]));
        for (const targets of edges.values()) for (const t of targets) waiting.set(t, waiting.get(t) + 1);
        const done = new Set();
        while (done.size < items.size) {
            const rest = [...items].filter(i => !done.has(i)).sort((a, b) => top(a) - top(b));
            let next = rest.find(i => waiting.get(i) === 0);
            if (!next) {                                                // cycle: position decides
                next = rest[0];
                const preds = i => [...edges].filter(([from, to]) => !done.has(from) && to.has(i)).map(([from]) => from);
                const walk = [];                                        // follow the waits until they return: the cycle
                let i = next;
                while (!walk.includes(i)) { walk.push(i); i = preds(i)[0]; }
                const ring = walk.slice(walk.indexOf(i));
                const loops = ring.filter(i => design.loops.some(l => l.id === i));
                if (loops.length) conflicts.push(...loops.map(i => design.loops.find(l => l.id === i)).filter(l => !conflicts.includes(l)));
                else cycles.push(...collect(next).map(c => design.skeletons.find(s => s.id === c)));
            }
            done.add(next);
            for (const t of edges.get(next) ?? []) waiting.set(t, waiting.get(t) - 1);
            if (design.skeletons.some(s => s.id === next)) order.push(design.skeletons.find(s => s.id === next)); else emit(next);
        }
    };
    emit(null);
    return { order, cycles, conflicts, versions, unordered };
}

function parentLoop(design, id) {
    const node = design.skeletons.find(s => s.id === id) ?? design.loops.find(l => l.id === id);
    return design.loops.find(l => l.id === node?.parent) ?? null;
}

/** The containers a call reads and writes (an external block: those it is connected to). */
export const readsOf = call => (call?.reads ?? [...(call?.elwise ?? []), ...(call?.proxies ?? []), call?.vertical, call?.horizontal]).filter(Boolean);
export const writesOf = call => call?.writes ?? (call?.out ? [call.out] : []);

const LETTERS = 'abcdefghijklmnopqrstuvwxyz';

/** The user functions of a call: name, parameters, return type and stub body. */
function userFunctions(call, names, used) {
    const s = call.skeleton, T = call.elementType;
    const uniforms = s.uniforms.filter(u => u.name).map(u => ({ name: u.name, type: u.type || 'float', role: 'uniform', value: u.value }));
    const proxyParams = call.proxies.map((c, i) => ({ name: identifier(`${{ Vector: 'vec', Matrix: 'table', Tensor3: 'ten3', Tensor4: 'ten4' }[c.kind] ?? 'vec'}_${i}`, new Set(), 'p'), type: c.type,
        role: 'proxy', container: c }));
    const letters = n => Array.from({ length: n }, (_, i) => LETTERS[i % 26]);
    // `stub`: the default body in C++ and Python; `bodies`: the user's, per language
    const fn = (suffix, params, returns, stub) => ({
        id: `${s.id}${suffix}`, call, suffix, name: identifier(`${call.name}${suffix}`, used), params, returns,
        stub: typeof stub === 'string' ? { cpp: stub, py: stub } : stub, bodies: s.bodies?.[suffix] ?? {},
    });
    switch (s.pattern) {
        case 'External':
            return [{ ...fn('_host', [], 'void', { cpp: '// TODO: host code', py: 'pass  # TODO: host code' }), host: true }];
        case 'Map': {
            const params = [];
            if (call.indexed) params.push({ name: 'index', type: call.indexType, role: 'index' });
            if (call.random) params.push({ name: 'random', type: 'Random', role: 'random' });
            letters(call.elwise.length).forEach((n, i) => params.push({ name: n, type: call.elwise[i].type, role: 'elwise' }));
            params.push(...proxyParams, ...uniforms);
            const first = call.elwise.find(c => c.type === T);
            const at = call.indexType === 'Index2D' ? 'row' : 'i';
            const stub = call.random ? { cpp: 'return random.getNormalized();', py: 'return random.get_normalized()' }
                : call.indexed ? { cpp: `return (${T})index.${at};`, py: T === 'int' ? `return index.${at}` : `return float(index.${at})` }
                : first ? { cpp: 'return a;', py: 'return a' } : { cpp: 'return 0;', py: 'return 0' };
            return [fn('_f', params, T, stub)];
        }
        case 'MapReduce': {
            const params = letters(call.elwise.length).map((n, i) => ({ name: n, type: call.elwise[i].type, role: 'elwise' }));
            if (call.random) params.unshift({ name: 'random', type: 'Random', role: 'random' });
            params.push(...proxyParams, ...uniforms);
            return [fn('_map', params, call.elwise[0].type, { cpp: 'return a;', py: 'return a' }),
                    fn('_reduce', [{ name: 'a', type: call.elwise[0].type, role: 'elwise' }, { name: 'b', type: call.elwise[0].type, role: 'elwise' }], call.elwise[0].type,
                       { cpp: 'return a + b;', py: 'return a + b' })];
        }
        case 'Reduce': case 'Scan': {
            const t = call.elwise[0]?.type ?? 'float';
            if (call.second)               // the first reduces within a row (or column), the second reduces the results
                return [fn('_row', [{ name: 'a', type: t, role: 'elwise' }, { name: 'b', type: t, role: 'elwise' }], t, { cpp: 'return a + b;', py: 'return a + b' }),
                        fn('_col', [{ name: 'a', type: t, role: 'elwise' }, { name: 'b', type: t, role: 'elwise' }], t, { cpp: 'return a + b;', py: 'return a + b' })];
            return [fn('_f', [{ name: 'a', type: t, role: 'elwise' }, { name: 'b', type: t, role: 'elwise' }], t, { cpp: 'return a + b;', py: 'return a + b' })];
        }
        case 'MapOverlap': {
            const dims = dimsOf(s), zeros = Array(dims).fill(0).join(', ');
            const params = [...(call.random ? [{ name: 'random', type: 'Random', role: 'random' }] : []),
                { name: 'r', type: `Region${dims}D`, role: 'region', elementType: call.elwise[0]?.type ?? 'float' }, ...proxyParams, ...uniforms];
            return [fn('_f', params, T, { cpp: `return r(${zeros});`, py: `return r[${zeros}]` })];
        }
        case 'MapPool': {
            const dims = dimsOf(s), zeros = Array(dims).fill(0).join(', ');
            const params = [{ name: 'p', type: `Pool${dims}D`, role: 'region', elementType: call.elwise[0]?.type ?? 'float' }, ...proxyParams, ...uniforms];
            return [fn('_f', params, T, { cpp: `return p(${zeros});`, py: `return p[${zeros}]` })];
        }
        case 'MapPairs': {
            const params = [...(call.random ? [{ name: 'random', type: 'Random', role: 'random' }] : []),
                { name: 'v', type: call.vertical?.type ?? 'float', role: 'elwise' }, { name: 'h', type: call.horizontal?.type ?? 'float', role: 'elwise' },
                ...proxyParams, ...uniforms];
            return [fn('_f', params, T, { cpp: 'return v * h;', py: 'return v * h' })];
        }
        case 'MapPairsReduce': {
            const params = [...(call.random ? [{ name: 'random', type: 'Random', role: 'random' }] : []),
                { name: 'v', type: call.vertical?.type ?? 'float', role: 'elwise' }, { name: 'h', type: call.horizontal?.type ?? 'float', role: 'elwise' },
                ...proxyParams, ...uniforms];
            return [fn('_map', params, T, { cpp: 'return v * h;', py: 'return v * h' }),
                    fn('_reduce', [{ name: 'a', type: T, role: 'elwise' }, { name: 'b', type: T, role: 'elwise' }], T, { cpp: 'return a + b;', py: 'return a + b' })];
        }
    }
    return [];
}


// ── Transfers ────────────────────────────────────────────────────────────────

const PLACE = { cpu: 'host', openmp: 'host', opencl: 'device' };

/** Where a call runs: 'host' or 'device'. A call without a backend uses `unset` ('host'); undefined if unknown. */
export const placeOf = (call, unset) => call.host ? 'host' : PLACE[call.skeleton.backend] ?? unset;

/**
 * Tracks where the current value of each container is and what moves for a call: a device call (OpenCL) reading host-written data, or the reverse.
 * The program has no transfers (SkePU makes them); the graph and the trace show them. `unset`: the place of calls without a backend (see placeOf);
 * without it such calls are not followed.
 */
export class Residency {
    constructor(program, unset) {
        this.unset = unset;
        this.valid = new Map(program.containers.map(c => [objectKey(c), new Set(['host'])]));        // by object: the updates are one piece of data
        this.copies = new Map();         // object -> Map(place -> the transfer that made the copy there: { container, direction, consumers })
        this.records = [];               // every transfer that was made
    }
    /**
     * The transfers a call needs: [{ container, direction }]; its outputs are then where it ran. A transfer anchors the data on that side:
     * each call reading that copy until the next write is a consumer (`records`: { container, direction, consumers }).
     */
    step(call) {
        const place = placeOf(call, this.unset), moves = [];
        for (const c of readsOf(call)) {
            if (!place) continue;
            const key = objectKey(c);
            const where = this.valid.get(key) ?? this.valid.set(key, new Set(['host'])).get(key);
            if (where.size && !where.has(place)) {
                const record = { container: c, direction: place === 'device' ? 'host-to-device' : 'device-to-host', consumers: new Set() };
                moves.push({ container: c, direction: record.direction });
                this.records.push(record);
                (this.copies.get(key) ?? this.copies.set(key, new Map()).get(key)).set(place, record);
            }
            where.add(place);
            this.copies.get(key)?.get(place)?.consumers.add(call.skeleton.id);
        }
        for (const c of writesOf(call)) { this.valid.set(objectKey(c), new Set(place ? [place] : [])); this.copies.delete(objectKey(c)); }
        return moves;
    }
}

/**
 * The transfers between calls on different explicit backends (calls without a backend are not followed; a loop is followed twice).
 * Returns [{ container, consumer, direction, consumers }]: the moved node, the first call that needed it and all calls reading the copy.
 */
export function transfers(program) {
    const found = new Map(), residency = new Residency(program);
    const run = steps => {
        for (const step of steps) {
            if (step.type !== 'call') { for (let pass = 0; pass < Math.min(2, step.count); pass++) run(step.steps); continue; }
            residency.step(step.call);
        }
    };
    run(program.steps);
    for (const r of residency.records) {
        const consumers = [...r.consumers], key = `${r.container.id}|${consumers[0]}|${r.direction}`;
        const known = found.get(key);
        if (known) known.consumers = [...new Set([...known.consumers, ...consumers])];
        else found.set(key, { container: r.container.id, consumer: consumers[0], direction: r.direction, consumers });
    }
    return [...found.values()];
}

// ── Duplicating ──────────────────────────────────────────────────────────────

/**
 * Copies nodes into the design (a region brings its contents) and returns the copy ids. Edges between copied nodes are copied. A copied call reads
 * what the original reads (shared data and streams) and writes a new container in the original's output format. A copied container whose writer
 * is not copied keeps its format. Copies are offset down and right, in the regions of the originals.
 */
export function duplicateNodes(design, ids) {
    const nodes = [...design.containers, ...design.skeletons, ...design.loops];
    const byId = new Map(nodes.map(n => [n.id, n]));
    const used = new Set([...nodes, ...design.edges].map(x => x.id));
    const fresh = prefix => { let n = 1; while (used.has(prefix + n)) n++; used.add(prefix + n); return prefix + n; };
    const labels = new Set(nodes.map(n => n.label));
    const labelOf = label => {
        const base = String(label).replace(/_copy\d*$/, '');
        for (let n = 1; ; n++) { const l = n === 1 ? `${base}_copy` : `${base}_copy${n}`; if (!labels.has(l)) { labels.add(l); return l; } }
    };

    // what is copied: the selection, and what is in the regions of it
    const picked = new Set([...ids].filter(id => byId.has(id)));
    for (let grew = true; grew;) {
        grew = false;
        for (const n of [...design.skeletons, ...design.loops]) if (n.parent && picked.has(n.parent) && !picked.has(n.id)) { picked.add(n.id); grew = true; }
    }
    if (!picked.size) return [];

    const effective = new Map(effectiveContainers(design).map(c => [c.id, c]));
    const copy = new Map();                 // original id -> copy
    const made = [];
    // the copies of the nodes of an object form an object of their own; a copy of one node is a new object
    const keys = new Set(design.containers.map(objectKey));
    const newKey = () => { for (let k = 1; ; k++) if (!keys.has(`o${k}`)) { keys.add(`o${k}`); return `o${k}`; } };
    const objectCopy = new Map();           // object key -> { key, label }
    const clone = (n, extra = {}) => {
        const isData = design.containers.includes(n), key = objectKey(n);
        const shared = isData && !extra.own && [...picked].filter(id => { const x = byId.get(id); return design.containers.includes(x) && objectKey(x) === key; }).length > 1;
        const copy = shared ? objectCopy.get(key) ?? objectCopy.set(key, { key: newKey(), label: labelOf(n.label) }).get(key) : null;
        const c = { ...JSON.parse(JSON.stringify(n)), id: fresh(isData ? 'c' : design.skeletons.includes(n) ? 's' : 'l'), label: copy ? copy.label : labelOf(n.label), ...extra };
        if (isData) delete c.object;
        if (copy) c.object = copy.key;
        delete c.own;
        const at = design.positions[n.id];
        if (at) design.positions[c.id] = { x: at.x + 40, y: at.y + 40 };
        made.push(c.id);
        return c;
    };
    const written = new Set(design.edges.filter(e => e.role === 'out' && picked.has(e.from)).map(e => e.to));
    const added = { containers: [], skeletons: [], loops: [], edges: [] };
    for (const n of nodes) {
        if (!picked.has(n.id)) continue;
        const isContainer = design.containers.includes(n), isLoop = design.loops.includes(n);
        let c;
        if (isContainer) {
            const producedOutside = n.auto && !written.has(n.id);      // writer not copied: keeps its format
            const e = effective.get(n.id);
            c = clone(n, producedOutside ? { kind: e.kind, type: e.type, size: [...e.size], auto: false } : {});
        } else c = clone(n);
        copy.set(n.id, c);
        (isContainer ? added.containers : isLoop ? added.loops : added.skeletons).push(c);
    }
    for (const c of copy.values()) if (c.parent !== undefined && c.parent !== null) c.parent = copy.get(c.parent)?.id ?? c.parent;

    for (const e of [...design.edges]) {
        const from = picked.has(e.from), to = picked.has(e.to);
        if (from && to) added.edges.push({ id: fresh('e'), from: copy.get(e.from).id, to: copy.get(e.to).id, role: e.role });
        else if (!from && to && e.role === 'random') {                                                                                       // the same stream: the copy continues the chain
            const tail = randomTail({ edges: [...design.edges, ...added.edges] }, e.from);
            added.edges.push({ id: fresh('e'), from: tail, to: copy.get(e.to).id, role: 'random' });
        }
        else if (!from && to && e.role !== 'out') added.edges.push({ id: fresh('e'), from: e.from, to: copy.get(e.to).id, role: e.role });      // the same input
        else if (from && !to && e.role === 'out') {                                                                                          // data of its own
            const original = effective.get(e.to);
            const out = clone(byId.get(e.to), { own: true, auto: true, kind: original.kind, type: original.type, size: [...original.size] });
            added.containers.push(out);
            added.edges.push({ id: fresh('e'), from: copy.get(e.from).id, to: out.id, role: 'out' });
        }
    }
    design.containers.push(...added.containers);
    design.skeletons.push(...added.skeletons);
    design.loops.push(...added.loops);
    design.edges.push(...added.edges);
    return made;
}

// ── A design to start from ───────────────────────────────────────────────────

/** Example: a stencil iterated over a vector, then summed. */
export function sampleDesign() {
    const d = emptyDesign();
    d.name = 'smoothing';
    d.containers.push({ ...defaultContainer('c1', 'signal'), size: [4096], init: '1.0f' },
                      { ...defaultContainer('c2', 'smoothed'), size: [4096], init: '0.0f', auto: true },
                      { ...defaultContainer('c3', 'scaled'), size: [4096], init: '0.0f', auto: true });
    d.skeletons.push({ ...defaultSkeleton('MapOverlap', 's1', 'smooth'), params: { dims: 1, overlap: [1, 1, 1, 1], edge: 'Duplicate', pad: '0' } },
                     { ...defaultSkeleton('Map', 's2', 'scale'), uniforms: [{ name: 'factor', type: 'float', value: '0.5f' }] },
                     { ...defaultSkeleton('MapReduce', 's3', 'energy') });
    d.loops.push({ id: 'l1', label: 'Smoothing', count: 10, parent: null });
    d.skeletons[0].parent = 'l1';
    d.edges.push({ id: 'e1', from: 'c1', to: 's1', role: 'elwise' }, { id: 'e2', from: 's1', to: 'c2', role: 'out' },
                 { id: 'e3', from: 'c2', to: 's2', role: 'elwise' }, { id: 'e4', from: 's2', to: 'c3', role: 'out' },
                 { id: 'e5', from: 'c3', to: 's3', role: 'elwise' });
    d.positions = { c1: { x: 0, y: 0 }, s1: { x: 0, y: 120 }, c2: { x: 0, y: 240 }, s2: { x: 0, y: 360 }, c3: { x: 0, y: 480 }, s3: { x: 0, y: 600 } };
    return d;
}

/** Every skeleton, each way of giving a Map its arguments, and a loop (the round-trip test design). */
export function allSkeletonsDesign() {
    const d = emptyDesign();
    d.name = 'all_skeletons';
    const AUTO = new Set(['c4', 'c5', 'c7', 'c8', 'c11', 'c14', 'c15']);       // containers whose format is inferred
    const vec = (id, label, n, init = '1.0f') => d.containers.push({ ...defaultContainer(id, label), size: [n], init, auto: AUTO.has(id) });
    const mat = (id, label, r, c, init = '0.0f') => d.containers.push({ ...defaultContainer(id, label, 'Matrix'), size: [r, c], init, auto: AUTO.has(id) });
    vec('c1', 'ramp', 1024, '0.0f'); vec('c2', 'noise', 1024, '0.0f'); vec('c3', 'table', 16, '2.0f'); vec('c4', 'values', 1024);
    vec('c5', 'smoothed', 1024, '0.0f'); mat('c6', 'grid', 32, 32); mat('c7', 'blurred', 32, 32); vec('c8', 'prefix', 1024, '0.0f');
    vec('c9', 'rows', 32); vec('c10', 'columns', 48); mat('c11', 'pairs', 32, 48);
    d.containers.push(defaultPrng('c12', 'noise_stream'));
    vec('c13', 'summary', 8);
    vec('c14', 'values', 1024);                                    // the update of `values` written by the loop (c4 precedes it): the same object
    d.containers[3].object = d.containers[13].object = 'o1';
    vec('c15', 'sums', 32);
    const sk = (pattern, id, label, patch = {}) => d.skeletons.push({ ...defaultSkeleton(pattern, id, label), ...patch });
    sk('Map', 's1', 'ramp_up');
    sk('Map', 's2', 'make_noise');
    sk('Map', 's3', 'mix', { uniforms: [{ name: 'weight', type: 'float', value: '0.25f' }] });
    sk('MapOverlap', 's4', 'smooth', { parent: 'l1', params: { dims: 1, overlap: [2, 1], edge: 'Pad', pad: '0.0f' } });
    sk('Map', 's5', 'damp', { parent: 'l1' });
    sk('Map', 's6', 'checker');
    sk('MapOverlap', 's7', 'blur', { params: { dims: 2, overlap: [1, 1], edge: 'Duplicate', pad: '0' } });
    sk('Reduce', 's8', 'total');
    sk('Scan', 's9', 'running', { params: { mode: 'Exclusive', start: '0.0f' } });
    sk('MapReduce', 's10', 'dot');
    sk('MapPairs', 's11', 'outer');
    sk('External', 's12', 'inspect');
    sk('MapPairsReduce', 's13', 'row_sums', { params: { mode: 'RowWise', start: '' } });
    sk('IO', 's14', 'I/O');
    d.loops.push({ id: 'l1', label: 'Relax', count: 4, parent: null });
    let e = 0;
    const edge = (from, to, role) => d.edges.push({ id: `e${++e}`, from, to, role });
    edge('s1', 'c1', 'out');
    edge('s2', 'c2', 'out'); edge('c12', 's2', 'random');
    edge('c1', 's3', 'elwise'); edge('c2', 's3', 'elwise'); edge('c3', 's3', 'proxy'); edge('s3', 'c4', 'out');
    edge('c4', 's4', 'elwise'); edge('s4', 'c5', 'out');
    edge('c5', 's5', 'elwise'); edge('s5', 'c14', 'out');
    edge('s6', 'c6', 'out');
    edge('c6', 's7', 'elwise'); edge('s7', 'c7', 'out');
    edge('c14', 's8', 'elwise');
    edge('c14', 's9', 'elwise'); edge('s9', 'c8', 'out');
    edge('c14', 's10', 'elwise'); edge('c5', 's10', 'elwise');
    edge('c8', 's12', 'read'); edge('s12', 'c13', 'out');
    edge('c9', 's11', 'vertical'); edge('c10', 's11', 'horizontal'); edge('s11', 'c11', 'out');
    edge('c9', 's13', 'vertical'); edge('c10', 's13', 'horizontal'); edge('s13', 'c15', 'out');
    edge('c13', 's14', 'read'); edge('c15', 's14', 'read'); edge('s8', 's14', 'scalar');
    d.skeletons.forEach((s, i) => { d.positions[s.id] = { x: 100, y: i * 110 }; });
    d.containers.forEach((c, i) => { d.positions[c.id] = { x: -200 + (i % 2) * 40, y: i * 90 }; });
    return d;
}

/** Pooling and the Reduce modes: a MapPool, row and column reductions to vectors, and a two-function reduction. */
export function poolingDesign() {
    const d = emptyDesign();
    d.name = 'pooling_and_reductions';
    const mat = (id, label, r, c, init, auto = false) => d.containers.push({ ...defaultContainer(id, label, 'Matrix'), size: [r, c], init, auto });
    const vec = (id, label, n, auto = false) => d.containers.push({ ...defaultContainer(id, label), size: [n], init: '1.0f', auto });
    mat('c1', 'image', 32, 48, '1.0f'); mat('c2', 'pooled', 16, 24, '0.0f', true);
    vec('c3', 'row_sums', 32, true); vec('c4', 'col_sums', 48, true);
    const sk = (pattern, id, label, patch = {}) => d.skeletons.push({ ...defaultSkeleton(pattern, id, label), ...patch });
    sk('MapPool', 's1', 'pool', { params: { dims: 2, pool: [2, 2], stride: [2, 2] } });
    sk('Reduce', 's2', 'row_total', { params: { mode: 'RowWise', start: '', second: false } });
    sk('Reduce', 's3', 'col_total', { params: { mode: 'ColWise', start: '', second: false } });
    sk('Reduce', 's4', 'max_row_sum', { params: { mode: 'RowWise', start: '', second: true } });
    sk('Reduce', 's5', 'total');
    sk('IO', 's6', 'I/O');
    let e = 0;
    const edge = (from, to, role) => d.edges.push({ id: `e${++e}`, from, to, role });
    edge('c1', 's1', 'elwise'); edge('s1', 'c2', 'out');
    edge('c1', 's2', 'elwise'); edge('s2', 'c3', 'out');
    edge('c1', 's3', 'elwise'); edge('s3', 'c4', 'out');
    edge('c1', 's4', 'elwise');
    edge('c3', 's5', 'elwise');
    edge('s4', 's6', 'scalar'); edge('s5', 's6', 'scalar');
    d.skeletons.forEach((s, i) => { d.positions[s.id] = { x: 100, y: i * 110 }; });
    d.containers.forEach((c, i) => { d.positions[c.id] = { x: -200 + (i % 2) * 40, y: i * 90 }; });
    return d;
}
