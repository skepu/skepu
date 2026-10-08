// model.js — the design of a SkePU program as a graph (containers and skeletons, with the data flow between them),
// and its lowering to a program: validated, ordered, named. No DOM: the editor, the tests and (later) a server
// use the same functions.
//
// The design is plain JSON:
//   containers  { id, label, kind: 'Vector'|'Matrix', type: 'float'|'int'|'double', size: [n] | [rows, cols], init, auto }
//               or a random stream: { id, label, kind: 'PRNG', init: the seed }
//               (`auto`: the kind, type and size are inferred from the call that writes the container, see inferOutput)
//               A container node is an update of an object, as a data node of a trace is: the nodes that have the same `object` (a hidden
//               key; a node that has none is an object of its own, its id is the key) are the same object (one declaration in the program,
//               one name, one format), and the calls order them: the versions of the object are the order in which its writers run. A node that
//               nothing writes is the allocation of the object (it comes first), and a node has at most one writer.
//   skeletons   { id, label, pattern, backend, indexed, uniforms: [{ name, type, value }], params: {...}, bodies: {...} }
//   edges       { id, from, to, role }   container -> skeleton: 'elwise' | 'proxy' | 'vertical' | 'horizontal' | 'read' (an external)
//                                         random stream -> skeleton: 'random' (the first Map that draws from it); a random stream is not
//                                         updated like data: the next Map that draws from it is connected from the one before it, skeleton ->
//                                         skeleton: 'random'. A stream and each Map in its chain has one edge out.
//                                         skeleton -> container: 'out' (what the skeleton writes; an external can write several)
//   loops       { id, label, count, scope, parent }   a region around skeletons (and regions): a trace region (SKEPU_TRACE_SCOPE). A loop
//               repeats its calls `count` times; a scope (`scope: true`) is a group of calls, run once. The data that a call writes is in the
//               region of the call (it is not stored: see `dataRegion` of lower())
//   positions   { id: { x, y } }   where the nodes are in the editor. The calls are in the order of the data flow; where that leaves a choice, the
//                                  vertical position (from the top) decides.
'use strict';

export const ELEMENT_TYPES = ['float', 'int', 'double'];
export const BACKENDS = ['', 'cpu', 'openmp', 'opencl'];     // '' is the one that the program is started with
export const EDGES = ['None', 'Cyclic', 'Duplicate', 'Pad'];

/**
 * The skeletons that a design can have. `roles`: what a container can be to it, and how many (min, max); `out`: how many
 * containers it writes (a reduction has a scalar result instead).
 */
export const PATTERNS = {
    Map:       { title: 'Map',       roles: { elwise: [0, 9], proxy: [0, 9], random: [0, 1] }, out: 1,
                 doc: 'f(a, b, ...) on each element of the inputs; with no input, on each index' },
    MapReduce: { title: 'MapReduce', roles: { elwise: [1, 9], proxy: [0, 9], random: [0, 1] }, out: 0, scalar: true,
                 doc: 'a Map and a reduction in one call; the result is a scalar' },
    Reduce:    { title: 'Reduce',    roles: { elwise: [1, 1] }, out: 0, scalar: true,
                 doc: 'the elements to one value' },
    Scan:      { title: 'Scan',      roles: { elwise: [1, 1] }, out: 1,
                 doc: 'prefix sums (or another associative function) of the elements' },
    MapOverlap: { title: 'MapOverlap', roles: { elwise: [1, 1], proxy: [0, 9], random: [0, 1] }, out: 1,
                 doc: 'f(region) for the neighbourhood of each element (a stencil)' },
    External:  { title: 'External',  roles: { read: [0, 99] }, out: [0, 99], host: true,
                 doc: 'host code that reads and writes data (skepu::external): the data that it reads is what the edges into it come from, what it writes is what the edges out of it go to' },
    MapPairs:  { title: 'MapPairs',  roles: { vertical: [1, 1], horizontal: [1, 1], proxy: [0, 9], random: [0, 1] }, out: 1,
                 doc: 'f(v, h) for each pair of an element of the vertical and one of the horizontal vector: a matrix' },
    MapPairsReduce: { title: 'MapPairsReduce', roles: { vertical: [1, 1], horizontal: [1, 1], proxy: [0, 9], random: [0, 1] }, out: 1,
                 doc: 'f(v, h) for each pair of an element of the vertical and one of the horizontal vector, reduced along the rows (a vector as long as the vertical one) or the columns (as long as the horizontal one) by a reduction g(a, b)' },
    IO:        { title: 'I/O', roles: { read: [0, 99] }, out: [0, 0], host: true, io: true,
                 doc: 'prints data (skepu::io::cout): an external block that reads what the edges into it come from, as in SkeVU; a Reduce or MapReduce can be connected to it to print its result' },
};

/** The kinds of data: Vector (1D), Matrix (2D), Tensor3 and Tensor4 (3D and 4D). A MapOverlap of `dims` dimensions is on the kind of that rank. */
export const CONTAINER_KINDS = ['Vector', 'Matrix', 'Tensor3', 'Tensor4'];
export const rankOf = kind => CONTAINER_KINDS.indexOf(kind) + 1;
export const defaultSize = kind => ({ Vector: [1024], Matrix: [32, 32], Tensor3: [16, 16, 16], Tensor4: [8, 8, 8, 8] })[kind] ?? [1024];
export const INDEX_TYPES = ['Index1D', 'Index2D', 'Index3D', 'Index4D'];
/** The dimensions of a MapOverlap (1 to 4), the radius of the overlap in each (a missing one is 1). */
export const dimsOf = s => [1, 2, 3, 4].includes(+s.params?.dims) ? +s.params.dims : 1;
export const overlapOf = s => Array.from({ length: dimsOf(s) }, (_, i) => s.params.overlap?.[i] ?? 1);

export const defaultSkeleton = (pattern, id, label) => ({
    id, label, pattern, backend: '', indexed: false, uniforms: [],
    params: pattern === 'MapOverlap' ? { dims: 1, overlap: [1, 1, 1, 1], edge: 'Duplicate', pad: '0' }
          : pattern === 'Scan' ? { mode: 'Inclusive', start: '' }
          : pattern === 'MapPairsReduce' ? { mode: 'RowWise', start: '' } : {},
    bodies: {},
});

export const defaultContainer = (id, label, kind = 'Vector') => ({
    id, label, kind, type: 'float', size: defaultSize(kind), init: '0',
});

/** A random stream (skepu::PRNG), a node like a container: connect it to the call that draws from it (a Map, MapReduce, MapOverlap, MapPairs or MapPairsReduce). `init` is the seed. */
export const defaultPrng = (id, label) => ({ id, label, kind: 'PRNG', type: '', size: [], init: '' });

export const emptyDesign = () => ({ version: 1, name: 'design', containers: [], skeletons: [], edges: [], loops: [], positions: {} });

// ── Objects: the container nodes that are updates of the same data ──────────

/** The key of the object that a container node is an update of. */
export const objectKey = c => c.object ?? c.id;

/** The nodes of an object, in the order of the design. */
export const membersOf = (design, key) => design.containers.filter(c => objectKey(c) === key);

/** The objects of a design: Map(key -> its nodes), in the order of the design. */
export function objectsOf(design) {
    const objects = new Map();
    for (const c of design.containers) (objects.get(objectKey(c)) ?? objects.set(objectKey(c), []).get(objectKey(c))).push(c);
    return objects;
}

const freshKey = design => {
    const used = new Set(design.containers.map(objectKey));
    for (let n = 1; ; n++) if (!used.has(`o${n}`)) return `o${n}`;
};

/** A label that no other object has: `label`, or `label_2`, `label_3`, ... (the object `except` does not count). */
export function freeLabel(design, label, except = null) {
    const taken = new Set(design.containers.filter(c => objectKey(c) !== except).map(c => c.label));
    let name = label;
    for (let n = 2; taken.has(name); n++) name = `${label}_${n}`;
    return name;
}

/** Makes the nodes of the object of `node` have the name and the format that `node` has (the properties are those of the object). */
export function syncObject(design, node) {
    for (const c of membersOf(design, objectKey(node)))
        if (c !== node) Object.assign(c, { label: node.label, kind: node.kind, type: node.type, size: [...node.size], init: node.init });
}

/** `node` and its object are another object's now: the nodes are updates of the data `target` is (its name, and the format that it has). */
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
 * Gives the object of `node` a name. A name that another object has makes them one object (the nodes are updates of the same data: the
 * ones of `node` join the other). Returns { merged: the object that they joined } or { refused: why not } (a random stream and data are
 * not the same object).
 */
export function renameObject(design, node, label) {
    label = String(label);
    const other = design.containers.find(c => c.label === label && objectKey(c) !== objectKey(node));
    if (!other) { for (const c of membersOf(design, objectKey(node))) c.label = label; return {}; }
    if ((other.kind === 'PRNG') !== (node.kind === 'PRNG')) return { refused: `${label} is ${other.kind === 'PRNG' ? 'a random stream' : 'data'}: ${node.kind === 'PRNG' ? 'a random stream' : 'data'} cannot be an update of it` };
    join(design, node, other);
    return { merged: label };
}

/** The data nodes `a` and `b` are updates of the same object, which has the name and the format of `a`. As renameObject(): { merged } or { refused }. */
export function joinObjects(design, a, b) {
    if (objectKey(a) === objectKey(b)) return { refused: `${b.label} is an update of ${a.label} already` };
    if ((a.kind === 'PRNG') !== (b.kind === 'PRNG')) return { refused: 'A random stream and data are not the same object' };
    if (a.kind === 'PRNG') return { refused: 'A random stream is one node: connect it to each Map that draws from it' };
    join(design, b, a);
    return { merged: a.label };
}

/** `node` is an object of its own again, with a name that no other has. Returns the name (null: it was one already). */
export function detachNode(design, node) {
    if (membersOf(design, objectKey(node)).length < 2) return null;
    node.label = freeLabel(design, node.label);
    node.object = freshKey(design);
    return node.label;
}

/** The end of the chain of random edges that starts at `id` (a random stream, or a Map in its chain): where the next Map that draws from the stream is connected from. */
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

/** A label as an identifier of C++ and Python, different from the ones in `used` (which gets it). */
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
const showSize = c => c.size.join(' x ');


// ── What a skeleton writes ───────────────────────────────────────────────────

/**
 * The container that a skeleton writes, from what it reads and from its own properties: { kind, type, size }, or null when that is
 * not known (a Map of indices has no input to take the size from; a reduction writes a scalar, not a container).
 * `inputs`: the containers that it reads, by role ({ elwise: [...], vertical: [...], ... }).
 */
export function inferOutput(skeleton, inputs) {
    const first = (inputs.elwise ?? [])[0];
    const copy = c => ({ kind: c.kind, type: c.type, size: [...c.size] });
    switch (skeleton.pattern) {
        case 'Map': case 'Scan': return first ? copy(first) : null;
        case 'MapOverlap': {
            if (!first) return null;
            const out = copy(first);
            // the dimensions of the overlap are the kind of the container: a mismatch is the user's to see, so keep the input's
            // (with no edge mode the output is smaller by the overlap on each side, if there is room for that)
            if (skeleton.params?.edge === 'None' && first.size.length === dimsOf(skeleton)) {
                const size = first.size.map((n, i) => n - 2 * overlapOf(skeleton)[i]);
                if (size.every(n => n >= 1)) out.size = size;
            }
            return out;
        }
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
 * The containers of a design with the kind, type and size of the `auto` ones filled in from the call that writes them (the first
 * edge into the container), in the order of the design. Everything else is the same objects. A container that its own call reads, or
 * that is in a cycle, keeps what it has.
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
 * Checks a design and makes the program of it: the containers, skeletons and user functions with their names, and the steps in
 * order (calls, inside the loops). Problems are in `errors` (the program cannot be made) and `warnings` (it can).
 * Each has the `id` of the node or edge that it is about.
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
    // the nodes of an object are one variable of the program: one name
    const objects = new Map();                           // key -> the (effective) nodes of the object, in the order of the design
    for (const c of all) (objects.get(objectKey(c)) ?? objects.set(objectKey(c), []).get(objectKey(c))).push(c);
    for (const [key, nodes] of objects) { const name = identifier(nodes[0].label, used, 'data'); for (const c of nodes) names[c.id] = name; }
    for (const s of design.skeletons) names[s.id] = identifier(s.label, used, s.pattern.toLowerCase());

    // what each skeleton reads and writes
    const ins = {}, outs = {};
    for (const s of design.skeletons) { ins[s.id] = []; outs[s.id] = []; }
    // a random stream is not updated like data: a call that draws from it is followed by the next call that does, by an edge between the two (a chain
    // that starts at the stream), so the stream has one edge out and so has each Map in the chain
    const randomIn = {}, randomOut = {};
    const scalarIn = {};                 // an I/O block -> the edges from the calls whose scalar results it prints
    design.edges.forEach((e, i) => {
        const s = byId[e.to], c = byId[e.from];
        if (e.role === 'scalar') {
            if (!s || !PATTERNS[s.pattern]?.io || !c || !PATTERNS[c.pattern]?.scalar) return err(e.id, 'a scalar edge goes from a Reduce or MapReduce (its result) to an I/O block');
            (scalarIn[s.id] ??= []).push(e);
            return;
        }
        if (e.role === 'random') {
            if (!s || !ins[s.id]) return err(e.id, 'a random edge goes to a call that draws random numbers');
            if (!c || !(containers[c.id] ? c.kind === 'PRNG' : !!ins[c.id] && !!PATTERNS[c.pattern]?.roles.random)) return err(e.id, containers[c?.id] ? `${c.label} is data: a random stream is what a call draws from` : 'a random edge goes from a random stream, or from the call before it in its chain');
            (randomIn[s.id] ??= []).push(e);
            (randomOut[c.id] ??= []).push(e);
            return;
        }
        if (e.role === 'out') {
            const sk = byId[e.from], co = byId[e.to];
            if (!sk || !co || !outs[sk.id] || !containers[co.id]) return err(e.id, 'an output edge goes from a skeleton to a container');
            if (co.kind === 'PRNG') return err(e.id, `${co.label} is a random stream: a call does not write it`);
            outs[sk.id].push({ container: co, edge: e, order: i });
        } else {
            if (!c || !s || !containers[c.id] || !ins[s.id]) return err(e.id, 'an input edge goes from a container to a skeleton');
            ins[s.id].push({ container: c, role: e.role, edge: e, order: i });
        }
    });

    for (const [id, list] of Object.entries(randomOut))
        if (list.length > 1 && containers[id]) err(id, `${byId[id].label}: ${list.length} random edges out: the calls that draw from a random stream are a chain, one after another`);

    const calls = {};
    for (const s of design.skeletons) {
        const spec = PATTERNS[s.pattern];
        if (!spec) { err(s.id, `unknown skeleton ${s.pattern}`); continue; }
        const mine = ins[s.id].sort((a, b) => a.order - b.order);
        const nm = c => c && ({ ...c, name: names[c.id], object: objectKey(c) });
        const of = role => mine.filter(i => i.role === role).map(i => nm(i.container));
        for (const [role, [min, max]] of Object.entries(spec.roles)) {
            if (role === 'random') continue;                               // (see below)
            const n = of(role).length;
            if (n < min || n > max) err(s.id, `${s.label}: ${n} ${role === 'elwise' ? 'input' : role} containers, a ${s.pattern} has ${min === max ? min : `${min} to ${max}`}`);
        }
        for (const e of randomIn[s.id] ?? []) if (!spec.roles.random) err(e.id, `${s.label}: a ${s.pattern} does not draw random numbers`);
        if ((randomIn[s.id] ?? []).length > 1) err(s.id, `${s.label}: ${randomIn[s.id].length} random edges in, a call draws from one random stream (the one before it in the chain)`);
        if ((randomOut[s.id] ?? []).length > 1) err(s.id, `${s.label}: ${randomOut[s.id].length} random edges out: the calls that draw from a random stream are a chain, one after another`);
        for (const i of mine) if (!(i.role in spec.roles)) err(i.edge.id, `${s.label}: a ${s.pattern} has no ${i.role} input`);
        for (const i of mine) if (i.container.kind === 'PRNG' && i.role in spec.roles)
            err(i.edge.id, `${i.container.label} is a random stream: connect it as the random numbers of a call`);
        const o = outs[s.id].map(x => nm(x.container));
        const [omin, omax] = Array.isArray(spec.out) ? spec.out : [spec.out, spec.out];
        if (o.length < omin || o.length > omax) err(s.id, spec.scalar
            ? `${s.label}: a ${s.pattern} has a scalar result, not an output container`
            : `${s.label}: ${o.length} output containers, a ${s.pattern} has ${omin}`);

        const call = { skeleton: s, name: names[s.id], pattern: s.pattern, elwise: of('elwise'), proxies: of('proxy'),
            vertical: of('vertical')[0], horizontal: of('horizontal')[0], out: spec.host ? null : o[0] ?? null };
        if (spec.host) Object.assign(call, { host: true, reads: of('read'), writes: o });
        if (spec.io) call.io = true;
        calls[s.id] = call;
        const isIndexed = s.pattern === 'Map' && (s.indexed || call.elwise.length === 0);
        call.indexed = isIndexed;
        // the random stream: the one that the chain of random edges back from this call starts at (and the call before it in the chain)
        call.random = null; call.randomAfter = null;
        if (spec.roles.random && randomIn[s.id]?.length) {
            const seen = new Set([s.id]);
            for (let e = randomIn[s.id][0]; e;) {
                const from = byId[e.from];
                if (containers[from.id]) { call.random = nm(from); break; }
                if (seen.has(from.id)) { err(e.id, `${s.label}: the random edges are a cycle, with no random stream to start at`); break; }
                seen.add(from.id);
                if (e === randomIn[s.id][0]) call.randomAfter = from.id;
                e = randomIn[from.id]?.[0];
                if (!e) err(s.id, `${s.label}: draws random numbers after ${from.label}, which has no random stream`);
            }
        }

        // sizes and types
        const first = call.elwise[0];
        if (s.pattern === 'Map' || s.pattern === 'Scan' || s.pattern === 'MapOverlap') {
            const ref = call.out;
            const shrinks = s.pattern === 'MapOverlap' && s.params.edge === 'None';          // (with no edge mode the output is smaller: see below)
            for (const x of call.elwise) if (ref && !shrinks && sizeOf(x) !== sizeOf(ref))
                err(s.id, `${s.label}: input ${x.label} has ${showSize(x)} elements, the output ${ref.label} has ${showSize(ref)}`);
            if (s.pattern === 'MapOverlap' && ref && first && s.params.edge !== 'None' && !sameSize(first, ref))
                err(s.id, `${s.label}: with the edge mode ${s.params.edge} the output has the size of the input`);
            if (shrinks && ref && first && first.kind === ref.kind) {
                const want = first.size.map((n, i) => n - 2 * overlapOf(s)[i]);
                if (want.some(n => n < 1)) err(s.id, `${s.label}: the overlap is more than ${first.label} (${showSize(first)}) has room for, with no edge mode`);
                else if (JSON.stringify(want) !== JSON.stringify(ref.size))
                    err(s.id, `${s.label}: with the edge mode None the output is smaller than the input by the overlap on each side: ${want.join(' x ')} for ${first.label} of ${showSize(first)}, not ${showSize(ref)}`);
            }
            if (s.pattern === 'MapOverlap' && first && first.kind !== CONTAINER_KINDS[dimsOf(s) - 1])
                err(s.id, `${s.label}: a ${dimsOf(s)}D MapOverlap is on a ${CONTAINER_KINDS[dimsOf(s) - 1]}, not a ${first.kind}`);
            if (s.pattern === 'Scan' && first && first.kind !== 'Vector') warn(s.id, `${s.label}: a Scan of a ${first.kind} (its elements in one sequence)`);
        }
        if (s.pattern === 'MapReduce' || s.pattern === 'Map')
            for (const x of call.elwise) if (first && sizeOf(x) !== sizeOf(first))
                err(s.id, `${s.label}: input ${x.label} has ${showSize(x)} elements, ${first.label} has ${showSize(first)}`);
        if (s.pattern === 'MapPairs' && call.vertical && call.horizontal && call.out) {
            if (call.vertical.kind !== 'Vector' || call.horizontal.kind !== 'Vector') err(s.id, `${s.label}: the inputs of a MapPairs are vectors`);
            else if (call.out.kind !== 'Matrix' || call.out.size[0] !== call.vertical.size[0] || call.out.size[1] !== call.horizontal.size[0])
                err(s.id, `${s.label}: the output of a MapPairs of ${call.vertical.label} (${call.vertical.size[0]}) and ${call.horizontal.label} (${call.horizontal.size[0]}) is a ${call.vertical.size[0]} x ${call.horizontal.size[0]} Matrix`);
        }
        if (s.pattern === 'MapPairsReduce' && call.vertical && call.horizontal && call.out) {
            const along = s.params.mode === 'ColWise' ? call.horizontal : call.vertical;
            if (call.vertical.kind !== 'Vector' || call.horizontal.kind !== 'Vector') err(s.id, `${s.label}: the inputs of a MapPairsReduce are vectors`);
            else if (call.out.kind !== 'Vector' || call.out.size[0] !== along.size[0])
                err(s.id, `${s.label}: reduced along the ${s.params.mode === 'ColWise' ? 'columns' : 'rows'}, the output is a Vector of the size of ${along.label} (${along.size[0]})`);
        }
        if (s.pattern === 'Map' && call.indexed && !call.out) { /* already an error */ }
        if (s.pattern === 'Map' && call.indexed && call.out)
            call.indexType = INDEX_TYPES[Math.max(1, rankOf(call.out.kind)) - 1];
        call.elementType = (call.out ?? first ?? call.vertical)?.type ?? 'float';
        // a call that reads and writes the same object (an update of it reads the one before)
        // (a Map is element by element, so in place is fine; through a proxy the function can read what it has already written)
        const written = writesOf(call);
        for (const x of readsOf(call)) if (written.some(w => w.id === x.id))
            err(s.id, `${s.label} reads and writes ${x.label} as the same update: what it writes is a later update of it (give another data node the same name)`);
        if (call.out && s.pattern === 'Map' && call.proxies.some(x => x.object === call.out.object)) warn(s.id, `${s.label} writes ${call.out.label}, which its function reads through a proxy`);
        else if (call.out && s.pattern !== 'Map' && [...call.elwise, ...call.proxies, call.vertical, call.horizontal].some(x => x && x.object === call.out.object))
            warn(s.id, `${s.label} reads and writes ${call.out.label} (in place): a ${s.pattern} is not element by element`);
    }

    // the scalar results that I/O blocks print; a result that none of them prints is printed by the call that makes it
    for (const s of design.skeletons) if (calls[s.id]?.io) calls[s.id].scalars = (scalarIn[s.id] ?? []).map(e => calls[e.from]).filter(Boolean);
    for (const call of Object.values(calls)) if (PATTERNS[call.pattern]?.scalar) call.printed = !Object.values(calls).some(c => c.io && c.scalars.includes(call));

    // an update has one writer
    for (const c of all) {
        const writers = design.skeletons.filter(s => calls[s.id] && writesOf(calls[s.id]).some(x => x.id === c.id));
        if (writers.length > 1) err(c.id, `${c.label} is written by ${writers.map(w => w.label).join(' and ')}: an update has one writer (for another write, make another data node with the same name)`);
    }

    // the order of the calls: the order of the data flow first, the place from the top where that leaves a choice; the loops are where their calls are
    const { order, cycles, conflicts, versions, unordered } = orderCalls(design, calls);
    for (const s of cycles) warn(s.id, `${s.label} is in a cycle of dependences: it comes in the order of its place from the top`);
    // the updates of an object: the versions (the nodes that are written, in order; a node that nothing writes is its allocation)
    const writerOf = new Map();                          // node id -> the call that writes it
    for (const s of design.skeletons) if (calls[s.id]) for (const c of writesOf(calls[s.id])) if (!writerOf.has(c.id)) writerOf.set(c.id, s);
    for (const { a, b } of unordered)
        warn(b, `${byId[b].label}: the data flow does not say whether ${writerOf.get(a).label} or ${writerOf.get(b).label} writes it first, so the order of their places from the top decides: the result depends on it (make one of the calls depend on the other)`);
    const versionOf = {};
    for (const list of versions.values()) {
        const first = byId[list[0]];
        list.forEach((id, i) => {
            const c = byId[id];
            if (i === 0) return;
            if (!writerOf.has(id) && !writerOf.has(list[0])) err(id, `${c.label} has a second node that nothing writes: only the first is the allocation of the object`);
            if (JSON.stringify([c.kind, c.type, c.size]) !== JSON.stringify([first.kind, first.type, first.size]))
                err(id, `${c.label}: ${c.kind === 'PRNG' || first.kind === 'PRNG' ? 'a random stream and data are' : `${first.kind}<${first.type}> ${showSize(first)} and ${c.kind}<${c.type}> ${showSize(c)} are`} not the same object (the updates of an object have one format)`);
        });
        const written = list.filter(id => writerOf.has(id));
        written.forEach((id, i) => { versionOf[id] = { version: i + 1, count: written.length }; });
    }
    for (const l of conflicts) err(l.id, `${l.label}: its calls are not consecutive (a call between them depends on them, and they on it)`);
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
            if (left.has(chain[k].id)) err(chain[k].id, `${chain[k].label}: its calls are not consecutive (another call comes between them)`);
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
    for (const l of design.loops) if (!order.some(s => chainOf(s.id).some(x => x.id === l.id))) warn(l.id, `${l.label} has no calls`);

    // what is read before it is written
    const written = new Set();
    for (const s of order) {
        for (const x of readsOf(calls[s.id]))
            if (!written.has(x.id) && design.skeletons.some(w => writesOf(calls[w.id]).some(y => y.id === x.id)))
                warn(s.id, `${s.label} reads ${x.label} before the call that writes it (it has its initial value)`);
        for (const x of writesOf(calls[s.id])) written.add(x.id);
    }
    for (const c of design.containers)
        if (!design.edges.some(e => e.from === c.id || e.to === c.id)) warn(c.id, `${c.label} is not used`);

    // the user functions
    const functions = [];
    for (const s of order) functions.push(...userFunctions(calls[s.id], names, used));
    // the objects, as the program declares them: the format of the first node (the allocation, or the first update), and the nodes that are updates of it
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
 * The calls in the order that the program makes them: a partial order from what they read and write, and the place from the top (the y of
 * the node) for what that does not decide. An update is written before it is read. The updates of an object are in the order that
 * their writers have by the data flow (where it does not say, by the place of the writers); what reads an update comes before the write
 * of each later one, and the writes are in that order. The calls of a loop stay together: the order is made for the things of each
 * level (calls and loops), a loop having the dependences of all its calls.
 * `cycles`: the calls that the order could only be decided for by their place (a cycle of dependences); `conflicts`: the loops that
 * are in such a cycle (a call that must come between their calls); `versions`: Map(object key -> its node ids: the allocation, then
 * the updates in order); `unordered`: [{ a, b }] the consecutive updates whose order the data flow does not give.
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
    for (const s of list) for (const source of calls[s.id].scalars ?? []) before.push([source.skeleton.id, s.id]);                     // (a result is printed after it is made)      // (a random stream is continued in the order of the chain)

    // the updates of each object, in the order that the data flow gives them: an update is before another when its writer is before the writer of
    // the other (a path of the pairs above); where it does not say, the place from the top of the writers does, and `unordered` has the pair
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
        // what reads an update comes before the write of the next one, and the writes are in order
        const all = versions.get(key);
        for (let i = 0; i < all.length; i++) for (let j = i + 1; j < all.length; j++) {
            const later = writerOf(all[j]);
            if (!later) continue;
            for (const r of readers.get(all[i]) ?? []) if (r !== later) before.push([r, later]);
            const earliest = writerOf(all[i]);
            if (earliest && earliest !== later) before.push([earliest, later]);
        }
    }

    // at each level, the things (calls and loops) of it and what comes before what
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
            if (!next) {                                                // a cycle: the place decides
                next = rest[0];
                const preds = i => [...edges].filter(([from, to]) => !done.has(from) && to.has(i)).map(([from]) => from);
                const walk = [];                                        // follow what is waited for until it comes round: that is the cycle
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

/** The containers that a call reads and writes (an external reads and writes what it is connected to). */
export const readsOf = call => (call?.reads ?? [...(call?.elwise ?? []), ...(call?.proxies ?? []), call?.vertical, call?.horizontal]).filter(Boolean);
export const writesOf = call => call?.writes ?? (call?.out ? [call.out] : []);

const LETTERS = 'abcdefghijklmnopqrstuvwxyz';

/** The user functions of a call: name, parameters (name, type, role), the type it returns, and the stub that it has until the user writes one. */
function userFunctions(call, names, used) {
    const s = call.skeleton, T = call.elementType;
    const uniforms = s.uniforms.filter(u => u.name).map(u => ({ name: u.name, type: u.type || 'float', role: 'uniform', value: u.value }));
    const proxyParams = call.proxies.map((c, i) => ({ name: identifier(`${{ Vector: 'vec', Matrix: 'table', Tensor3: 'ten3', Tensor4: 'ten4' }[c.kind] ?? 'vec'}_${i}`, new Set(), 'p'), type: c.type,
        role: 'proxy', container: c }));
    const letters = n => Array.from({ length: n }, (_, i) => LETTERS[i % 26]);
    // `stub`: the body that a function has until the user writes one, in C++ and in Python; `bodies` has the user's, for each language
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
            return [fn('_f', [{ name: 'a', type: t, role: 'elwise' }, { name: 'b', type: t, role: 'elwise' }], t, { cpp: 'return a + b;', py: 'return a + b' })];
        }
        case 'MapOverlap': {
            const dims = dimsOf(s), zeros = Array(dims).fill(0).join(', ');
            const params = [...(call.random ? [{ name: 'random', type: 'Random', role: 'random' }] : []),
                { name: 'r', type: `Region${dims}D`, role: 'region', elementType: call.elwise[0]?.type ?? 'float' }, ...proxyParams, ...uniforms];
            return [fn('_f', params, T, { cpp: `return r(${zeros});`, py: `return r[${zeros}]` })];
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

/** Where a call runs: 'host' or 'device'. A call without a backend of its own is as the program is started: `unset` says what that is ('host'), or, left out, nothing known (undefined). */
export const placeOf = (call, unset) => call.host ? 'host' : PLACE[call.skeleton.backend] ?? unset;

/**
 * Where the current value of each container is, and what has to move for a call: a call on a device (OpenCL) that reads what was last written
 * on the host (CPU, OpenMP, or the initial value), or the other way. The program says nothing of it (SkePU does it); the graph shows it as
 * transfers, and the trace has them as events. `unset`: the place of the calls that have no backend of their own (see placeOf): when
 * there is none, they are not followed, and what they write is not known to be anywhere.
 */
export class Residency {
    constructor(program, unset) {
        this.unset = unset;
        this.valid = new Map(program.containers.map(c => [objectKey(c), new Set(['host'])]));        // by object: the updates are one piece of data
        this.copies = new Map();         // object -> Map(place -> the transfer that made the copy there: { container, direction, consumers })
        this.records = [];               // every transfer that was made
    }
    /**
     * What a call needs moved first: [{ container, direction: 'host-to-device' | 'device-to-host' }]; then what it writes is where it ran.
     * A transfer is where the data is anchored on that side: every call that reads the copy that it made (in the same place, until the data is
     * written again) is a consumer of it (`records`: { container, direction, consumers: the ids of the calls }).
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
 * The transfers that a program has between calls on different explicit backends. Calls that have the backend as the program is started
 * are not followed. A loop is followed twice, to see what moves from one iteration to the next.
 * Returns [{ container, consumer, direction, consumers }] (ids), once for each transfer: `container` is the data node that is moved, `consumer`
 * the first call that needed it moved, and `consumers` all the calls that read the copy (the transfer is what they read it from).
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
 * Copies nodes of a design (containers, skeletons, regions; a region brings what is in it), into the design, and returns the ids of the
 * copies. What is between copied nodes is between the copies. A copied call reads what the original reads (the data that is not copied is
 * shared, a random stream too) and writes data of its own: a new container in the format of the original's output, so that two calls do
 * not write the same data. A copied container that was written by a call that is not copied has no call to write it: it keeps the
 * format that it has. The copies are a little below and to the right of the originals, and are in the regions of the originals.
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
    // the copies of the nodes of an object are an object of their own when there are several of them (one name, one key), and a copy of one node
    // is a new object (a copy is not an update of what it is a copy of)
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
            const producedOutside = n.auto && !written.has(n.id);      // its call is not copied: the format that it has is its own now
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

/** A heat-like example: a vector of values, a stencil iterated, and the sum of the result. */
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

/** A design with every skeleton, the ways to give a Map its arguments, and a loop: what the round trip test builds and runs. */
export function allSkeletonsDesign() {
    const d = emptyDesign();
    d.name = 'all_skeletons';
    const AUTO = new Set(['c4', 'c5', 'c7', 'c8', 'c11', 'c14', 'c15']);       // the ones that the calls that write them give their format
    const vec = (id, label, n, init = '1.0f') => d.containers.push({ ...defaultContainer(id, label), size: [n], init, auto: AUTO.has(id) });
    const mat = (id, label, r, c, init = '0.0f') => d.containers.push({ ...defaultContainer(id, label, 'Matrix'), size: [r, c], init, auto: AUTO.has(id) });
    vec('c1', 'ramp', 1024, '0.0f'); vec('c2', 'noise', 1024, '0.0f'); vec('c3', 'table', 16, '2.0f'); vec('c4', 'values', 1024);
    vec('c5', 'smoothed', 1024, '0.0f'); mat('c6', 'grid', 32, 32); mat('c7', 'blurred', 32, 32); vec('c8', 'prefix', 1024, '0.0f');
    vec('c9', 'rows', 32); vec('c10', 'columns', 48); mat('c11', 'pairs', 32, 48);
    d.containers.push(defaultPrng('c12', 'noise_stream'));
    vec('c13', 'summary', 8);
    vec('c14', 'values', 1024);                                    // the update of `values` that the loop writes (c4 is the one before it): the same object
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

/**
 * Gives the skeletons the places from the top that they have, in an order in which the calls of every loop are consecutive:
 * the order by place, with the skeletons of a loop together where the first of them is. (An arrangement of the graph by the data
 * flow puts a call between the calls of a loop when it is on their level.) Changes `design.positions`.
 */
export function keepLoopsTogether(design) {
    const y = s => design.positions[s.id]?.y ?? 0;
    const chain = s => { const c = []; for (let p = parentLoop(design, s.id); p; p = parentLoop(design, p.id)) c.unshift(p.id); return c; };
    const chains = new Map(design.skeletons.map(s => [s.id, chain(s)]));
    const places = design.skeletons.map(y).sort((a, b) => a - b);
    const order = [];
    const emit = (list, depth) => {
        const done = new Set();
        for (const s of list) {
            const key = chains.get(s.id)[depth] ?? null;
            if (key === null) order.push(s);
            else if (!done.has(key)) { done.add(key); emit(list.filter(x => chains.get(x.id)[depth] === key), depth + 1); }
        }
    };
    emit([...design.skeletons].sort((a, b) => y(a) - y(b)), 0);
    order.forEach((s, i) => { design.positions[s.id] = { ...(design.positions[s.id] ?? { x: 0 }), y: places[i] }; });
    return design;
}
