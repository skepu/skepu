// random.js — a random design: a graph of containers, skeletons, externals, random streams and regions that makes a valid program
// (model.js lower() has no errors). The settings bias the choices, they do not fix the result:
//   size   small | medium | large   about 12, 40 or 100 nodes
//   arity  small | large            how many inputs, proxies and uniforms the calls have, and how much the externals read and write
//   shape  deep | balanced | wide   deep: each call reads what the last calls wrote (long chains); wide: many sources and calls on
//                                   data that is read again and again (many short chains side by side)
//   seed   text; the same settings and seed make the same design (empty: a seed is made, and returned)
// A call often writes data that exists already, as SkePU programs do: a new update of an object that has the name (in place, for a Map; or
// data that the call depends on), and the Maps that draw random numbers are sometimes a chain on one stream.
// Pure: no DOM. The calls are made in order (each reads only what was made before), so the design has no cycles and the order of the
// calls from the top is the order in which they were made; the regions are intervals of that order.
'use strict';
import { defaultSkeleton, defaultContainer, defaultPrng, emptyDesign, objectKey } from './model.js';

export const SIZES = ['small', 'medium', 'large'];
export const ARITIES = ['small', 'large'];
export const SHAPES = ['deep', 'balanced', 'wide'];

// ── A seeded random number generator ─────────────────────────────────────────

function seeded(text) {
    let h = 1779033703 ^ text.length;                          // (a hash of the text: xmur3)
    for (let i = 0; i < text.length; i++) { h = Math.imul(h ^ text.charCodeAt(i), 3432918353); h = (h << 13) | (h >>> 19); }
    h = Math.imul(h ^ (h >>> 16), 2246822507); h = Math.imul(h ^ (h >>> 13), 3266489909);
    let a = (h ^ (h >>> 16)) >>> 0;
    return () => {                                              // (mulberry32)
        a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const newSeed = () => Math.random().toString(36).slice(2, 8);

// ── The biases ───────────────────────────────────────────────────────────────

const CALLS = { small: [4, 6], medium: [14, 22], large: [40, 52] };           // calls (about as many containers: 12, 40, 100 nodes)
/** The weights of the inputs of an elementwise call: 1 input, 2, 3, ... */
const INPUTS = { small: [0.72, 0.24, 0.04], large: [0.05, 0.3, 0.3, 0.2, 0.15] };
const SHAPE = {
    //        how much the newest data is preferred to the old (the weight of the data of age n is e^(-recency n)), how often a call starts
    //        from new data instead of reading what there is, and how much data that is read already is read again
    //        and how often a call writes a new update of data that exists, instead of new data
    deep:     { recency: 1.6, fresh: 0.06, reuse: 0, again: 0.3 },
    balanced: { recency: 0.35, fresh: 0.2, reuse: 0.4, again: 0.3 },
    wide:     { recency: 0, fresh: 0.45, reuse: 1.2, again: 0.2 },
};
const VECTORS = [512, 1024, 2048, 4096], SMALL_VECTORS = [16, 32, 48, 64], MATRICES = [[32, 32], [64, 64], [32, 48]];

/**
 * @param {{ size?: string, arity?: string, shape?: string, seed?: string }} settings
 * @returns {{ design, seed }}  the design, and the seed (the one given, or the one that was made)
 */
export function randomDesign({ size = 'medium', arity = 'small', shape = 'balanced', seed = '' } = {}) {
    seed = String(seed ?? '').trim() || newSeed();
    const rnd = seeded(`${size}|${arity}|${shape}|${seed}`);
    const chance = p => rnd() < p;
    const int = (lo, hi) => lo + Math.floor(rnd() * (hi - lo + 1));
    const pick = list => list[Math.floor(rnd() * list.length)];
    const weighted = (items, weight) => {
        const w = items.map(weight), total = w.reduce((a, b) => a + b, 0);
        if (!(total > 0)) return pick(items);
        let r = rnd() * total;
        for (let i = 0; i < items.length; i++) { r -= w[i]; if (r < 0) return items[i]; }
        return items[items.length - 1];
    };
    const bias = SHAPE[shape] ?? SHAPE.balanced;
    const large = arity === 'large';

    const d = emptyDesign();
    d.name = `random_${seed}`.replace(/[^A-Za-z0-9_]/g, '_');
    const n = { c: 0, s: 0, l: 0 };
    const id = prefix => `${prefix}${++n[prefix]}`;
    const labels = { data: 0, random: 0 };
    const counts = {};
    const step = 90, lane = 130;
    let row = 0;

    // the formats: a few sizes of vectors, small vectors (for the pairs) and a matrix or two
    const vectorSizes = [...VECTORS].sort(() => rnd() - 0.5).slice(0, 2);
    const matrixShapes = [...MATRICES].sort(() => rnd() - 0.5).slice(0, 2);
    const formatKey = c => `${c.kind}:${c.size.join('x')}`;

    // the data that there is: { container, age (calls since it was made), readers }
    const pool = [];
    let calls = 0;
    const place = (node, x) => { d.positions[node.id] = { x: Math.round(x), y: Math.round(row * step + int(-10, 10)) }; };

    const container = (format, { fresh = false, auto = false } = {}) => {
        const c = { ...defaultContainer(id('c'), `data${++labels.data}`, format.kind), ...format, size: [...format.size], init: fresh ? '1.0f' : '0.0f', ...(auto ? { auto: true } : {}) };
        d.containers.push(c);
        pool.push({ container: c, born: calls, readers: 0 });
        return c;
    };
    const randomFormat = () => chance(0.7) ? { kind: 'Vector', type: 'float', size: [pick(vectorSizes)] } : { kind: 'Matrix', type: 'float', size: [...pick(matrixShapes)] };
    /** New data that is read but not written by a call: an allocation. */
    const source = (format = randomFormat()) => {
        const c = container(format, { fresh: true });
        place(c, int(0, 6) * lane);
        return c;
    };

    /** An input: data that there is (the newest preferred for a deep graph, the data that is read often for a wide one), or new data. */
    const input = (matching, avoid = new Set()) => {
        const candidates = pool.filter(p => !avoid.has(p.container.id) && (!matching || matching(p.container)));
        if (!candidates.length || chance(bias.fresh)) return null;
        return weighted(candidates, p => Math.exp(-bias.recency * (calls - p.born)) * (1 + bias.reuse * Math.min(p.readers, 4)));
    };
    const read = p => { p.readers++; return p; };

    const arityOf = () => 1 + weightedIndex(INPUTS[arity] ?? INPUTS.small);
    const weightedIndex = w => { let r = rnd() * w.reduce((a, b) => a + b, 0); for (let i = 0; i < w.length; i++) { r -= w[i]; if (r < 0) return i; } return w.length - 1; };

    let edges = 0;
    const link = (from, to, role) => d.edges.push({ id: `e${++edges}`, from, to, role });

    const skeleton = (pattern, patch = {}) => {
        const s = { ...defaultSkeleton(pattern, id('s'), `${pattern.toLowerCase()}${(counts[pattern] = (counts[pattern] ?? 0) + 1)}`), ...patch };
        const backend = chance(0.4) ? pick(['cpu', 'openmp', 'opencl']) : '';
        if (backend && pattern !== 'External' && pattern !== 'IO') s.backend = backend;          // (a host block has no backend)
        d.skeletons.push(s);
        return s;
    };
    // an I/O block ("I/O", as in SkeVU): it prints data, or the result of a reduction (the block is connected from the call, and is in its region)
    const ioLabel = () => counts.IO ? `I/O ${counts.IO + 1}` : 'I/O';
    const followers = [];
    const printResult = (s, p) => {
        if (!chance(p)) return;
        const io = skeleton('IO', { label: ioLabel() });
        link(s.id, io.id, 'scalar');
        place(io, (d.positions[s.id]?.x ?? 0) + 90);
        followers.push([io, s]);
    };

    // the first input of a call, and the others of the same format
    const inputs = (k, format) => {
        const chosen = [];
        const avoid = new Set();
        let first = input(format ? c => formatKey(c) === format : null);
        const base = first ? first.container : source(format ? { kind: format.split(':')[0], type: 'float', size: format.split(':')[1].split('x').map(Number) } : randomFormat());
        if (first) read(first);
        chosen.push(base); avoid.add(base.id);
        for (let i = 1; i < k; i++) {
            const more = input(c => formatKey(c) === formatKey(base), avoid);
            const c = more ? more.container : (chance(0.5) ? source({ kind: base.kind, type: 'float', size: [...base.size] }) : null);
            if (!c) break;
            if (more) read(more);
            chosen.push(c); avoid.add(c.id);
        }
        return chosen;
    };
    /** Proxies and uniforms of a call, as the arity wants them. */
    const extras = (s, taken) => {
        const proxies = large ? int(0, 2) : (chance(0.12) ? 1 : 0);
        const avoid = new Set(taken.map(c => c.id));
        for (let i = 0; i < proxies; i++) {
            const p = input(null, avoid);
            if (!p) break;
            read(p); avoid.add(p.container.id);
            link(p.container.id, s.id, 'proxy');
        }
        const uniforms = large ? int(0, 2) : (chance(0.25) ? 1 : 0);
        for (let i = 0; i < uniforms; i++) s.uniforms.push({ name: `u${i + 1}`, type: 'float', value: pick(['0.5f', '1.0f', '2.0f', '0.25f']) });
    };
    // where the data was written: the call before each update (for what a call depends on), and the end of the chain of each random stream
    const writerOf = new Map(), tailOf = new Map();
    const dependsOn = (call, target) => {
        const seen = new Set(), todo = [call];
        while (todo.length) {
            const x = todo.pop();
            for (const e of d.edges) if (e.to === x && e.role !== 'out') {
                const w = writerOf.get(e.from) ?? (e.from.startsWith('s') ? e.from : null);       // (an edge from a call is a random edge)
                if (w === target) return true;
                if (w && !seen.has(w)) { seen.add(w); todo.push(w); }
            }
        }
        return false;
    };
    /**
     * A new update of data that exists, in the format that the call writes: the data that the call depends on the writer of (so that the data
     * flow orders the updates), or that nothing wrote; and only the newest update of data can be read after that. A Map may write what it
     * reads element by element (in place), no other call may write what it reads.
     */
    const update = (s, format) => {
        const key = `${format.kind}:${format.size.join('x')}:${format.type}`;
        const usable = pool.filter(p => p.container.kind !== 'PRNG' && `${p.container.kind}:${p.container.size.join('x')}:${p.container.type}` === key && (() => {
            const roles = d.edges.filter(e => e.to === s.id && e.from === p.container.id).map(e => e.role);
            if (roles.length && !(s.pattern === 'Map' && roles.every(r => r === 'elwise'))) return false;
            return !writerOf.has(p.container.id) || dependsOn(s.id, writerOf.get(p.container.id));
        })());
        if (!usable.length) return null;
        const p = pick(usable);
        const c = { ...defaultContainer(id('c'), p.container.label, format.kind), ...format, size: [...format.size], init: p.container.init, auto: true, object: objectKey(p.container) };
        d.containers.push(c);
        p.container = c; p.born = calls; p.readers = 0;
        return c;
    };
    const output = (s, format, auto) => {
        const c = chance(bias.again) ? update(s, format) : null;
        const made = c ?? container(format, { auto });
        if (c && !auto) made.auto = false;
        place(made, (d.positions[s.id]?.x ?? 0) + int(-1, 1) * 60);
        link(s.id, made.id, 'out');
        writerOf.set(made.id, s.id);
        return made;
    };

    /** Random numbers for a call, now and then: a stream that calls share (the call continues its chain: a stream has one edge out, and so has each call). */
    const randomly = (s, p) => {
        if (!chance(p)) return;
        const streams = d.containers.filter(c => c.kind === 'PRNG');
        let stream = streams.length && chance(0.5) ? pick(streams) : null;
        if (!stream) { stream = defaultPrng(id('c'), `random${++labels.random}`); d.containers.push(stream); place(stream, (d.positions[s.id]?.x ?? 0) + 90); }
        link(tailOf.get(stream.id) ?? stream.id, s.id, 'random');
        tailOf.set(stream.id, s.id);
    };

    const ops = {
        Map() {
            const indexed = chance(0.08);
            const s = skeleton('Map');
            if (indexed) {                                                  // no input: the size is the output's
                place(s, int(0, 6) * lane);
                output(s, randomFormat(), false);
                return s;
            }
            const ins = inputs(arityOf());
            ins.forEach(c => link(c.id, s.id, 'elwise'));
            place(s, avgX(ins));
            extras(s, ins);
            randomly(s, 0.08);
            output(s, formatOf(ins[0]), true);
            return s;
        },
        MapReduce() {
            const s = skeleton('MapReduce');
            const ins = inputs(arityOf());
            ins.forEach(c => link(c.id, s.id, 'elwise'));
            place(s, avgX(ins));
            extras(s, ins);
            randomly(s, 0.04);
            printResult(s, 0.5);
            return s;
        },
        Reduce() {
            const s = skeleton('Reduce');
            const ins = inputs(1);
            link(ins[0].id, s.id, 'elwise');
            place(s, avgX(ins));
            printResult(s, 0.5);
            return s;
        },
        Scan() {
            const vector = c => c.kind === 'Vector';
            const p = input(vector);
            const c = p ? read(p).container : source({ kind: 'Vector', type: 'float', size: [pick(vectorSizes)] });
            const s = skeleton('Scan', { params: { mode: pick(['Inclusive', 'Exclusive']), start: chance(0.3) ? '1.0f' : '' } });
            link(c.id, s.id, 'elwise');
            place(s, avgX([c]));
            output(s, formatOf(c), true);
            return s;
        },
        MapOverlap() {
            const p = input(null);
            const c = p ? read(p).container : source();
            const dims = c.kind === 'Matrix' ? 2 : 1;
            const s = skeleton('MapOverlap', { params: { dims, overlap: [int(1, 3), int(1, 3)], edge: weighted(['Duplicate', 'None', 'Cyclic', 'Pad'], w => w === 'Duplicate' ? 3 : 1), pad: '0.0f' } });
            link(c.id, s.id, 'elwise');
            place(s, avgX([c]));
            extras(s, [c]);
            randomly(s, 0.04);
            const format = formatOf(c);
            if (s.params.edge === 'None') format.size = c.size.map((n, i) => n - 2 * s.params.overlap[i]);          // (with no edge mode the output is smaller by the overlap on each side)
            output(s, format, true);
            return s;
        },
        MapPairs() {
            const small = c => c.kind === 'Vector' && c.size[0] <= 128;
            const pickSmall = avoid => { const p = input(small, avoid); return p ? read(p).container : source({ kind: 'Vector', type: 'float', size: [pick(SMALL_VECTORS)] }); };
            const v = pickSmall(new Set()), h = pickSmall(new Set([v.id]));
            const s = skeleton('MapPairs');
            link(v.id, s.id, 'vertical');
            link(h.id, s.id, 'horizontal');
            place(s, avgX([v, h]));
            extras(s, [v, h]);
            randomly(s, 0.04);
            output(s, { kind: 'Matrix', type: 'float', size: [v.size[0], h.size[0]] }, true);
            return s;
        },
        External() {
            const k = large ? int(1, 4) : (chance(0.7) ? 1 : 0);
            const reads = [];
            const avoid = new Set();
            for (let i = 0; i < k; i++) {
                const p = input(null, avoid);
                if (!p) break;
                read(p); reads.push(p.container); avoid.add(p.container.id);
            }
            const s = skeleton('External', { backend: '' });
            reads.forEach(c => link(c.id, s.id, 'read'));
            place(s, avgX(reads));
            const writes = large ? int(0, 3) : (chance(0.6) ? 1 : 0);
            for (let i = 0; i < writes; i++) output(s, reads[0] ? formatOf(reads[0]) : { kind: 'Vector', type: 'float', size: [pick(vectorSizes)] }, false);
            return s;
        },
        IO() {
            const reads = [], avoid = new Set();
            for (let i = chance(0.4) ? 2 : 1; i > 0; i--) {
                const p = input(null, avoid);
                if (!p) break;
                read(p); reads.push(p.container); avoid.add(p.container.id);
            }
            if (!reads.length) return ops.Map();                          // (nothing to print yet)
            const s = skeleton('IO', { label: ioLabel() });
            reads.forEach(c => link(c.id, s.id, 'read'));
            place(s, avgX(reads));
            return s;
        },
        MapPairsReduce() {
            const small = c => c.kind === 'Vector' && c.size[0] <= 128;
            const pickSmall = avoid => { const p = input(small, avoid); return p ? read(p).container : source({ kind: 'Vector', type: 'float', size: [pick(SMALL_VECTORS)] }); };
            const v = pickSmall(new Set()), h = pickSmall(new Set([v.id]));
            const mode = chance(0.3) ? 'ColWise' : 'RowWise';
            const s = skeleton('MapPairsReduce', { params: { mode, start: '' } });
            link(v.id, s.id, 'vertical');
            link(h.id, s.id, 'horizontal');
            place(s, avgX([v, h]));
            extras(s, [v, h]);
            randomly(s, 0.04);
            output(s, { kind: 'Vector', type: 'float', size: [(mode === 'ColWise' ? h : v).size[0]] }, true);
            return s;
        },
    };
    const formatOf = c => ({ kind: c.kind, type: c.type, size: [...c.size] });
    const avgX = list => list.length ? list.reduce((a, c) => a + (d.positions[c.id]?.x ?? 0), 0) / list.length : int(0, 6) * lane;

    // how often each pattern is chosen: a reduction ends a chain, so a deep graph has few
    const choose = () => weighted(Object.keys(ops), op => ({
        Map: 38, MapOverlap: 12, Scan: 8, MapPairs: 6, MapPairsReduce: 4, External: 10, IO: 5,
        MapReduce: shape === 'deep' ? 3 : 9, Reduce: shape === 'deep' ? 2 : 7,
    })[op]);

    const [lo, hi] = CALLS[size] ?? CALLS.medium;
    const total = int(lo, hi);
    const made = [];
    while (calls < total) {
        row++;
        made.push(ops[choose()]());
        calls++;
    }

    // regions: intervals of the calls in order, nested or apart; a loop repeats its calls, a scope groups them
    const regions = size === 'small' ? int(0, 1) : size === 'medium' ? int(1, 3) : int(3, 7);
    const accepted = [];
    for (let attempt = 0; attempt < regions * 12 && accepted.length < regions; attempt++) {
        const len = int(2, Math.max(2, Math.min(made.length - 1, Math.round(made.length * (shape === 'deep' ? 0.5 : 0.25)))));
        const from = int(0, made.length - len), to = from + len - 1;
        const fits = accepted.every(r => (to < r.from || from > r.to) || (from >= r.from && to <= r.to && !(from === r.from && to === r.to)) || (r.from >= from && r.to <= to && !(from === r.from && to === r.to)));
        if (fits && len >= 2 && made.length >= 3) accepted.push({ from, to });
    }
    accepted.sort((a, b) => (b.to - b.from) - (a.to - a.from));          // the outer ones first
    const loops = [];
    for (const r of accepted) {
        const scope = chance(0.5);
        loops.push({ id: id('l'), label: scope ? `Region ${loops.filter(l => l.scope).length + 1}` : `Loop ${loops.filter(l => !l.scope).length + 1}`,
                     count: scope ? 1 : int(2, 6), scope, parent: null, from: r.from, to: r.to });
    }
    for (const l of loops) {                                               // the smallest region that is around it
        const inside = loops.filter(o => o !== l && o.from <= l.from && o.to >= l.to);
        l.parent = inside.length ? inside.sort((a, b) => (a.to - a.from) - (b.to - b.from))[0].id : null;
    }
    made.forEach((s, i) => {
        const inside = loops.filter(l => l.from <= i && l.to >= i).sort((a, b) => (a.to - a.from) - (b.to - b.from));
        if (inside.length) s.parent = inside[0].id;
    });
    for (const [io, s] of followers) if (s.parent) io.parent = s.parent;                // (the block that prints a result is where the call is)
    for (const l of loops) { delete l.from; delete l.to; d.loops.push(l); }
    return { design: d, seed };
}
