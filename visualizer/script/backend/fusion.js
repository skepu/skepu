// fusion.js — which skeleton calls could be fused, and why others could not.
//
// Works on the events of a trace (trace.js), by the containers (object ids) that the events read
// and write, not on the graph of the views, so that the rules can be stated, tested and explained.
//
// SERIAL fusion (producer -> consumer). A call A that writes a container C and the call B that
// reads it element by element are fused into one call, which does not write and read C in memory,
// when:
//   * B is the only reader of that version of C: no other call, no host access, no transfer, no
//     snapshot (C is overwritten or freed afterwards, or is live-out: then it is still written);
//   * B reads C element by element, not through a proxy (random access);
//   * the patterns can be fused (rules.serialProducers, rules.serialConsumers) and the calls
//     run on the same backend, over the same elements, and do not draw from the same PRNG stream;
//   * the fused call can be placed at the position of the last call of the chain (sinking the
//     earlier ones) or of the first (hoisting the later ones), that is, no other event between
//     them conflicts with a call that has to move past it (read after write, write after read,
//     write after write on the same container).
// Chains and trees are made from the pairs that satisfy this, each checked as a whole.
//
// PARALLEL fusion (siblings). Independent calls of a fusable pattern (rules.parallel) over the
// same elements on the same backend, none of which reads or writes what another writes, are fused
// into one call (one launch, the shared inputs read once), under the same placement rule.
//
// A Map can also be fused into the skeletons that read a container with a neighbourhood or a pair
// of vectors (rules.serialConsumersRecompute: MapOverlap, MapPool, MapPairs, MapPairsReduce). There
// the fused call computes the Map's function at an element once for each output that reads it, so
// it trades memory traffic for computation (`recompute`, with the factor when the trace has the sizes):
// a MapPool whose windows do not overlap and a MapOverlap of radius 0 would not recompute anything.
// Nothing can be fused into a Scan (there is no MapScan skeleton: the Map's function could not take
// the proxies and uniform arguments of the Scan), nor a Reduce into anything (its result is a scalar).
//
// The pattern tables are assumptions about what SkePU can fuse (examples/fusion.cpp has the >>, || and ^ that exist); change them with `rules`.
'use strict';

import { duration, eventId } from './analysis.js';

export const DEFAULT_RULES = {
    serialProducers: ['Map', 'MapPairs'],                   // examples/fusion.cpp: Map >> Map, Map >> Reduce, MapPairs >> Map, MapPairs >> Reduce
    serialConsumers: ['Map', 'Reduce', 'MapReduce'],        // a Map into a Reduce is a MapReduce
    serialConsumersRecompute: ['MapOverlap', 'MapPool', 'MapPairs', 'MapPairsReduce'],
    parallel:        ['Map', 'Reduce', 'MapPairs'],          // examples/fusion.cpp: Map || Map, Reduce || Reduce, MapPairs || MapPairs
};

/** 'MapOverlap 4D' -> 'MapOverlap', 'Reduce 1D-Full' -> 'Reduce' */
const baseOf = pattern => (pattern ?? '').split(' ')[0];

/**
 * What fusing the Map `A` into the consumer `B` costs in recomputation, or null if nothing:
 *   pairs    A is a vector that B reads at every row (or column): each element is computed once per
 *            element of the other vector; the factor is known from the sizes
 *   overlap  each element is computed once per output whose neighbourhood contains it (the trace
 *            does not have the radius)
 *   pool     as overlap if the windows overlap, none if they do not (the trace does not have the windows)
 */
function recomputeOf(A, B) {
    const prm = B.parameters ?? {};
    // each element is read by as many outputs as there are windows (of the pool, or of the overlap) over it
    const readers = (size, stride) => size.reduce((n, s, d) => n * Math.ceil(s / Math.max(1, stride?.[d] ?? 1)), 1);
    switch (baseOf(B.pattern)) {
        case 'MapPairs':
        case 'MapPairsReduce': {
            const [h, v] = B.elements ?? [];
            const n = (A.elements ?? [])[0];
            const factor = n === h && n !== v ? v : n === v && n !== h ? h : n;
            return { kind: 'pairs', factor, note: `each element is computed ${factor} times (once per element of the other vector)` };
        }
        case 'MapOverlap': {
            if (!prm.overlap) return { kind: 'overlap', factor: null, note: 'each element is computed once for each output whose neighbourhood contains it' };
            const factor = readers(prm.overlap.map(o => 2 * o + 1), prm.stride);
            return { kind: 'overlap', factor, note: `each element is computed ${factor} times (once for each output whose neighbourhood of ${prm.overlap.map(o => 2 * o + 1).join(' x ')} contains it)` };
        }
        case 'MapPool': {
            if (!prm.pool_size) return { kind: 'pool', factor: null, note: 'none if the pool windows do not overlap, else each element is computed once per window that contains it' };
            const factor = readers(prm.pool_size, prm.stride);
            return factor === 1 ? null : { kind: 'pool', factor, note: `each element is computed ${factor} times (the pool windows ${prm.pool_size.join(' x ')} with stride ${prm.stride.join(' x ')} overlap)` };
        }
    }
    return null;
}

/** Whether the element count of the Map A fits the consumer B (for the consumers that are not element-wise, the count of one of its inputs). */
function sizeFits(A, B) {
    switch (baseOf(B.pattern)) {
        case 'MapPairs':
        case 'MapPairsReduce': {
            const n = (A.elements ?? [])[0], [h, v] = B.elements ?? [];
            return A.elements?.length === 1 && (n === h || n === v);
        }
        case 'MapOverlap':
        case 'MapPool': return true;       // the output has other extents than the input (valid convolution, pooling)
    }
    // A MapPairs is traced as (Hsize, Vsize), and a Map over a matrix as its element count.
    if (baseOf(A.pattern) === 'MapPairs' && A.elements?.length === 2) {
        if (B.elements?.length === 2) return sameElements(A, B) || (A.elements[0] === B.elements[1] && A.elements[1] === B.elements[0]);
        if (B.elements?.length === 1) return A.elements[0] * A.elements[1] === B.elements[0];     // (a Map over a matrix)
    }
    return sameElements(A, B);
}
// ── What an event reads and writes ───────────────────────────────────────────

function accessOf(e) {
    const reads = new Set(), writes = new Set(), proxy = new Set(), elwise = new Set();
    switch (e.type) {
        case 'skeleton_call':
        case 'external':
            for (const id of e.elwise_inputs ?? []) { reads.add(id); elwise.add(id); }
            for (const id of e.proxy_inputs  ?? []) { reads.add(id); proxy.add(id); }
            for (const id of e.scalar_inputs ?? []) reads.add(id);
            for (const id of e.outputs       ?? []) writes.add(id);
            if (e.prng != null) { reads.add(e.prng); writes.add(e.prng); }   // the state of the stream
            break;
        case 'copy':           reads.add(e.source); writes.add(e.object_id); break;
        case 'communication':  for (const id of e.object_ids ?? []) { reads.add(id); writes.add(id); } break;   // (a move: nothing can be fused across it)
        case 'transfer':
        case 'snapshot':       reads.add(e.object_id); break;
        case 'element_access': reads.add(e.source);    break;
        case 'deallocation':   writes.add(e.object_id); break;
    }
    return { reads, writes, proxy, elwise };
}

class Index {
    constructor(events) {
        this.events = events;
        this.access = new Map();            // idx -> access
        this.touch  = new Map();            // container -> [{ idx, read, write }] in trace order
        this.label  = new Map();            // container -> label
        for (const e of events) {
            const a = accessOf(e);
            this.access.set(e.idx, a);
            if (e.type === 'allocation' && e.label) this.label.set(e.object_id, e.label);
            const ids = new Set([...a.reads, ...a.writes]);
            for (const id of ids) {
                let list = this.touch.get(id);
                if (!list) this.touch.set(id, list = []);
                list.push({ idx: e.idx, read: a.reads.has(id), write: a.writes.has(id) });
            }
        }
    }
    labelOf(id) { return this.label.get(id) ?? `#${id}`; }

    /** The accesses of a container with an index in (lo, hi). */
    *between(id, lo, hi) {
        const list = this.touch.get(id) ?? [];
        let a = 0, z = list.length;
        while (a < z) { const mid = (a + z) >>> 1; if (list[mid].idx <= lo) a = mid + 1; else z = mid; }
        for (let i = a; i < list.length && list[i].idx < hi; i++) yield list[i];
    }

    /** The accesses of a container after an event, up to and including the first write. */
    *after(id, idx) {
        for (const t of this.between(id, idx, Infinity)) { yield t; if (t.write) return; }
    }
}

// ── Legality of a group of calls ─────────────────────────────────────────────

/**
 * Whether the calls (a set of trace indices) can be one call: 'sink' (at the last one's place),
 * 'hoist' (at the first one's) or null, with the reason. `intermediates` are the containers
 * that the group produces and consumes inside, which are not written.
 */
function placement(ix, members, intermediates) {
    const idxs = [...members].sort((a, b) => a - b);
    const first = idxs[0], last = idxs[idxs.length - 1];

    // Inside the group the calls keep their order, element by element; a call that reads a container
    // through a proxy (any element) would see the writes of another call at other elements.
    const proxyReaders = new Map(), writers = new Map();
    for (const m of idxs) {
        const a = ix.access.get(m);
        for (const x of a.proxy)  (proxyReaders.get(x) ?? proxyReaders.set(x, new Set()).get(x)).add(m);
        for (const x of a.writes) (writers.get(x)      ?? writers.set(x, new Set()).get(x)).add(m);
    }
    for (const [x, ps] of proxyReaders) {
        const ws = writers.get(x);
        if (ws && [...ws].some(w => !ps.has(w) || ps.size > 1 || ws.size > 1)) return { how: null, reason: 'proxy-access-to-written-container' };
    }

    let sink = true, hoist = true;
    for (const m of idxs) {
        const a = ix.access.get(m);
        for (const x of new Set([...a.reads, ...a.writes])) {
            if (intermediates.has(x)) continue;
            const writes = a.writes.has(x), reads = a.reads.has(x);
            for (const t of ix.between(x, first, last)) {
                if (members.has(t.idx)) continue;
                const conflict = writes || (reads && t.write);
                if (!conflict) continue;
                if (t.idx > m) sink = false; else hoist = false;
            }
        }
    }
    return sink ? { how: 'sink' } : hoist ? { how: 'hoist' } : { how: null, reason: 'hazard' };
}

const sameElements = (a, b) => JSON.stringify(a.elements ?? null) === JSON.stringify(b.elements ?? null);

// ── Serial fusion ────────────────────────────────────────────────────────────

function serialPairs(ix, rules, blocked) {
    const pairs = [];
    const block = (reason, producer, consumer, container, extra = {}) =>
        blocked.push({ kind: 'serial', reason, producer: producer.idx, consumer: consumer?.idx ?? null, container, ...extra });

    for (const A of ix.events) {
        if (A.type !== 'skeleton_call') continue;
        for (const C of A.outputs ?? []) {
            // the readers of this version of C, and how it ends
            const readers = []; let ended = null;
            for (const t of ix.after(C, A.idx)) {
                if (t.read) readers.push(ix.events[t.idx]);
                if (t.write) ended = ix.events[t.idx];
            }
            if (readers.length === 0) continue;               // not read: not an intermediate

            const other = readers.filter(r => r.type !== 'skeleton_call');
            if (other.length) { block('host-access', A, readers[0], C, { with: other.map(r => r.type) }); continue; }
            if (readers.length > 1) { block('multiple-readers', A, readers[0], C, { readers: readers.length }); continue; }
            const B = readers[0], bAcc = ix.access.get(B.idx);
            if (['Reduce', 'MapReduce'].includes(baseOf(A.pattern))) { block('scalar-output', A, B, C); continue; }
            if (!bAcc.elwise.has(C) || bAcc.proxy.has(C)) { block('proxy-access', A, B, C); continue; }

            const reasons = [];
            // a Map or MapReduce with a stride reads and writes every other, or every third, element: not one to one
            if ([A, B].some(c => (c.parameters?.strides ?? []).some(s => s !== 1))) { block('strided', A, B, C); continue; }
            const free = rules.serialConsumers.includes(baseOf(B.pattern));
            const costly = rules.serialConsumersRecompute.includes(baseOf(B.pattern));
            if (!rules.serialProducers.includes(baseOf(A.pattern)) || !(free || costly)) reasons.push('pattern');
            else if (!free && !rules.allowRecompute) reasons.push('recompute');
            if ((A.backend ?? null) !== (B.backend ?? null)) reasons.push('backend-mismatch');
            if (!sizeFits(A, B)) reasons.push('size-mismatch');
            if (A.prng != null && A.prng === B.prng) reasons.push('shared-prng');
            if (reasons.length) { block(reasons[0], A, B, C, { all: reasons, patterns: [A.pattern, B.pattern] }); continue; }

            pairs.push({ A, B, C, liveOut: ended === null, overwrittenBy: ended && ended.idx === B.idx, recompute: free ? null : recomputeOf(A, B) });
        }
    }
    return pairs;
}

function serialGroups(ix, pairs, blocked) {
    const parent = new Map();                                  // union-find over call indices
    const find = x => { while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); } return x; };
    const members = new Map(), inter = new Map();              // root -> Set of calls / Map container -> pair
    const add = x => { if (!parent.has(x)) { parent.set(x, x); members.set(x, new Set([x])); inter.set(x, new Map()); } };

    // A pair is added to a group if the whole group can still be one call. A pair that cannot be
    // added may be able to later, when other calls have joined (they change what is in the way).
    let todo = pairs.sort((p, q) => p.B.idx - q.B.idx || p.A.idx - q.A.idx);
    for (let progress = true; progress && todo.length;) {
        progress = false;
        const failed = [];
        for (const pr of todo) {
            add(pr.A.idx); add(pr.B.idx);
            const ra = find(pr.A.idx), rb = find(pr.B.idx);
            const set = new Set([...members.get(ra), ...members.get(rb)]);
            const im = new Map([...inter.get(ra), ...inter.get(rb), [pr.C, pr]]);
            if (!placement(ix, set, new Set(im.keys())).how) { failed.push(pr); continue; }
            if (ra !== rb) { parent.set(rb, ra); members.set(ra, set); }
            inter.set(ra, im);
            progress = true;
        }
        todo = failed;
    }
    for (const pr of todo) blocked.push({ kind: 'serial', reason: 'hazard', producer: pr.A.idx, consumer: pr.B.idx, container: pr.C });

    const groups = [];
    for (const [root, set] of members) {
        if (find(root) !== root || set.size < 2) continue;
        groups.push({ calls: [...set].sort((a, b) => a - b), pairs: [...inter.get(root).values()] });
    }
    return groups;
}

// ── Parallel fusion ──────────────────────────────────────────────────────────

function independent(ix, a, b) {
    const x = ix.access.get(a), y = ix.access.get(b);
    for (const w of x.writes) if (y.reads.has(w) || y.writes.has(w)) return false;
    for (const w of y.writes) if (x.reads.has(w)) return false;
    return true;
}

function parallelGroups(ix, rules, taken) {
    const buckets = new Map();                                 // backend|elements -> groups (most recent last)
    for (const e of ix.events) {
        if (e.type !== 'skeleton_call' || taken.has(e.idx) || !rules.parallel.includes(baseOf(e.pattern))) continue;
        const key = `${e.backend}|${JSON.stringify(e.elements ?? null)}|${baseOf(e.pattern)}`;
        const list = buckets.get(key) ?? buckets.set(key, []).get(key);
        let placed = false;
        for (let i = list.length - 1; i >= 0 && i >= list.length - 8 && !placed; i--) {
            const g = list[i];
            if (!g.every(m => independent(ix, m, e.idx))) continue;
            if (g.some(m => ix.events[m].prng != null && ix.events[m].prng === e.prng)) continue;
            const s = new Set([...g, e.idx]);
            if (placement(ix, s, new Set()).how) { g.push(e.idx); placed = true; }
        }
        if (!placed) list.push([e.idx]);
    }
    return [...buckets.values()].flat().filter(g => g.length >= 2);
}

// ── Result ───────────────────────────────────────────────────────────────────

const callInfo = e => ({
    id: eventId(e), label: e.label, pattern: e.pattern, duration_us: duration(e),
    ...(e.file ? { file: e.file } : {}), ...(e.line != null ? { line: e.line } : {}),
});

// A loop makes the same group over and over: the groups, and the pairs that cannot be fused, are
// reported once per kind with the number of times they occur.
function aggregate(groups) {
    const byKey = new Map();
    for (const g of groups) {
        const key = [g.kind, g.placement, g.calls.map(c => `${c.label}|${c.pattern}|${c.file ?? ''}|${c.line ?? ''}`).join('>'),
                     (g.intermediates ?? []).map(i => i.container).join(',')].join('#');
        let a = byKey.get(key);
        if (!a) byKey.set(key, a = { ...g, occurrences: 0, example_ids: [], call_time_us: 0, total_launches_saved: 0, ...(g.kind === 'serial' ? { memory_traffic_saved_elements: 0 } : {}) });
        a.occurrences++;
        if (a.example_ids.length < 3) a.example_ids.push(g.calls[0].id);
        a.call_time_us += g.call_time_us;
        a.total_launches_saved += g.launches_saved;
        if (g.kind === 'serial') a.memory_traffic_saved_elements += g.memory_traffic_saved_elements;
    }
    return [...byKey.values()];
}

function patternText([a, b] = []) {
    const [pa, pb] = [baseOf(a), baseOf(b)];
    if (pb === 'Scan') return 'a Scan cannot take a Map: there is no MapScan skeleton (its proxies and uniform arguments have no place in a Scan)';
    if (pa && pa !== 'Map') return `only a Map can be fused into the call that reads its output, not a ${pa}`;
    if (pb) return `a Map cannot be fused into a ${pb}`;
    return 'the skeleton patterns cannot be fused';
}

const REASON_TEXT = {
    'multiple-readers':  c => `"${c}" is read by other calls too`,
    'host-access':       c => `"${c}" is accessed by the host (element access, snapshot, transfer, copy or external) before it changes`,
    'proxy-access':      c => `"${c}" is read through a proxy (random access), not element by element`,
    'pattern':           (c, patterns) => patternText(patterns),
    'recompute':         () => 'fusing would compute the Map once per use of each element, and recomputation is turned off',
    'backend-mismatch':  () => 'the calls run on different backends',
    'size-mismatch':     () => 'the calls have different element counts',
    'shared-prng':       () => 'both draw from the same PRNG stream',
    'scalar-output':     c => `"${c}" is a scalar result of a reduction`,
    'hazard':            () => 'an event between the calls conflicts with moving either of them',
    'strided':           () => 'a call has a stride: it does not read and write the elements one to one',
};

/** A sentence for the reason a pair was not fused, from an entry of `blocked` and the label of the container. */
export const explainBlocked = (reason, container, patterns) => (REASON_TEXT[reason] ?? (() => reason))(container, patterns);

/**
 * The fusion analysis with trace indices and container ids (what the views need to mark the nodes):
 *   serial   [{ kind, placement, calls: [idx], pairs: [{ A, B, C, liveOut, overwrittenBy }] }]   (A, B events, C container)
 *   parallel [{ kind, placement, calls: [idx], pairs: [] }]
 *   blocked  [{ kind, reason, producer: idx, consumer: idx | null, container: id, ... }]
 *   ix       the index of the events (labelOf)
 * Options: rules, include_overlap, as for fusionCandidates().
 */
export function analyzeFusion(events, { rules = {}, include_overlap = false } = {}) {
    const R = { allowRecompute: true, ...DEFAULT_RULES, ...rules };
    const ix = new Index(events);
    const blocked = [];
    const serial = serialGroups(ix, serialPairs(ix, R, blocked), blocked).map(g => ({
        kind: 'serial', calls: g.calls, pairs: g.pairs,
        placement: placement(ix, new Set(g.calls), new Set(g.pairs.map(p => p.C))).how,
    }));
    const inSerial = new Set(serial.flatMap(g => g.calls));
    const parallel = parallelGroups(ix, R, include_overlap ? new Set() : inSerial).map(g => ({
        kind: 'parallel', calls: g, pairs: [], placement: placement(ix, new Set(g), new Set()).how,
    }));
    return { ix, rules: R, serial, parallel, blocked };
}

/**
 * Chains and trees of calls that could be fused serially, groups of independent calls that
 * could be fused in parallel, and the pairs that cannot be fused with the reason.
 *
 * Options: rules (see DEFAULT_RULES); blocked_limit (default 30); include_overlap (let a call be in a
 * serial and a parallel group; default false: parallel groups only use the calls left over);
 * aggregate (default true: equal groups, e.g. of the iterations of a loop, are one entry with the
 * number of occurrences, the times and launches summed; false: every group).
 */
export function fusionCandidates(events, { rules = {}, blocked_limit = 30, include_overlap = false, aggregate: doAggregate = true } = {}) {
    const { ix, rules: R, serial: serialRaw, parallel: parallelRaw, blocked } = analyzeFusion(events, { rules, include_overlap });

    const serial = serialRaw.map(g => {
        const calls = g.calls.map(i => events[i]);
        const intermediates = g.pairs.map(p => ({
            container: ix.labelOf(p.C), elements: p.A.elements ?? null,
            ends: p.liveOut ? 'live-out' : p.overwrittenBy ? 'overwritten by the consumer' : 'overwritten or freed',
        }));
        const elems = n => (Array.isArray(n) ? n.reduce((a, b) => a * b, 1) : 0);
        const recompute = g.pairs.filter(p => p.recompute).map(p => ({
            producer: p.A.label, consumer: p.B.label, container: ix.labelOf(p.C), kind: p.recompute.kind,
            factor: p.recompute.factor, note: p.recompute.note,
        }));
        return {
            kind: 'serial', placement: g.placement,
            calls: calls.map(callInfo), intermediates,
            ...(recompute.length ? { recompute } : {}),
            call_time_us: calls.reduce((s, e) => s + duration(e), 0),
            launches_saved: calls.length - 1,
            memory_traffic_saved_elements: g.pairs.reduce((s, p) => s + elems(p.A.elements) * (p.liveOut ? 1 : 2), 0),
        };
    });

    const parallel = parallelRaw.map(g => {
        const calls = g.calls.map(i => events[i]);
        const reads = new Map();
        for (const i of g.calls) for (const x of ix.access.get(i).reads) reads.set(x, (reads.get(x) ?? 0) + 1);
        return {
            kind: 'parallel', placement: g.placement,
            calls: calls.map(callInfo),
            shared_inputs: [...reads].filter(([, n]) => n > 1).map(([x]) => ix.labelOf(x)),
            call_time_us: calls.reduce((s, e) => s + duration(e), 0),
            launches_saved: calls.length - 1,
        };
    });

    // The pairs that could not be fused.
    const byReason = {};
    const sig = new Map();
    for (const b of blocked) {
        byReason[b.reason] = (byReason[b.reason] ?? 0) + 1;
        const p = events[b.producer], c = b.consumer != null ? events[b.consumer] : null;
        const key = [b.reason, p.label, c?.label, ix.labelOf(b.container)].join('#');
        let a = sig.get(key);
        if (!a) sig.set(key, a = {
            kind: b.kind, reason: b.reason, container: ix.labelOf(b.container),
            producer: callInfo(p), ...(c ? { consumer: callInfo(c) } : {}),
            ...(b.all ? { all_reasons: b.all } : {}), ...(b.patterns ? { patterns: b.patterns } : {}), ...(b.with ? { with: b.with } : {}),
            occurrences: 0, call_time_us: 0,
        });
        a.occurrences++;
        a.call_time_us += duration(p) + (c ? duration(c) : 0);
    }
    const blockedList = [...sig.values()].sort((a, b) => b.call_time_us - a.call_time_us);

    const shown = doAggregate ? aggregate([...serial, ...parallel]) : [...serial, ...parallel];
    const total = k => shown.reduce((s, g) => s + k(g), 0);
    return {
        serial_groups: serial.length, parallel_groups: parallel.length,         // in the trace, every occurrence
        distinct_groups: doAggregate ? shown.length : null,
        fusible_calls: [...serial, ...parallel].reduce((s, g) => s + g.calls.length, 0),
        call_time_us: [...serial, ...parallel].reduce((s, g) => s + g.call_time_us, 0),
        launches_saved: [...serial, ...parallel].reduce((s, g) => s + g.launches_saved, 0),
        groups: shown.sort((a, b) => b.call_time_us - a.call_time_us),
        blocked_by_reason: byReason,
        blocked: blockedList.slice(0, blocked_limit),
        rules: R,
    };
}
