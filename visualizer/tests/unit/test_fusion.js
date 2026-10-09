// test_fusion.js — the fusion detector (backend/fusion.js): what is fused, and why not.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fusionCandidates, explainBlocked } from '../../script/backend/fusion.js';
import { parseTrace } from '../../script/backend/trace.js';
import { builder, trace } from '../event_builders.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const names = g => g.calls.map(c => c.label);

// X is read by A, which writes C; B reads C and writes Y. (Allocations first, so that the labels are known.)
function chain(make) {
    const b = builder();
    const events = [b.alloc(1, 'X'), b.alloc(2, 'C'), b.alloc(3, 'Y'), b.alloc(4, 'Z'), ...make(b)];
    return fusionCandidates(trace(events));
}
const reasonOf = r => r.blocked.map(x => x.reason);

describe('serial fusion', () => {
    it('fuses a producer with the only consumer of its output', () => {
        const r = chain(b => [b.call('A', { ins: [1], outs: [2] }), b.call('B', { ins: [2], outs: [3] }), b.dealloc(2, 'C')]);
        expect(r.serial_groups).toBe(1);
        const g = r.groups[0];
        expect(names(g)).toEqual(['A', 'B']);
        expect(g.intermediates).toMatchObject([{ container: 'C', ends: 'overwritten or freed' }]);
        expect(g).toMatchObject({ kind: 'serial', launches_saved: 1, call_time_us: 200, memory_traffic_saved_elements: 200 });
    });

    it('says when the intermediate is live-out, and then saves only the read', () => {
        const r = chain(b => [b.call('A', { ins: [1], outs: [2] }), b.call('B', { ins: [2], outs: [3] })]);
        expect(r.groups[0].intermediates[0].ends).toBe('live-out');
        expect(r.groups[0].memory_traffic_saved_elements).toBe(100);
    });

    it('fuses a consumer that overwrites the intermediate', () => {
        const r = chain(b => [b.call('A', { ins: [1], outs: [2] }), b.call('B', { ins: [2], outs: [2] })]);
        expect(r.groups[0].intermediates[0].ends).toBe('overwritten by the consumer');
    });

    it('does not fuse when another call reads the intermediate too', () => {
        const r = chain(b => [b.call('A', { ins: [1], outs: [2] }), b.call('B', { ins: [2], outs: [3] }), b.call('Q', { ins: [2], outs: [4] })]);
        expect(r.serial_groups).toBe(0);
        expect(r.blocked[0]).toMatchObject({ reason: 'multiple-readers', container: 'C' });
    });

    it('does not fuse when the host looks at the intermediate (element access, snapshot, transfer, external)', () => {
        for (const observe of [b => b.access(9, 2), b => b.snapshot(2), b => b.transfer(2), b => b.external('print', { scalars: [2] }), b => b.copy(2, 4), b => b.comm([2])]) {
            const r = chain(b => [b.call('A', { ins: [1], outs: [2] }), observe(b), b.call('B', { ins: [2], outs: [3] })]);
            expect(reasonOf(r)[0]).toBe('host-access');
            expect(r.serial_groups).toBe(0);
        }
    });

    it('does not fuse a consumer that reads the intermediate through a proxy', () => {
        const r = chain(b => [b.call('A', { ins: [1], outs: [2] }), b.call('B', { proxy: [2], outs: [3] })]);
        expect(reasonOf(r)).toEqual(['proxy-access']);
    });

    it('does not fuse across backends, element counts, or one PRNG stream', () => {
        let r = chain(b => [b.call('A', { ins: [1], outs: [2], backend: 'OpenCL' }), b.call('B', { ins: [2], outs: [3], backend: 'OpenMP' })]);
        expect(reasonOf(r)).toEqual(['backend-mismatch']);
        r = chain(b => [b.call('A', { ins: [1], outs: [2], n: 100 }), b.call('B', { ins: [2], outs: [3], n: 50 })]);
        expect(reasonOf(r)).toEqual(['size-mismatch']);
        r = chain(b => [b.alloc(9, 'rng'), b.call('A', { ins: [1], outs: [2], prng: 9 }), b.call('B', { ins: [2], outs: [3], prng: 9 })]);
        expect(reasonOf(r)).toEqual(['shared-prng']);
        r = chain(b => [b.alloc(9, 'r1'), b.alloc(8, 'r2'), b.call('A', { ins: [1], outs: [2], prng: 9 }), b.call('B', { ins: [2], outs: [3], prng: 8 })]);
        expect(r.serial_groups).toBe(1);
    });

    it('follows the table of fusable patterns', () => {
        const scan = chain(b => [b.call('A', { ins: [1], outs: [2] }), b.call('S', { pattern: 'Scan', ins: [2], outs: [3] })]);
        expect(scan.blocked[0]).toMatchObject({ reason: 'pattern', patterns: ['Map', 'Scan'] });
        const overlap = chain(b => [b.call('O', { pattern: 'MapOverlap 4D', proxy: [1], outs: [2] }), b.call('B', { ins: [2], outs: [3] })]);
        expect(reasonOf(overlap)).toEqual(['pattern']);
        const reduce = chain(b => [b.call('A', { ins: [1], outs: [2] }), b.call('R', { pattern: 'Reduce 1D-Full', ins: [2], outs: [3] })]);
        expect(names(reduce.groups[0])).toEqual(['A', 'R']);
        const scalar = chain(b => [b.call('R', { pattern: 'Reduce', ins: [1], outs: [2] }), b.call('B', { ins: [2], outs: [3] })]);
        expect(reasonOf(scalar)).toEqual(['scalar-output']);
        // the table can be changed
        const b = builder();
        const ev = trace([b.alloc(1, 'X'), b.alloc(2, 'C'), b.alloc(3, 'Y'), b.call('O', { pattern: 'MapOverlap', proxy: [1], outs: [2] }), b.call('B', { ins: [2], outs: [3] })]);
        expect(fusionCandidates(ev, { rules: { serialProducers: ['Map', 'MapOverlap'] } }).serial_groups).toBe(1);
    });

    describe('hazards of moving a call past the events between', () => {
        it('hoists the consumer when the producer cannot sink past an overwrite of its input', () => {
            // A reads X; W overwrites X; B reads C. Fusing at B's place would make A read the new X,
            // but B does not depend on W, so the fused call goes to A's place.
            const r = chain(b => [b.call('A', { ins: [1], outs: [2] }), b.call('W', { ins: [4], outs: [1] }), b.call('B', { ins: [2], outs: [3] })]);
            expect(r.groups[0]).toMatchObject({ kind: 'serial', placement: 'hoist' });
        });
        it('does not fuse when neither end can move', () => {
            // W (a Scan, which cannot be fused) overwrites A's input X and produces B's other input Z:
            // A cannot sink past W, B cannot hoist past W.
            const r = chain(b => [b.call('A', { ins: [1], outs: [2] }), b.call('W', { pattern: 'Scan', ins: [4], outs: [1, 4] }), b.call('B', { ins: [2, 4], outs: [3] })]);
            expect(r.serial_groups).toBe(0);
            expect(r.blocked).toContainEqual(expect.objectContaining({ reason: 'hazard', container: 'C' }));
        });
        it('fuses the call in the way too when it is fusable itself', () => {
            // The same with W a Map: A, W and B become one call, which keeps their order for every element.
            const r = chain(b => [b.call('A', { ins: [1], outs: [2] }), b.call('W', { ins: [4], outs: [1, 4] }), b.call('B', { ins: [2, 4], outs: [3] })]);
            expect(names(r.groups[0])).toEqual(['A', 'W', 'B']);
        });
        it('sinks the producer when only the consumer\'s side is in the way', () => {
            // W writes Z, which B reads: B cannot hoist. A reads X, which W leaves alone: A can sink.
            const r = chain(b => [b.call('A', { ins: [1], outs: [2] }), b.call('W', { ins: [1], outs: [4] }), b.call('B', { ins: [2, 4], outs: [3] })]);
            expect(r.groups.find(g => g.kind === 'serial')).toMatchObject({ placement: 'sink' });
        });
        it('fuses a pair that is in the way of itself once the calls between it join the group', () => {
            // A writes C and D; R reads D and writes E; B reads C and E. A->B alone cannot move (A cannot sink past
            // R, which reads D; B cannot hoist past R, which writes E), but A->R->B is a chain and then C goes with it.
            const r = chain(b => [b.alloc(5, 'D'), b.alloc(6, 'E'), b.call('A', { ins: [1], outs: [2, 5] }), b.call('R', { ins: [5], outs: [6] }), b.call('B', { ins: [2, 6], outs: [3] })]);
            expect(names(r.groups.find(g => g.kind === 'serial'))).toEqual(['A', 'R', 'B']);
            expect(r.groups[0].intermediates.map(i => i.container).sort()).toEqual(['C', 'D', 'E']);
            expect(r.blocked).toHaveLength(0);
        });
        it('does not fuse a producer with a reader of its output when a call that reads the other output is in the way and nothing can move', () => {
            // A writes C; R (not fusable: it uses A's input X as a proxy-written... ) -- see the proxy case below
            const r = chain(b => [b.call('A', { ins: [1], outs: [2] }), b.call('R', { proxy: [2], outs: [4] }), b.call('B', { ins: [2], outs: [3] })]);
            expect(r.serial_groups).toBe(0);
        });
        it('does not fuse calls that read through a proxy a container that another call of the group writes', () => {
            // O reads X by proxy, B writes X: in one call, O would see B's writes at other elements.
            const r = chain(b => [b.call('O', { proxy: [1], outs: [2] }), b.call('B', { ins: [2], outs: [1] })]);
            expect(r.serial_groups).toBe(0);
        });
    });

    it('fuses a fan-in tree and a producer with two consumers into one group', () => {
        const fanIn = chain(b => [b.alloc(5, 'C2'), b.call('A1', { ins: [1], outs: [2] }), b.call('A2', { ins: [1], outs: [5] }), b.call('B', { ins: [2, 5], outs: [3] })]);
        expect(names(fanIn.groups.find(g => g.kind === 'serial'))).toEqual(['A1', 'A2', 'B']);
        const fanOut = chain(b => [b.alloc(5, 'D'), b.alloc(6, 'E'), b.call('A', { ins: [1], outs: [2, 5] }), b.call('B', { ins: [2], outs: [3] }), b.call('Q', { ins: [5], outs: [6] })]);
        expect(names(fanOut.groups.find(g => g.kind === 'serial'))).toEqual(['A', 'B', 'Q']);
    });

    it('makes a long chain one group', () => {
        const r = chain(b => [b.alloc(5, 'D'), b.call('A', { ins: [1], outs: [2] }), b.call('B', { ins: [2], outs: [3] }), b.call('C2', { ins: [3], outs: [5] })]);
        expect(r.groups[0]).toMatchObject({ kind: 'serial', launches_saved: 2 });
        expect(names(r.groups[0])).toEqual(['A', 'B', 'C2']);
    });
});

describe('parallel fusion', () => {
    it('fuses independent calls over the same elements that share an input', () => {
        const r = chain(b => [b.alloc(5, 'P'), b.call('M1', { ins: [1], outs: [2] }), b.call('M2', { ins: [1], outs: [3] }), b.call('M3', { ins: [1], outs: [5] })]);
        expect(r.parallel_groups).toBe(1);
        expect(r.groups[0]).toMatchObject({ kind: 'parallel', launches_saved: 2, shared_inputs: ['X'] });
    });
    it('does not fuse calls where one reads or writes what the other writes', () => {
        const dep = chain(b => [b.call('M1', { ins: [1], outs: [2] }), b.call('M2', { ins: [4], outs: [2] })]);   // both write C
        expect(dep.parallel_groups).toBe(0);
        const war = chain(b => [b.call('M1', { ins: [1], outs: [2] }), b.call('M2', { ins: [4], outs: [1] })]);   // M2 overwrites M1's input
        expect(war.parallel_groups).toBe(0);
    });
    it('does not fuse calls of other element counts, backends or patterns', () => {
        expect(chain(b => [b.call('M1', { ins: [1], outs: [2] }), b.call('M2', { ins: [1], outs: [3], n: 50 })]).parallel_groups).toBe(0);
        expect(chain(b => [b.call('M1', { ins: [1], outs: [2] }), b.call('M2', { ins: [1], outs: [3], backend: 'OpenCL' })]).parallel_groups).toBe(0);
        expect(chain(b => [b.call('M1', { ins: [1], outs: [2] }), b.call('S', { pattern: 'Scan', ins: [1], outs: [3] })]).parallel_groups).toBe(0);
    });
    it('does not fuse across an event that the later call depends on', () => {
        // M1, then W writes Z, then M2 reads Z: M2 cannot hoist to M1; M1 can sink to M2 only if W does not use M1's outputs or inputs.
        const r = chain(b => [b.call('M1', { ins: [1], outs: [2] }), b.call('W', { ins: [1], outs: [4] }), b.call('M2', { ins: [4], outs: [3] })]);
        expect(r.groups.every(g => g.kind !== 'parallel' || !names(g).includes('M2') || !names(g).includes('M1'))).toBe(true);
    });
    it('does not fuse two calls that draw from one PRNG stream', () => {
        const r = chain(b => [b.alloc(9, 'rng'), b.call('M1', { ins: [1], outs: [2], prng: 9 }), b.call('M2', { ins: [1], outs: [3], prng: 9 })]);
        expect(r.parallel_groups).toBe(0);
    });
    it('leaves the calls of serial groups out unless asked', () => {
        const b = builder();
        const ev = trace([b.alloc(1, 'X'), b.alloc(2, 'C'), b.alloc(3, 'Y'), b.alloc(4, 'Z'),
                          b.call('A', { ins: [1], outs: [2] }), b.call('B', { ins: [2], outs: [3] }), b.call('M', { ins: [1], outs: [4] })]);
        expect(fusionCandidates(ev).parallel_groups).toBe(0);
        expect(fusionCandidates(ev, { include_overlap: true }).parallel_groups).toBeGreaterThan(0);
    });
});

describe('the result', () => {
    it('does not change the events and is the same twice', () => {
        const b = builder();
        const events = trace([b.alloc(1, 'X'), b.alloc(2, 'C'), b.alloc(3, 'Y'), b.call('A', { ins: [1], outs: [2] }), b.call('B', { ins: [2], outs: [3] })]);
        const before = JSON.stringify(events);
        expect(fusionCandidates(events)).toEqual(fusionCandidates(events));
        expect(JSON.stringify(events)).toBe(before);
    });
    it('counts the blocked pairs by reason', () => {
        const r = chain(b => [b.call('A', { ins: [1], outs: [2] }), b.call('B', { proxy: [2], outs: [3] }), b.call('C2', { ins: [3], outs: [4] }), b.access(9, 3)]);
        expect(r.blocked_by_reason).toMatchObject({ 'proxy-access': 1 });
    });
});

describe('the DNN CIFAR-10 trace (OpenCL)', () => {
    const events = parseTrace(readFileSync(join(ROOT, 'examples/dnn_cifar10/trace_dnn_cifar10_opencl.json'), 'utf8')).events;
    const r = fusionCandidates(events);
    const raw = fusionCandidates(events, { aggregate: false });

    it('finds the weight update that follows the weight gradient, and says what blocks the rest', () => {
        const update = r.groups.find(g => g.kind === 'serial' && names(g).join(',') === 'L1 Conv2D backprop weights,L1 Conv2D Adam Weights');
        expect(update).toBeDefined();
        expect(update.intermediates[0].container).toBe('L1 Conv2D weight deltas');
        expect(Object.keys(r.blocked_by_reason).sort()).toEqual(['hazard', 'host-access', 'multiple-readers', 'pattern', 'proxy-access']);
    });
    it('reports the groups of a loop once, with how many there are', () => {
        const update = r.groups.find(g => names(g).join(',') === 'L1 Conv2D backprop weights,L1 Conv2D Adam Weights');
        expect(update.occurrences).toBeGreaterThan(1);
        expect(update.call_time_us).toBeGreaterThan(update.calls.reduce((s, c) => s + c.duration_us, 0));
        expect(r.distinct_groups).toBeLessThan(r.serial_groups + r.parallel_groups);
        expect(raw.groups.length).toBe(r.serial_groups + r.parallel_groups);
    });
    it('never puts a call in two serial groups', () => {
        const seen = new Set();
        for (const g of raw.groups.filter(x => x.kind === 'serial')) for (const c of g.calls) { expect(seen.has(c.id)).toBe(false); seen.add(c.id); }
    });
    it('puts only independent calls in a parallel group', () => {
        for (const g of raw.groups.filter(x => x.kind === 'parallel')) expect(new Set(g.calls.map(c => c.id)).size).toBe(g.calls.length);
    });
});

describe('a Map into the skeletons that read with a neighbourhood or in pairs', () => {
    const into = (pattern, opts = {}, rules) => {
        const b = builder();
        const events = [b.alloc(1, 'X'), b.alloc(2, 'C'), b.alloc(3, 'Y'), b.call('A', { ins: [1], outs: [2] }), b.call('B', { pattern, ins: [2], outs: [3], ...opts })];
        return fusionCandidates(trace(events), { rules });
    };

    it('fuses a Map into a MapReduce and a Reduce without recomputation', () => {
        for (const p of ['MapReduce', 'Reduce 1D-Full']) {
            const r = into(p);
            expect(r.serial_groups).toBe(1);
            expect(r.groups[0].recompute).toBeUndefined();
        }
    });

    it('fuses a Map into a MapOverlap or a MapPool, with the output of other extents, and says it recomputes', () => {
        const o = into('MapOverlap 2D', { elements: [98] });
        expect(o.serial_groups).toBe(1);
        expect(o.groups[0].recompute).toMatchObject([{ producer: 'A', consumer: 'B', kind: 'overlap', factor: null }]);
        const p = into('MapPool 4D', { elements: [5, 5] });
        expect(p.groups[0].recompute[0].kind).toBe('pool');
    });

    it('fuses a Map into a MapPairs or MapPairsReduce when it is one of the vectors, with the factor from the other', () => {
        for (const p of ['MapPairs', 'MapPairsReduce']) {
            const r = into(p, { elements: [100, 7] });             // Hsize 100, Vsize 7: A is the H vector, used for each of 7 rows
            expect(r.serial_groups).toBe(1);
            expect(r.groups[0].recompute).toMatchObject([{ kind: 'pairs', factor: 7 }]);
            expect(into(p, { elements: [7, 100] }).groups[0].recompute[0].factor).toBe(7);
            expect(into(p, { elements: [7, 9] }).blocked[0].reason).toBe('size-mismatch');
        }
    });

    it('can be told not to recompute', () => {
        const r = into('MapOverlap 2D', { elements: [98] }, { allowRecompute: false });
        expect(r.serial_groups).toBe(0);
        expect(r.blocked[0].reason).toBe('recompute');
    });

    it('does not fuse into a Scan, and says why', () => {
        const r = into('Scan');
        expect(r.serial_groups).toBe(0);
        expect(r.blocked[0]).toMatchObject({ reason: 'pattern', patterns: ['Map', 'Scan'] });
        expect(explainBlocked('pattern', 'C', ['Map', 'Scan'])).toMatch(/MapScan/);
    });

    it('does not fuse a MapOverlap into a Map', () => {
        const b = builder();
        const r = fusionCandidates(trace([b.alloc(1, 'X'), b.alloc(2, 'C'), b.alloc(3, 'Y'), b.call('O', { pattern: 'MapOverlap 2D', ins: [1], outs: [2] }), b.call('B', { ins: [2], outs: [3] })]));
        expect(r.serial_groups).toBe(0);
        expect(explainBlocked('pattern', 'C', ['MapOverlap 2D', 'Map'])).toMatch(/only a Map/);
    });
});

describe('fusion with the parameters that the tracer records', () => {
    const into = (pattern, params, opts = {}) => {
        const b = builder();
        const events = [b.alloc(1, 'X'), b.alloc(2, 'C'), b.alloc(3, 'Y'), b.call('A', { ins: [1], outs: [2] }), b.call('B', { pattern, ins: [2], outs: [3], params, ...opts })];
        return fusionCandidates(trace(events));
    };

    it('fuses into a pool without recomputation if the windows do not overlap', () => {
        const r = into('MapPool 2D', { pool: true, pool_size: [2, 2], stride: [2, 2] }, { elements: [8, 8] });
        expect(r.serial_groups).toBe(1);
        expect(r.groups[0].recompute).toBeUndefined();
    });

    it('says how many times an element is computed in a pool whose windows overlap, and in an overlap', () => {
        expect(into('MapPool 2D', { pool: true, pool_size: [3, 3], stride: [1, 2] }, { elements: [8, 8] }).groups[0].recompute[0])
            .toMatchObject({ kind: 'pool', factor: 3 * 2 });
        expect(into('MapOverlap 2D', { overlap: [1, 2] }, { elements: [8, 8] }).groups[0].recompute[0])
            .toMatchObject({ kind: 'overlap', factor: 3 * 5 });
        expect(into('MapOverlap 1D', { overlap: [2], stride: [2] }).groups[0].recompute[0]).toMatchObject({ factor: 3 });
    });

    it('does not fuse a call that has a stride', () => {
        const r = into('Map', { strides: [2, 1] });
        expect(r.serial_groups).toBe(0);
        expect(r.blocked[0].reason).toBe('strided');
        expect(into('Map', undefined).serial_groups).toBe(1);
    });

    it('fuses a MapPairs into a Map over a matrix, which has the element count of the matrix', () => {
        const b = builder();
        const events = [b.alloc(1, 'U'), b.alloc(2, 'D'), b.alloc(3, 'R'),
            b.call('P', { pattern: 'MapPairs', elements: [64, 32], ins: [1], outs: [2] }), b.call('M', { elements: [2048], ins: [2], outs: [3] })];
        expect(fusionCandidates(trace(events)).serial_groups).toBe(1);
    });
});

// examples/fusion_patterns: one region for each case, made by examples/synthetic_traces.py.
describe('the fusion_patterns example', () => {
    const { events } = parseTrace(readFileSync(join(ROOT, 'examples/fusion_patterns/trace_fusion_patterns_synthetic.json'), 'utf8'));
    const r = fusionCandidates(events);
    const groups = r.groups.map(g => `${g.kind}: ${names(g).join(' + ')}`).sort();

    it('finds the groups of the fusible blocks', () => {
        expect(groups).toEqual([
            'parallel: inc + dec', 'parallel: max_of + min_of', 'parallel: sq + neg + halve', 'parallel: times_two + negate_it',
            'serial: brighten + laplace', 'serial: normalize + distance', 'serial: pair_sum + reciprocal', 'serial: relu + max_pool',
            'serial: scale + shift + clamp', 'serial: square_sum + add_all', 'serial: square_u + negate_v + add_uv',
        ]);
    });

    it('says which fusions recompute', () => {
        const cost = Object.fromEntries(r.groups.filter(g => g.recompute).map(g => [g.calls[1].label, g.recompute[0].kind + ':' + g.recompute[0].factor]));
        expect(cost).toEqual({ laplace: 'overlap:9', distance: 'pairs:160' });      // (the pool has windows that do not overlap: no cost)
    });

    it('gives the reason of each call that is almost fusible', () => {
        const why = Object.fromEntries(r.blocked.map(b => [`${b.producer.label}>${b.consumer?.label}`, b.reason]));
        expect(why).toMatchObject({
            'double_x>prefix_sum': 'pattern', 'blur>threshold': 'pattern', 'plus_one>times_two': 'multiple-readers',
            'omp_add>T': 'host-access', 'add_one>T': 'host-access', 'bump>lookup': 'proxy-access', 'shift_vec>scale_mat': 'size-mismatch',
            'jitter>noise': 'shared-prng', 'increment>combine': 'hazard', 'total_of>normalize_by': 'scalar-output', 'step_f>step_g': 'strided',
        });
    });

    it('does not fuse across a copy of the intermediate, and says that the copy reads it', () => {
        const withCopy = r.blocked.filter(b => b.with?.includes('copy'));
        expect(withCopy.map(b => `${b.reason} ${b.producer.label}`)).toEqual(['host-access add_c']);
    });
});
