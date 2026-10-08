// test_designer.js — the designer: a design is checked and lowered to a program (designer/model.js), and the program is written as
// SkePU C++ and SkePY Python (designer/codegen.js).
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { lower, sampleDesign, allSkeletonsDesign, keepLoopsTogether, emptyDesign, defaultContainer, defaultSkeleton, identifier, inferOutput, effectiveContainers, transfers, defaultPrng, duplicateNodes, Residency, placeOf,
         objectKey, membersOf, renameObject, joinObjects, detachNode, syncObject } from '../../designer/model.js';
import { compareTrace } from '../../designer/compare.js';
import { emitCpp, emitPython, containerPlacement } from '../../designer/codegen.js';
import { synthesizeTrace } from '../../designer/trace.js';
import { randomDesign, SIZES, ARITIES, SHAPES } from '../../designer/random.js';
import { DirectedGraph } from '../../script/backend/graph.js';

let n = 0;
/**
 * A design of containers [handle, kind, size, object], skeletons [label, pattern, patch] and edges [from, to, role], by label; the calls from the
 * top in the order given. A container that has an `object` (the handle of another container) is an update of it: the same object, so the
 * same name (the handle of an update is only what the edges say).
 */
function make(containers, skeletons, edges, loops = []) {
    const d = emptyDesign();
    const id = {};
    for (const [handle, kind = 'Vector', size, object] of containers) {
        id[handle] = `c${++n}`;
        d.containers.push({ ...defaultContainer(id[handle], object ?? handle, kind), ...(size ? { size } : {}), ...(object ? { object: id[object] } : {}) });
    }
    skeletons.forEach(([label, pattern, patch = {}], i) => {
        id[label] = `s${++n}`;
        d.skeletons.push({ ...defaultSkeleton(pattern, id[label], label), ...patch });
        d.positions[id[label]] = { x: 0, y: i * 100 };
    });
    for (const [l, parent] of loops) { id[l] = `l${++n}`; d.loops.push({ id: id[l], label: l, count: 3, parent: null }); }
    for (const [from, to, role] of edges) d.edges.push({ id: `e${++n}`, from: id[from], to: id[to], role });
    return { d, id };
}
const problems = r => r.errors.map(e => e.msg);

describe('lowering a design', () => {
    it('makes a program of the example', () => {
        const r = lower(sampleDesign());
        expect(r.ok).toBe(true);
        expect(r.program.calls.map(c => c.name)).toEqual(['smooth', 'scale', 'energy']);
        expect(r.program.functions.map(f => f.name)).toEqual(['smooth_f', 'scale_f', 'energy_map', 'energy_reduce']);
        expect(r.program.steps.map(s => s.type)).toEqual(['loop', 'call', 'call']);
    });

    it('orders the calls by the data flow, whatever their places', () => {
        const { d, id } = make([['a'], ['b'], ['c']], [['first', 'Map'], ['second', 'Map']], [['a', 'first', 'elwise'], ['first', 'b', 'out'], ['b', 'second', 'elwise'], ['second', 'c', 'out']]);
        d.positions[id.first].y = 500;
        const r = lower(d);
        expect(r.program.calls.map(c => c.name)).toEqual(['first', 'second']);
        expect(r.warnings.map(w => w.msg).join()).not.toMatch(/before the call that writes it/);
    });

    it('orders by place the calls that the data flow does not order', () => {
        const { d, id } = make([['a'], ['b'], ['c'], ['x'], ['y']], [['p', 'Map'], ['q', 'Map'], ['r', 'Map']],
            [['a', 'p', 'elwise'], ['p', 'b', 'out'], ['b', 'q', 'elwise'], ['q', 'c', 'out'], ['x', 'r', 'elwise'], ['r', 'y', 'out']]);
        d.positions[id.r].y = -10;                      // r is independent: its place decides, above everything
        expect(lower(d).program.calls.map(c => c.name)).toEqual(['r', 'p', 'q']);
        d.positions[id.r].y = 50;                       // between p (0) and q (100)
        expect(lower(d).program.calls.map(c => c.name)).toEqual(['p', 'r', 'q']);
    });

    it('orders the readers and writers of the updates of an object: a reader comes before the write of the next update', () => {
        // w1 writes a; w2 reads it and writes the next update of the same data (a2); rd reads a too, so it must come before w2
        const { d, id } = make([['a'], ['a2', 'Vector', undefined, 'a'], ['r']], [['w1', 'Map'], ['rd', 'Map'], ['w2', 'Map']],
            [['w1', 'a', 'out'], ['a', 'rd', 'elwise'], ['rd', 'r', 'out'], ['a', 'w2', 'elwise'], ['w2', 'a2', 'out']]);
        d.positions[id.rd].y = 500;                     // (below w2 by place: the data decides)
        d.positions[id.w2].y = 200;
        const r = lower(d);
        expect(r.errors).toEqual([]);
        expect(r.program.calls.map(c => c.name)).toEqual(['w1', 'rd', 'w2']);
        expect(r.versions).toEqual({ [id.a]: { version: 1, count: 2 }, [id.a2]: { version: 2, count: 2 }, [id.r]: { version: 1, count: 1 } });
        expect(r.program.containers.map(c => c.name)).toEqual(['a', 'r']);             // one variable for the object
        expect(r.program.calls[2].out.name).toBe('a');                                 // w2(a, a): the update is written in place
    });

    it('orders the updates of an object by the places of their writers where the data flow does not, and says so', () => {
        const { d, id } = make([['a'], ['a2', 'Vector', undefined, 'a']], [['w1', 'Map'], ['w2', 'Map']], [['w1', 'a', 'out'], ['w2', 'a2', 'out']]);
        d.positions[id.w1].y = 300; d.positions[id.w2].y = 100;
        const r = lower(d);
        expect(r.errors).toEqual([]);
        expect(r.program.calls.map(c => c.name)).toEqual(['w2', 'w1']);                // by place
        expect(r.versions[id.a2]).toEqual({ version: 1, count: 2 });
        expect(r.warnings.map(w => w.msg).join()).toMatch(/does not say whether w[12] or w[12] writes it first/);
    });

    it('has a node that nothing writes as the allocation of the object, and the updates after it', () => {
        const { d, id } = make([['a'], ['a2', 'Vector', undefined, 'a']], [['f', 'Map']], [['a', 'f', 'elwise'], ['f', 'a2', 'out']]);
        const r = lower(d);
        expect(r.errors).toEqual([]);
        expect(r.versions).toEqual({ [id.a2]: { version: 1, count: 1 } });             // (one update: no number)
        expect(r.program.calls[0]).toMatchObject({ name: 'f' });
        expect(r.warnings.map(w => w.msg).join()).not.toMatch(/does not say/);
    });

    it('says when an update has two writers, or a second allocation, or another format, or is read and written by one call', () => {
        let { d, id } = make([['a']], [['f', 'Map'], ['g', 'Map']], [['f', 'a', 'out'], ['g', 'a', 'out']]);
        expect(problems(lower(d)).join()).toMatch(/a is written by f and g: an update has one writer/);
        ({ d, id } = make([['a'], ['a2', 'Vector', undefined, 'a']], [['f', 'Map']], [['a', 'f', 'elwise']]));
        expect(problems(lower(d)).join()).toMatch(/a has a second node that nothing writes/);
        ({ d, id } = make([['a', 'Vector', [8]], ['a2', 'Vector', [9], 'a']], [['f', 'Map'], ['g', 'Map']], [['f', 'a', 'out'], ['g', 'a2', 'out']]));
        expect(problems(lower(d)).join()).toMatch(/Vector<float> 8 and Vector<float> 9 are not the same object/);
        ({ d, id } = make([['a']], [['f', 'Map']], [['a', 'f', 'elwise'], ['f', 'a', 'out']]));
        expect(problems(lower(d)).join()).toMatch(/f reads and writes a as the same update/);
    });

    it('has the name of the object for all its nodes: one variable, and names that differ between objects', () => {
        const { d, id } = make([['a'], ['a2', 'Vector', undefined, 'a'], ['b'], ['c']], [['f', 'Map'], ['g', 'Map'], ['h', 'Map']],
            [['a', 'f', 'elwise'], ['f', 'a2', 'out'], ['a2', 'g', 'elwise'], ['g', 'b', 'out'], ['b', 'h', 'elwise'], ['h', 'c', 'out']]);
        const r = lower(d);
        expect(r.errors).toEqual([]);
        expect(r.program.calls.map(c => [c.name, c.out.name, c.elwise.map(x => x.name)])).toEqual([['f', 'a', ['a']], ['g', 'b', ['a']], ['h', 'c', ['b']]]);
        expect(r.program.containers.map(c => [c.name, c.nodes.length])).toEqual([['a', 2], ['b', 1], ['c', 1]]);
        const text = emitCpp(r.program).text;
        expect(text.match(/skepu::Vector<float> a\(/g)).toHaveLength(1);
        expect(r.program.calls[0].out.id).toBe(id.a2);                                 // (the node that is written is the update)
    });

    it('keeps the calls of a loop together, whatever their places', () => {
        const { d, id } = make([['a'], ['b'], ['x'], ['y'], ['z']], [['f', 'Map'], ['g', 'Map'], ['h', 'Map']],
            [['a', 'f', 'elwise'], ['f', 'b', 'out'], ['x', 'h', 'elwise'], ['h', 'y', 'out'], ['b', 'g', 'elwise'], ['g', 'z', 'out']], [['Loop']]);
        d.skeletons.find(s => s.id === id.f).parent = id.Loop;
        d.skeletons.find(s => s.id === id.g).parent = id.Loop;
        d.positions[id.f].y = 0; d.positions[id.h].y = 100; d.positions[id.g].y = 200;     // h, which is not in the loop, is between by place
        const r = lower(d);
        expect(r.ok).toBe(true);
        expect(r.program.steps.map(s => s.type === 'loop' ? s.steps.map(c => c.call.name) : s.call.name)).toEqual([['f', 'g'], 'h']);
    });

    it('uses the place for a cycle of dependences, and says so', () => {
        const { d } = make([['a'], ['b']], [['f', 'Map'], ['g', 'Map']], [['a', 'f', 'elwise'], ['f', 'b', 'out'], ['b', 'g', 'elwise'], ['g', 'a', 'out']]);
        const r = lower(d);
        expect(r.program.calls.map(c => c.name)).toEqual(['f', 'g']);
        expect(r.warnings.map(w => w.msg).join()).toMatch(/is in a cycle of dependences/);
    });

    it('makes names that are identifiers and different', () => {
        const used = new Set();
        expect([identifier('my data', used), identifier('my data', used), identifier('3d', used), identifier('int', used), identifier('', used)])
            .toEqual(['my_data', 'my_data_2', '_3d', 'int_', 'x']);
    });

    it('checks the number of containers of each skeleton', () => {
        const { d } = make([['a'], ['b']], [['f', 'Reduce'], ['g', 'Scan']], [['a', 'f', 'elwise'], ['b', 'f', 'elwise'], ['f', 'b', 'out']]);
        expect(problems(lower(d))).toEqual(expect.arrayContaining([
            expect.stringMatching(/f: 2 input containers, a Reduce has 1/),
            expect.stringMatching(/f: a Reduce has a scalar result, not an output container/),
            expect.stringMatching(/g: 0 input containers, a Scan has 1/),
            expect.stringMatching(/g: 0 output containers, a Scan has 1/),
        ]));
    });

    it('checks sizes: of a Map, of a MapOverlap with an edge mode, of a MapPairs', () => {
        let { d } = make([['a', 'Vector', [10]], ['b', 'Vector', [12]]], [['f', 'Map']], [['a', 'f', 'elwise'], ['f', 'b', 'out']]);
        expect(problems(lower(d))[0]).toMatch(/input a has 10 elements, the output b has 12/);
        ({ d } = make([['v', 'Vector', [4]], ['h', 'Vector', [5]], ['m', 'Matrix', [5, 4]]], [['p', 'MapPairs']], [['v', 'p', 'vertical'], ['h', 'p', 'horizontal'], ['p', 'm', 'out']]));
        expect(problems(lower(d))[0]).toMatch(/is a 4 x 5 Matrix/);
        d.containers[2].size = [4, 5];
        expect(lower(d).ok).toBe(true);
    });

    it('checks that a MapOverlap is on a container of its dimensions', () => {
        const { d } = make([['m', 'Matrix', [4, 4]], ['o', 'Matrix', [4, 4]]], [['s', 'MapOverlap']], [['m', 's', 'elwise'], ['s', 'o', 'out']]);
        expect(problems(lower(d))[0]).toMatch(/a 1D MapOverlap is on a Vector, not a Matrix/);
        d.skeletons[0].params.dims = 2;
        expect(lower(d).ok).toBe(true);
    });

    it('makes a Map with no input an indexed one, on the kind of its output', () => {
        const { d } = make([['m', 'Matrix', [4, 4]]], [['gen', 'Map']], [['gen', 'm', 'out']]);
        const r = lower(d);
        expect(r.ok).toBe(true);
        expect(r.program.calls[0]).toMatchObject({ indexed: true, indexType: 'Index2D' });
        expect(r.program.functions[0].params[0]).toMatchObject({ name: 'index', type: 'Index2D' });
    });

    it('puts the calls of a loop in it, and says that they are not consecutive', () => {
        const { d, id } = make([['a'], ['b'], ['c']], [['f', 'Map'], ['g', 'Map'], ['h', 'Map']],
            [['a', 'f', 'elwise'], ['f', 'b', 'out'], ['b', 'g', 'elwise'], ['g', 'c', 'out'], ['c', 'h', 'elwise'], ['h', 'a', 'out']], [['Loop']]);
        d.skeletons.find(s => s.id === id.f).parent = id.Loop;
        d.skeletons.find(s => s.id === id.h).parent = id.Loop;
        expect(problems(lower(d)).join()).toMatch(/Loop: its calls are not consecutive/);
        d.skeletons.find(s => s.id === id.g).parent = id.Loop;
        const r = lower(d);
        expect(r.ok).toBe(true);
        expect(r.program.steps).toHaveLength(1);
        expect(r.program.steps[0]).toMatchObject({ type: 'loop', count: 3 });
        expect(r.program.steps[0].steps.map(s => s.call.name)).toEqual(['f', 'g', 'h']);
    });
});

describe('the program in C++', () => {
    const cpp = emitCpp(lower(sampleDesign()).program).text;

    it('has the user functions, the containers, the skeletons and the calls', () => {
        expect(cpp).toContain('float smooth_f(skepu::Region1D<float> r)');
        expect(cpp).toContain('float scale_f(float a, float factor)');
        expect(cpp).toContain('float energy_reduce(float a, float b)');
        expect(cpp).toContain('skepu::Vector<float> signal(4096, "signal", 1.0f);');
        expect(cpp).toContain('auto smooth = skepu::MapOverlap(smooth_f);');
        expect(cpp).toContain('smooth.setOverlap(1);');
        expect(cpp).toContain('auto energy = skepu::MapReduce<1>(energy_map, energy_reduce);');
        expect(cpp).toContain('for (size_t iteration = 0; iteration < 10; ++iteration)');
        expect(cpp).toContain('SKEPU_TRACE_SCOPE("Smoothing");');
        expect(cpp).toContain('smooth(smoothed, signal);');
        expect(cpp).toContain('scale(scaled, smoothed, 0.5f);');
        expect(cpp).toContain('float energy_result = energy(scaled);');
    });

    it('has each skeleton on a line of its own, and every skeleton before the first call (what the precompiler takes)', () => {
        const lines = cpp.split('\n');
        const decl = lines.map((l, i) => /auto (?!spec)\w+ = skepu::/.test(l) ? i : -1).filter(i => i >= 0);
        const firstCall = lines.findIndex(l => /^\s+(for \(|scale\(|smooth\(|float energy_result)/.test(l));
        expect(decl.length).toBe(3);
        expect(Math.max(...decl)).toBeLessThan(firstCall);
        expect(lines.filter(l => /skepu::(Map|Reduce|Scan|MapOverlap|MapPairs|MapReduce)/.test(l) && l.includes('auto ')).every(l => l.trim().endsWith(';'))).toBe(true);
    });

    it('names, for each line, the nodes that it is for', () => {
        const { text, lines } = emitCpp(lower(sampleDesign()).program);
        const row = text.split('\n').findIndex(l => l.includes('smooth(smoothed, signal);'));
        expect(lines[row]).toEqual(['s1']);
        expect(lines[text.split('\n').findIndex(l => l.includes('"signal"'))]).toEqual(['c1']);
        expect(lines.length).toBe(text.replace(/\n$/, '').split('\n').length);
    });

    it('has the arguments in the order of the function: outputs, element-wise, proxies, uniforms', () => {
        const { d } = make([['x'], ['t'], ['y']], [['f', 'Map', { uniforms: [{ name: 'k', type: 'float', value: '2.5f' }] }]],
            [['x', 'f', 'elwise'], ['t', 'f', 'proxy'], ['f', 'y', 'out']]);
        const text = emitCpp(lower(d).program).text;
        expect(text).toContain('float f_f(float a, skepu::Vec<float> vec_0, float k)');
        expect(text).toContain('f(y, x, t, 2.5f);');
    });

    it('uses the body that the user wrote instead of the stub', () => {
        const { d, id } = make([['x'], ['y']], [['f', 'Map']], [['x', 'f', 'elwise'], ['f', 'y', 'out']]);
        d.skeletons[0].bodies = { _f: { cpp: 'return a * a;\n' } };
        const text = emitCpp(lower(d).program).text;
        expect(text).toContain('return a * a;');
        expect(text).not.toContain('TODO');
    });

    it('sets the backend of a skeleton and the mode of a scan', () => {
        const { d } = make([['x'], ['y']], [['s', 'Scan', { backend: 'opencl', params: { mode: 'Exclusive', start: '5.0f' } }]], [['x', 's', 'elwise'], ['s', 'y', 'out']]);
        const text = emitCpp(lower(d).program).text;
        expect(text).toContain('s.setScanMode(skepu::ScanMode::Exclusive);');
        expect(text).toContain('s.setStartValue(5.0f);');
        expect(text).toContain('s.setBackend(skepu::BackendSpec{"opencl"});');
    });
});

describe('the program in Python', () => {
    const py = emitPython(lower(sampleDesign()).program).text;

    it('is the same program with the calls of SkePY', () => {
        expect(py).toContain('def smooth_f(r):');
        expect(py).toContain('return r[0]');
        expect(py).toContain('signal = skepu.Vector(4096, 1.0)');
        expect(py).toContain('smooth = skepu.MapOverlap(smooth_f)');
        expect(py).toContain('smooth.set_overlap(1)');
        expect(py).toContain('energy = skepu.MapReduce(energy_map, energy_reduce)');
        expect(py).toContain('for iteration in range(10):');
        expect(py).toContain('with skepu.scope("Smoothing"):');
        expect(py).toContain('scale(scaled, inargs=smoothed, uniforms=0.5)');
        expect(py).toContain('energy_result = energy(inargs=scaled)');
        expect(py.trimEnd().endsWith('main()')).toBe(true);
    });

    it('calls a MapPairs with first and second, and a Map with an index with ix', () => {
        const { d } = make([['v', 'Vector', [4]], ['h', 'Vector', [5]], ['m', 'Matrix', [4, 5]], ['g', 'Vector', [8]]], [['p', 'MapPairs'], ['gen', 'Map']],
            [['v', 'p', 'vertical'], ['h', 'p', 'horizontal'], ['p', 'm', 'out'], ['gen', 'g', 'out']]);
        d.containers.push(defaultPrng('c900', 'gen_stream'));
        d.edges.push({ id: 'e900', from: 'c900', to: d.skeletons[1].id, role: 'random' });
        const text = emitPython(lower(d).program).text;
        expect(text).toContain('p(m, first=v, second=h)');
        expect(text).toContain('gen(g, ix=skepu.Index1D, rand=skepu.Random(1))');
        expect(text).toContain('gen_stream = skepu.PRNG()');
        expect(text).toContain('gen.set_prng(gen_stream)');
    });
});

describe('the round trip: the trace of the program that was made from a design', () => {
    // a trace of the program of allSkeletonsDesign(), built with the precompiler and run on OpenMP (designer/roundtrip.mjs makes one)
    const events = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), '../fixtures/designer_all_skeletons_openmp.json'), 'utf8'));
    const program = lower(allSkeletonsDesign()).program;

    it('has the calls of the design, with their containers and loops', () => {
        expect(compareTrace(program, events)).toEqual([]);
    });

    it('differs from a design that has another call', () => {
        const d = allSkeletonsDesign();
        d.skeletons.find(s => s.label === 'mix').label = 'blend';
        d.loops[0].count = 5;
        const found = compareTrace(lower(d).program, events);
        expect(found.join('\n')).toMatch(/call 3 \(blend\): the trace has mix/);
        expect(found.join('\n')).toMatch(/the trace has 18 calls, the design 20/);
    });
});

describe('arranging a design', () => {
    it('keeps the calls of a loop together, where the first of them is', () => {
        const { d, id } = make([['a']], [['f', 'Map'], ['g', 'Map'], ['h', 'Map'], ['k', 'Map']], [], [['Loop']]);
        for (const s of ['f', 'h']) d.skeletons.find(x => x.id === id[s]).parent = id.Loop;
        // by place: f, g, h, k with g between the calls of the loop
        keepLoopsTogether(d);
        const order = d.skeletons.slice().sort((a, b) => d.positions[a.id].y - d.positions[b.id].y).map(s => s.label);
        expect(order).toEqual(['f', 'h', 'g', 'k']);
        expect(Object.values(d.positions).map(p => p.y).sort((a, b) => a - b)).toEqual([0, 100, 200, 300]);   // the same places
    });
});

describe('the format of what a skeleton writes', () => {
    const vec = (n, type = 'float') => ({ kind: 'Vector', type, size: [n] });
    const mat = (r, c) => ({ kind: 'Matrix', type: 'float', size: [r, c] });

    it('is inferred from the pattern and the inputs', () => {
        expect(inferOutput({ pattern: 'Map' }, { elwise: [mat(4, 8), vec(32)] })).toEqual(mat(4, 8));
        expect(inferOutput({ pattern: 'Scan' }, { elwise: [vec(10, 'int')] })).toEqual(vec(10, 'int'));
        expect(inferOutput({ pattern: 'MapOverlap', params: { dims: 2 } }, { elwise: [mat(5, 6)] })).toEqual(mat(5, 6));
        expect(inferOutput({ pattern: 'MapPairs' }, { vertical: [vec(7)], horizontal: [vec(9)] })).toEqual(mat(7, 9));
    });

    it('is not known without inputs, and a reduction writes a scalar', () => {
        expect(inferOutput({ pattern: 'Map' }, {})).toBeNull();
        expect(inferOutput({ pattern: 'MapPairs' }, { vertical: [vec(7)] })).toBeNull();
        expect(inferOutput({ pattern: 'Reduce' }, { elwise: [vec(7)] })).toBeNull();
    });

    it('fills in the containers that are auto, through a chain of calls, and leaves the others', () => {
        const { d, id } = make([['a', 'Vector', [100]], ['b', 'Vector', [3]], ['c', 'Vector', [3]], ['m', 'Matrix', [2, 2]]],
            [['f', 'Map'], ['g', 'Scan'], ['p', 'MapPairs']],
            [['a', 'f', 'elwise'], ['f', 'b', 'out'], ['b', 'g', 'elwise'], ['g', 'c', 'out'], ['b', 'p', 'vertical'], ['c', 'p', 'horizontal'], ['p', 'm', 'out']]);
        for (const label of ['b', 'c', 'm']) d.containers.find(x => x.id === id[label]).auto = true;
        const sizes = Object.fromEntries(effectiveContainers(d).map(c => [c.label, c.size.join('x')]));
        expect(sizes).toEqual({ a: '100', b: '100', c: '100', m: '100x100' });
        expect(d.containers.find(x => x.id === id.b).size).toEqual([3]);          // the design itself is as it was
        expect(lower(d).errors).toEqual([]);                                      // and the sizes that are inferred agree with each other
        expect(emitCpp(lower(d).program).text).toMatch(/skepu::Matrix<float> m\(100, 100,/);
    });

    it('keeps what a container has when its own call reads it', () => {
        const { d, id } = make([['a', 'Vector', [8]]], [['f', 'Map']], [['a', 'f', 'elwise'], ['f', 'a', 'out']]);
        d.containers[0].auto = true;
        expect(effectiveContainers(d)[0].size).toEqual([8]);
    });
});

describe('the lines of a program that a trace has nodes for', () => {
    it('are the calls and the loops, with how many times they run and in what order', () => {
        const program = lower(sampleDesign()).program;
        for (const emit of [emitCpp, emitPython]) {
            const { sites, text } = emit(program);
            expect(sites.map(s => [s.type, s.id, s.count])).toEqual([['region', 'l1', 10], ['skeleton_call', 's1', 10], ['skeleton_call', 's2', 1], ['skeleton_call', 's3', 1]]);
            expect(sites.map(s => s.order)).toEqual([0, 1, 2, 3]);
            const lines = text.split('\n');
            expect(lines[sites[1].line - 1]).toMatch(/smooth\(/);
            expect(lines[sites[0].line - 1]).toMatch(/SKEPU_TRACE_SCOPE|skepu\.scope/);
        }
    });

    it('multiply the iterations of loops in loops', () => {
        const { d, id } = make([['a'], ['b']], [['f', 'Map']], [['a', 'f', 'elwise'], ['f', 'b', 'out']], [['Outer'], ['Inner']]);
        d.loops.find(l => l.id === id.Inner).parent = id.Outer;
        d.skeletons[0].parent = id.Inner;
        const sites = emitCpp(lower(d).program).sites;
        expect(sites.map(s => s.count)).toEqual([3, 9, 9]);
    });
});

describe('in place', () => {
    const warnings = (pattern, role = 'elwise') => {
        const { d } = make([['a'], ['a2', 'Vector', undefined, 'a']], [['f', pattern]], [['a', 'f', role], ['f', 'a2', 'out']]);        // (an update of what it reads)
        return lower(d).warnings.map(w => w.msg).filter(m => /place|proxy/.test(m));
    };
    it('is fine for a Map', () => expect(warnings('Map')).toEqual([]));
    it('is a warning through a proxy of a Map, and for the patterns that are not element by element', () => {
        expect(warnings('Map', 'proxy')[0]).toMatch(/proxy/);
        expect(warnings('MapOverlap')[0]).toMatch(/not element by element/);
    });
});

describe('transfers between backends', () => {
    // a -> f -> b -> g -> c -> h, with the backends given
    const run = (backends, loops = []) => {
        const { d, id } = make([['a'], ['b'], ['c'], ['e']], [['f', 'Map'], ['g', 'Map'], ['h', 'Map']],
            [['a', 'f', 'elwise'], ['f', 'b', 'out'], ['b', 'g', 'elwise'], ['g', 'c', 'out'], ['c', 'h', 'elwise'], ['h', 'e', 'out']], loops);
        ['f', 'g', 'h'].forEach((s, i) => { d.skeletons.find(x => x.id === id[s]).backend = backends[i]; });
        return { d, id, found: transfers(lower(d).program).map(t => [t.container, t.consumer, t.direction]) };
    };

    it('are between a host backend and a device backend, not between two host backends', () => {
        const { id, found } = run(['cpu', 'openmp', 'opencl']);
        expect(found).toEqual([[id.c, id.h, 'host-to-device']]);
    });

    it('go to the device and back', () => {
        const { id, found } = run(['opencl', 'cpu', 'opencl']);
        expect(found).toEqual([[id.a, id.f, 'host-to-device'], [id.b, id.g, 'device-to-host'], [id.c, id.h, 'host-to-device']]);
    });

    it('are not for calls that have no backend of their own', () => {
        expect(run(['', 'opencl', '']).found).toEqual([]);
        expect(run(['opencl', '', 'cpu']).found.map(t => t[2])).toEqual(['host-to-device']);
    });

    it('are found for what moves from one iteration of a loop to the next', () => {
        const { d, id } = make([['a'], ['a2', 'Vector', undefined, 'a'], ['a3', 'Vector', undefined, 'a']], [['f', 'Map', { backend: 'opencl' }], ['g', 'Map', { backend: 'cpu' }]],
            [['a', 'f', 'elwise'], ['f', 'a2', 'out'], ['a2', 'g', 'elwise'], ['g', 'a3', 'out']], [['Loop']]);
        for (const s of d.skeletons) s.parent = id.Loop;
        expect(transfers(lower(d).program).map(t => t[2] ?? t.direction).sort()).toEqual(['device-to-host', 'host-to-device']);
    });
});

describe('a transfer is the source of every call that reads its copy', () => {
    // gen (OpenCL) writes a; f (CPU) and dump (an external: on the host) read it; h (OpenCL) reads it too, where it was written
    const design = () => make([['a'], ['b'], ['c'], ['d']], [['gen', 'Map', { backend: 'opencl' }], ['f', 'Map', { backend: 'cpu' }], ['dump', 'External'], ['h', 'Map', { backend: 'opencl' }]],
        [['gen', 'a', 'out'], ['a', 'f', 'elwise'], ['f', 'b', 'out'], ['a', 'dump', 'read'], ['dump', 'c', 'out'], ['a', 'h', 'elwise'], ['h', 'd', 'out']]);

    it('has one transfer to the host for the calls on the host (the external too), and the device call reads the data itself', () => {
        const { d, id } = design();
        const found = transfers(lower(d).program);
        expect(found).toEqual([{ container: id.a, consumer: id.f, direction: 'device-to-host', consumers: [id.f, id.dump] }]);
    });

    it('has a new transfer when the data was written again, and one in the trace for the two readers', () => {
        const { d, id } = make([['a'], ['a2', 'Vector', undefined, 'a'], ['b'], ['c']], [['gen', 'Map', { backend: 'opencl' }], ['f', 'Map', { backend: 'cpu' }], ['gen2', 'Map', { backend: 'opencl' }], ['dump', 'External']],
            [['gen', 'a', 'out'], ['a', 'f', 'elwise'], ['f', 'b', 'out'], ['a', 'gen2', 'elwise'], ['gen2', 'a2', 'out'], ['a2', 'dump', 'read'], ['dump', 'c', 'out']]);
        const r = lower(d);
        expect(transfers(r.program).map(t => [t.container, t.consumers])).toEqual([[id.a, [id.f]], [id.a2, [id.dump]]]);
        const cpp = emitCpp(r.program);
        const trace = synthesizeTrace(r.program, { dir: '/x', file: 'a.cpp', lines: cpp.lines, sites: cpp.sites });
        expect(trace.events.filter(e => e.type === 'transfer').map(e => e.direction)).toEqual(['device-to-host', 'device-to-host']);
    });
});

describe('random streams', () => {
    const withStream = (patch = {}) => {
        const { d, id } = make([['g', 'Vector', [8]], ['g2', 'Vector', [8]]], [['gen', 'Map'], ['gen2', 'Map']], [['gen', 'g', 'out'], ['gen2', 'g2', 'out']]);
        d.containers.push({ ...defaultPrng('p1', 'noise'), ...patch });
        d.edges.push({ id: 'er1', from: 'p1', to: id.gen, role: 'random' }, { id: 'er2', from: id.gen, to: id.gen2, role: 'random' });      // a chain: the stream goes on from gen
        return { d, id };
    };

    it('are shared by the Maps that are connected to them: a chain, one edge out of the stream and of each Map', () => {
        const { d } = withStream({ init: '7' });
        expect(lower(d).errors).toEqual([]);
        const program = lower(d).program;
        expect(program.calls.map(c => c.randomAfter)).toEqual([null, program.calls[0].skeleton.id]);
        expect(program.calls.map(c => c.random.name)).toEqual(['noise', 'noise']);
        const cpp = emitCpp(program).text;
        expect(cpp).toContain('skepu::PRNG noise(7, "noise");');
        const { d: plain } = withStream();
        expect(emitCpp(lower(plain).program).text).toContain('skepu::PRNG noise("noise");');          // (no seed: the default one, and the name)
        expect(emitCpp(lower(plain).program).text).not.toContain('SKEPU_RAND_DEFAULT_SEED');
        expect(cpp).toContain('gen.setPRNG(noise);');
        expect(cpp).toContain('gen2.setPRNG(noise);');
        expect(emitPython(program).text).toContain('noise = skepu.PRNG(7)');
        expect(cpp).toMatch(/Random<1> &random/);     // the function takes the stream
    });

    it('have one edge out, and so has each Map in the chain, and a Map draws from one stream', () => {
        const { d, id } = withStream();
        d.edges.find(e => e.id === 'er2').from = 'p1';                                    // both Maps from the stream itself
        expect(problems(lower(d)).join()).toMatch(/noise: 2 random edges out|2 random edges out/);
        d.edges.find(e => e.id === 'er2').from = id.gen;
        d.edges.push({ id: 'er3', from: 'p1', to: id.gen2, role: 'random' });             // gen2 draws from two
        expect(problems(lower(d)).join()).toMatch(/gen2: 2 random edges in/);
    });

    it('go on in the order of the chain, whatever the places of the Maps', () => {
        const { d, id } = withStream();
        d.positions[id.gen].y = 500; d.positions[id.gen2].y = 100;                         // gen2 is above gen by place: the chain decides
        expect(lower(d).program.calls.map(c => c.name)).toEqual(['gen', 'gen2']);
    });

    it('say when a Map is in a chain that has no stream, or a chain is a cycle', () => {
        const { d, id } = withStream();
        d.edges = d.edges.filter(e => e.id !== 'er1');                                    // gen2 follows gen, which has no stream
        expect(problems(lower(d)).join()).toMatch(/gen2: draws random numbers after gen, which has no random stream/);
        d.edges.push({ id: 'er4', from: id.gen2, to: id.gen, role: 'random' });
        expect(problems(lower(d)).join()).toMatch(/cycle/);
    });

    it('are drawn from by the skeletons that SkePU gives a Random parameter: Map, MapReduce, MapOverlap and MapPairs, after the index', () => {
        const { d, id } = make([['v', 'Vector', [4]], ['h', 'Vector', [5]], ['m', 'Matrix', [4, 5]], ['x', 'Vector', [8]], ['y', 'Vector', [8]], ['z', 'Vector', [8]]],
            [['pairs', 'MapPairs'], ['stencil', 'MapOverlap'], ['total', 'MapReduce'], ['gen', 'Map']],
            [['v', 'pairs', 'vertical'], ['h', 'pairs', 'horizontal'], ['pairs', 'm', 'out'], ['x', 'stencil', 'elwise'], ['stencil', 'y', 'out'], ['y', 'total', 'elwise'], ['gen', 'z', 'out']]);
        d.containers.push(defaultPrng('p1', 'noise'));
        let from = 'p1';
        for (const name of ['pairs', 'stencil', 'total', 'gen']) { d.edges.push({ id: `r_${name}`, from, to: id[name], role: 'random' }); from = id[name]; }
        const r = lower(d);
        expect(r.errors).toEqual([]);
        expect(r.program.calls.map(c => [c.name, c.random?.name])).toEqual([['pairs', 'noise'], ['stencil', 'noise'], ['total', 'noise'], ['gen', 'noise']]);
        const cpp = emitCpp(r.program).text, py = emitPython(r.program).text;
        expect(cpp).toContain('float pairs_f(skepu::Random<1> &random, float v, float h)');
        expect(cpp).toContain('float stencil_f(skepu::Random<1> &random, skepu::Region1D<float> r)');
        expect(cpp).toContain('float total_map(skepu::Random<1> &random, float a)');
        expect(cpp).toContain('total.setPRNG(noise);');
        expect(py).toContain('def stencil_f(random, r):');
        expect(py).toContain('stencil(y, x, rand=skepu.Random(1))');
        expect(py).toContain('stencil.set_prng(noise)');
        d.skeletons.find(s => s.id === id.total).pattern = 'Reduce';                      // a Reduce has no Random parameter
        expect(problems(lower(d)).join()).toMatch(/total: a Reduce does not draw random numbers/);
    });

    it('can only be the random numbers of a Map', () => {
        const { d, id } = withStream();
        d.edges.find(e => e.id === 'er1').role = 'elwise';
        expect(lower(d).errors.map(e => e.msg).join()).toMatch(/is a random stream: connect it as the random numbers/);
        d.edges.find(e => e.id === 'er1').role = 'random';
        d.skeletons.find(s => s.id === id.gen).pattern = 'Scan';
        expect(lower(d).errors.map(e => e.msg).join()).toMatch(/does not draw random numbers/);
    });

    it('are not written by a call', () => {
        const { d, id } = withStream();
        d.edges.push({ id: 'bad', from: id.gen, to: 'p1', role: 'out' });
        expect(lower(d).errors.map(e => e.msg).join()).toMatch(/random stream: a call does not write it/);
    });
});

describe('external blocks', () => {
    const design = () => {
        const { d, id } = make([['a', 'Vector', [8]], ['b', 'Vector', [8]], ['c', 'Vector', [8]], ['a2', 'Vector', [8], 'a']], [['f', 'Map'], ['dump', 'External', { backend: '' }], ['g', 'Map']],
            [['a', 'f', 'elwise'], ['f', 'b', 'out'], ['b', 'dump', 'read'], ['dump', 'c', 'out'], ['c', 'g', 'elwise'], ['g', 'a2', 'out']]);
        return { d, id };
    };

    it('are calls with host code, the data that they read and the data that they write', () => {
        const { d } = design();
        const r = lower(d);
        expect(r.errors).toEqual([]);
        const cpp = emitCpp(r.program).text;
        expect(cpp).toMatch(/skepu::external\("dump", skepu::read\(b\), \[&\]\n\t\{\n\t\t\/\/ TODO: host code\n\t\}, skepu::write\(c\)\);/);
        expect(emitPython(r.program).text).toMatch(/with skepu\.external\(read=b, write=c, label="dump"\):\n {8}pass {2}# TODO: host code/);
        expect(r.program.functions.filter(f => !f.host).map(f => f.name)).toEqual(['f_f', 'g_f']);        // no function of its own
    });

    it('have the body that the user gives, for each language', () => {
        const { d, id } = design();
        d.skeletons.find(s => s.id === id.dump).bodies = { _host: { cpp: 'std::cout << b[0];', py: 'print(b[0])' } };
        expect(emitCpp(lower(d).program).text).toContain('\t\tstd::cout << b[0];');
        expect(emitPython(lower(d).program).text).toContain('        print(b[0])');
    });

    it('may have no reads and several writes, and are a site of the program like a call', () => {
        const { d, id } = design();
        d.edges = d.edges.filter(e => e.role !== 'read');
        d.containers.push({ ...defaultContainer('x1', 'x'), size: [8] });
        d.edges.push({ id: 'w2', from: id.dump, to: 'x1', role: 'out' });
        const r = lower(d);
        expect(r.errors).toEqual([]);
        const { text, sites } = emitCpp(r.program);
        expect(text).toContain('skepu::external("dump", [&]');
        expect(text).toContain('skepu::write(c, x)');
        expect(sites.find(s => s.id === id.dump).type).toBe('external');
    });

    it('run on the host: a read of what a device call wrote is a transfer', () => {
        const { d, id } = design();
        d.skeletons.find(s => s.id === id.f).backend = 'opencl';
        expect(transfers(lower(d).program).map(t => [t.container, t.consumer, t.direction])).toEqual([[id.a, id.f, 'host-to-device'], [id.b, id.dump, 'device-to-host']]);
    });
});

describe('the trace that a design could have', () => {
    const traceOf = d => {
        const r = lower(d), cpp = emitCpp(r.program);
        return { r, cpp, ...synthesizeTrace(r.program, { dir: '/designer', file: 'd.cpp', lines: cpp.lines, sites: cpp.sites }) };
    };

    it('has the calls of the design, which the check of the round trip finds, and SkeVU can read it', () => {
        for (const d of [sampleDesign(), allSkeletonsDesign()]) {
            const { r, events } = traceOf(d);
            expect(compareTrace(r.program, events)).toEqual([]);
            expect(DirectedGraph.fromEvents(events)).toBeTruthy();
        }
    });

    it('has the lines of the program, and the data and the parameters of the calls', () => {
        const { cpp, events, ids } = traceOf(sampleDesign());
        const calls = events.filter(e => e.type === 'skeleton_call');
        expect(calls.map(c => [c.label, c.pattern, c.backend])).toEqual([...Array(10).fill(['smooth', 'MapOverlap 1D', 'CPU']), ['scale', 'Map', 'CPU'], ['energy', 'MapReduce', 'CPU']]);
        expect(calls[0].elements).toEqual([4096]);
        expect(calls[0].parameters).toEqual({ overlap: [1] });
        const line = cpp.sites.find(s => s.id === 's1').line;
        expect(calls[0].line).toBe(line);
        expect(events.filter(e => e.type === 'region').map(e => [e.label, e.region_depth])).toEqual(Array(10).fill(['Smoothing', 1]));
        expect(calls[0].region).toBe('Smoothing');
        expect(events.filter(e => e.type === 'allocation').map(e => e.label)).toEqual(['signal', 'smoothed', 'scaled']);
        expect(ids[events.indexOf(calls[0])][0]).toBe('s1');
    });

    it('has the transfers of a call on a device, the time of each, and the scalars of a reduction that is printed', () => {
        const d = sampleDesign();
        d.skeletons.find(s => s.id === 's2').backend = 'opencl';
        const { events } = traceOf(d);
        const moves = events.filter(e => e.type === 'transfer');
        expect(moves.map(e => [e.label, e.direction, e.elements])).toEqual([['smoothed', 'host-to-device', 4096], ['scaled', 'device-to-host', 4096]]);
        for (const e of events.filter(e => e.type === 'skeleton_call' || e.type === 'transfer')) expect(e.end).toBeGreaterThan(e.start);
        const energy = events.find(e => e.label === 'energy');
        expect(energy.outputs).toHaveLength(1);
        expect(events.find(e => e.label === 'I/O').proxy_inputs).toEqual(energy.outputs);
    });

    it('is cut to the first iterations of long loops, and says how many', () => {
        const d = sampleDesign();
        d.loops[0].count = 100000;
        const { events, truncated } = traceOf(d);
        expect(truncated).toBeGreaterThan(0);
        expect(events.filter(e => e.type === 'region').length).toBe(truncated);
        expect(events.length).toBeLessThan(2000);
    });
});

describe('I/O blocks', () => {
    // a -> total (Reduce) ; a -> show (I/O: prints a, and the result of total)
    const design = () => make([['a', 'Vector', [8]], ['b', 'Vector', [8]]], [['total', 'Reduce'], ['f', 'Map', { backend: 'opencl' }], ['show', 'IO']],
        [['a', 'total', 'elwise'], ['a', 'f', 'elwise'], ['f', 'b', 'out'], ['b', 'show', 'read'], ['total', 'show', 'scalar']]);

    it('print the data that they read and the results that are connected to them: skepu::io::cout, and print in Python', () => {
        const { d } = design();
        const r = lower(d);
        expect(r.errors).toEqual([]);
        const show = r.program.calls.find(c => c.name === 'show');
        expect(show).toMatchObject({ host: true, io: true });
        expect(show.reads.map(c => c.name)).toEqual(['b']);
        expect(show.scalars.map(c => c.name)).toEqual(['total']);
        const cpp = emitCpp(r.program).text, py = emitPython(r.program).text;
        expect(cpp).toContain('skepu::io::cout << b << "\\n";');
        expect(cpp).toContain('skepu::io::cout << "total: " << total_result << "\\n";');
        expect(cpp.match(/skepu::io::cout << "total: "/g)).toHaveLength(1);          // (printed by the block, not by the call too)
        expect(cpp).not.toContain('skepu::external("show"');
        expect(py).toContain('print(b)');
        expect(py.match(/print\("total:"/g)).toHaveLength(1);
        expect(r.program.functions.filter(f => f.call === show.skeleton || f.call?.name === 'show')).toEqual([]);          // no user function
    });

    it('leave the printing to the call when no block prints its result, and come after the call that makes the result', () => {
        const { d, id } = design();
        d.edges = d.edges.filter(e => e.role !== 'scalar');
        expect(emitCpp(lower(d).program).text).toContain('skepu::io::cout << "total: " << total_result');          // (by the call)
        d.edges.push({ id: 'sc', from: id.total, to: id.show, role: 'scalar' });
        d.positions[id.show].y = -500; d.positions[id.total].y = 900;
        const r = lower(d);
        expect(r.program.calls.map(c => c.name).indexOf('show')).toBeGreaterThan(r.program.calls.map(c => c.name).indexOf('total'));      // by the data flow, not the place
    });

    it('are a trace event for each container that is printed, with a transfer to the host first (an I/O block runs there)', () => {
        const { d, id } = design();
        const r = lower(d), cpp = emitCpp(r.program);
        const { events, ids } = synthesizeTrace(r.program, { dir: '/x', file: 'a.cpp', lines: cpp.lines, sites: cpp.sites });
        const kinds = events.map(e => e.type === 'external' ? `I/O ${e.proxy_inputs.length}` : e.type === 'transfer' ? `transfer ${e.label}` : e.type === 'skeleton_call' ? e.label : '').filter(Boolean);
        expect(kinds).toEqual(['total', 'transfer a', 'f', 'transfer b', 'I/O 1', 'I/O 1']);          // b (a container), then the result of total
        expect(events.filter(e => e.label === 'I/O').map(e => e.line)).toEqual([cpp.sites.find(s => s.id === id.show).line, cpp.sites.find(s => s.id === id.show).line + 1]);
        expect(ids[events.findIndex(e => e.label === 'I/O')][0]).toBe(id.show);
        expect(compareTrace(r.program, events)).toEqual([]);
        expect(compareTrace(r.program, events.filter(e => e.label !== 'I/O'))).toEqual([expect.stringMatching(/the I\/O blocks print nothing, the design b/)]);
    });

    it('can only be connected from a Reduce or a MapReduce (its result) or from data', () => {
        const { d, id } = design();
        d.edges.push({ id: 'bad', from: id.f, to: id.show, role: 'scalar' });
        expect(problems(lower(d)).join()).toMatch(/a scalar edge goes from a Reduce or MapReduce \(its result\) to an I\/O block/);
        d.edges.pop();
        d.edges.push({ id: 'bad2', from: id.total, to: id.f, role: 'scalar' });
        expect(problems(lower(d)).join()).toMatch(/a scalar edge goes/);
    });

    it('are made by Generate, with their results, and the designs stay valid and in the trace', () => {
        let blocks = 0, results = 0;
        for (let k = 0; k < 20; k++) {
            const { design: d } = randomDesign({ size: 'large', seed: `io${k}` });
            blocks += d.skeletons.filter(s => s.pattern === 'IO').length;
            results += d.edges.filter(e => e.role === 'scalar').length;
            const r = lower(d);
            expect([r.errors.map(e => e.msg), r.warnings.map(e => e.msg)]).toEqual([[], []]);
        }
        expect(blocks).toBeGreaterThan(5);
        expect(results).toBeGreaterThan(0);
    });
});

describe('MapPairsReduce', () => {
    const design = (mode = 'RowWise', out = 6) => make([['v', 'Vector', [6]], ['h', 'Vector', [4]], ['r', 'Vector', [out]]], [['rows', 'MapPairsReduce', { params: { mode, start: '1.0f' } }]],
        [['v', 'rows', 'vertical'], ['h', 'rows', 'horizontal'], ['rows', 'r', 'out']]);

    it('has a map and a reduction, the inputs of a MapPairs and a vector as output, in C++ and in Python', () => {
        const { d } = design();
        const r = lower(d);
        expect(r.errors).toEqual([]);
        expect(r.program.functions.map(f => [f.name, f.params.map(p => p.name)])).toEqual([['rows_map', ['v', 'h']], ['rows_reduce', ['a', 'b']]]);
        const cpp = emitCpp(r.program).text, py = emitPython(r.program).text;
        expect(cpp).toContain('auto rows = skepu::MapPairsReduce<1, 1>(rows_map, rows_reduce);');
        expect(cpp).toContain('rows.setReduceMode(skepu::ReduceMode::RowWise);');
        expect(cpp).toContain('rows.setStartValue(1.0f);');
        expect(cpp).toContain('rows(r, v, h);');
        expect(py).toContain('rows = skepu.MapPairsReduce(rows_map, rows_reduce)');
        expect(py).toContain('rows.set_reduce_mode(skepu.REDUCEMODE_RowWise)');
        expect(py).toContain('rows(r, first=v, second=h)');
    });

    it('is reduced along the rows or the columns: the output is as long as the vertical or the horizontal vector', () => {
        expect(lower(design('RowWise', 6).d).errors).toEqual([]);
        expect(problems(lower(design('RowWise', 4).d)).join()).toMatch(/reduced along the rows, the output is a Vector of the size of v \(6\)/);
        expect(lower(design('ColWise', 4).d).errors).toEqual([]);
        expect(emitCpp(lower(design('ColWise', 4).d).program).text).toContain('rows.setReduceMode(skepu::ReduceMode::ColWise);');
        const { d, id } = design('ColWise', 4);
        d.containers.find(c => c.id === id.r).auto = true;                       // the format is inferred from the mode
        expect(effectiveContainers(d).find(c => c.id === id.r).size).toEqual([4]);
        d.skeletons[0].params.mode = 'RowWise';
        expect(effectiveContainers(d).find(c => c.id === id.r).size).toEqual([6]);
    });

    it('is in the trace as MapPairsReduce, with its elements, inputs, mode and start value', () => {
        const { d, id } = design('ColWise', 4);
        const r = lower(d), cpp = emitCpp(r.program);
        const { events } = synthesizeTrace(r.program, { dir: '/x', file: 'a.cpp', lines: cpp.lines, sites: cpp.sites });
        const call = events.find(e => e.type === 'skeleton_call');
        expect([call.pattern, call.elements, call.parameters]).toEqual(['MapPairsReduce', [4, 6], { mode: 'ColWise', start_value: 1 }]);
        expect(call.elwise_inputs).toHaveLength(2);
        expect(compareTrace(r.program, events)).toEqual([]);
    });

    it('can draw random numbers, and a call before it is connected with the vertical and horizontal edges by default', () => {
        const { d, id } = design();
        d.containers.push(defaultPrng('p1', 'noise'));
        d.edges.push({ id: 'er', from: 'p1', to: id.rows, role: 'random' });
        const r = lower(d);
        expect(r.errors).toEqual([]);
        expect(emitCpp(r.program).text).toContain('float rows_map(skepu::Random<1> &random, float v, float h)');
    });
});

describe('a MapOverlap with no edge mode', () => {
    const design = (out, overlap = [2, 1, 1, 1], dims = 1) => make([['a', 'Vector', [20]], ['o', 'Vector', [out]]], [['m', 'MapOverlap', { params: { dims, overlap, edge: 'None', pad: '0' } }]],
        [['a', 'm', 'elwise'], ['m', 'o', 'out']]);

    it('has an output that is smaller than the input by the overlap on each side (SkePU checks it)', () => {
        expect(lower(design(16).d).errors).toEqual([]);
        expect(problems(lower(design(20).d)).join()).toMatch(/with the edge mode None the output is smaller than the input by the overlap on each side: 16 for a of 20, not 20/);
        const { d } = design(16, [11]);
        expect(problems(lower(d)).join()).toMatch(/more than a \(20\) has room for/);
    });

    it('is inferred for data that the call writes, and has an output for each dimension of a Matrix', () => {
        const { d, id } = design(1);
        d.containers.find(c => c.id === id.o).auto = true;
        expect(effectiveContainers(d).find(c => c.id === id.o).size).toEqual([16]);
        const m = make([['g', 'Matrix', [10, 12]], ['h', 'Matrix', [8, 12]]], [['s', 'MapOverlap', { params: { dims: 2, overlap: [1, 0], edge: 'None', pad: '0' } }]], [['g', 's', 'elwise'], ['s', 'h', 'out']]);
        expect(lower(m.d).errors).toEqual([]);
        m.d.skeletons[0].params.edge = 'Duplicate';
        expect(problems(lower(m.d)).join()).toMatch(/with the edge mode Duplicate the output has the size of the input/);
    });
});

describe('3D and 4D tensors', () => {
    const design = () => {
        const { d, id } = make([['t', 'Tensor3', [4, 5, 6]], ['u', 'Tensor3', [4, 5, 6]], ['q', 'Tensor4', [3, 3, 3, 3]], ['w', 'Tensor4', [3, 3, 3, 3]], ['g', 'Tensor3', [4, 5, 6]], ['p', 'Tensor3', [4, 5, 6]]],
            [['gen', 'Map'], ['blur', 'MapOverlap', { params: { dims: 3, overlap: [1, 2, 1, 1], edge: 'Duplicate', pad: '0' } }], ['blur4', 'MapOverlap', { params: { dims: 4, overlap: [1, 1, 1, 2], edge: 'Duplicate', pad: '0' } }],
             ['total', 'Reduce'], ['look', 'Map']],
            [['gen', 't', 'out'], ['t', 'blur', 'elwise'], ['blur', 'u', 'out'], ['q', 'blur4', 'elwise'], ['blur4', 'w', 'out'], ['u', 'total', 'elwise'], ['g', 'look', 'elwise'], ['t', 'look', 'proxy'], ['look', 'p', 'out']]);
        return { d, id };
    };

    it('are data like vectors and matrices: declared with their sizes, in C++ and in Python', () => {
        const { d } = design();
        const r = lower(d);
        expect(r.errors).toEqual([]);
        const cpp = emitCpp(r.program).text, py = emitPython(r.program).text;
        expect(cpp).toContain('skepu::Tensor3<float> t(4, 5, 6, "t", 0);');
        expect(cpp).toContain('skepu::Tensor4<float> q(3, 3, 3, 3, "q", 0);');
        expect(py).toContain('t = skepu.Tensor3(4, 5, 6, 0)');
        expect(py).toContain('q = skepu.Tensor4(3, 3, 3, 3, 0)');
    });

    it('have a MapOverlap of 3 or 4 dimensions, with a radius for each, and a Region3D / Region4D', () => {
        const { d } = design();
        const r = lower(d), cpp = emitCpp(r.program).text, py = emitPython(r.program).text;
        expect(cpp).toContain('float blur_f(skepu::Region3D<float> r)');
        expect(cpp).toContain('float blur4_f(skepu::Region4D<float> r)');
        expect(cpp).toContain('return r(0, 0, 0, 0);');
        expect(cpp).toContain('blur.setOverlap(1, 2, 1);');
        expect(cpp).toContain('blur4.setOverlap(1, 1, 1, 2);');
        expect(py).toContain('blur.set_overlap(1, 2, 1)');
        expect(py).toContain('return r[0, 0, 0]');
    });

    it('are proxies Ten3 / Ten4, and an index is Index3D / Index4D', () => {
        const { d, id } = design();
        const r = lower(d), cpp = emitCpp(r.program).text;
        expect(cpp).toContain('skepu::Ten3<float> ten3_0');
        const gen = r.program.calls.find(c => c.name === 'gen');
        expect(gen).toMatchObject({ indexed: true, indexType: 'Index3D' });
        d.containers.find(c => c.id === id.t).kind = 'Tensor4'; d.containers.find(c => c.id === id.t).size = [2, 2, 2, 2];
        expect(lower(d).program.calls.find(c => c.name === 'gen').indexType).toBe('Index4D');
    });

    it('say when a MapOverlap is on a container of another rank, and check sizes', () => {
        const { d, id } = design();
        d.skeletons.find(s => s.id === id.blur).params.dims = 2;
        expect(problems(lower(d)).join()).toMatch(/blur: a 2D MapOverlap is on a Matrix, not a Tensor3/);
        d.skeletons.find(s => s.id === id.blur).params.dims = 3;
        d.containers.find(c => c.id === id.u).size = [4, 5, 7];
        expect(problems(lower(d)).join()).toMatch(/with the edge mode Duplicate the output has the size of the input|has 120 elements/);
    });

    it('are in the trace with the pattern and the elements of their dimensions', () => {
        const { d } = design();
        const r = lower(d), cpp = emitCpp(r.program);
        const { events } = synthesizeTrace(r.program, { dir: '/x', file: 'a.cpp', lines: cpp.lines, sites: cpp.sites });
        const blur = events.find(e => e.label === 'blur'), blur4 = events.find(e => e.label === 'blur4');
        expect([blur.pattern, blur.elements, blur.parameters]).toEqual(['MapOverlap 3D', [4, 5, 6], { overlap: [1, 2, 1] }]);
        expect([blur4.pattern, blur4.elements]).toEqual(['MapOverlap 4D', [3, 3, 3, 3]]);
        expect(compareTrace(r.program, events)).toEqual([]);
    });
});

describe('regions that are not loops', () => {
    const regionDesign = (scope = true) => {
        const { d, id } = make([['a'], ['b'], ['c'], ['d']], [['f', 'Map'], ['g', 'Map'], ['h', 'Map']],
            [['a', 'f', 'elwise'], ['f', 'b', 'out'], ['b', 'g', 'elwise'], ['g', 'c', 'out'], ['c', 'h', 'elwise'], ['h', 'd', 'out']], [['Group']]);
        const group = d.loops[0];
        group.scope = scope; group.count = scope ? 1 : 3;
        d.skeletons.find(s => s.id === id.f).parent = group.id;
        d.skeletons.find(s => s.id === id.g).parent = group.id;
        return { d, id, group };
    };

    it('group calls: a block with a trace scope, run once, and no for', () => {
        const { d, group } = regionDesign();
        const r = lower(d);
        expect(r.errors).toEqual([]);
        expect(r.program.steps[0]).toMatchObject({ type: 'loop', scope: true, count: 1 });
        const cpp = emitCpp(r.program);
        expect(cpp.text).not.toMatch(/for \(size_t/);
        expect(cpp.text).toMatch(/\{\n\t\tSKEPU_TRACE_SCOPE\("Group"\);\n\t\tf\(/);
        expect(emitPython(r.program).text).toMatch(/ {4}with skepu\.scope\("Group"\):\n {8}f\(/);
        expect(cpp.sites.find(s => s.id === group.id)).toMatchObject({ type: 'region', count: 1 });
    });

    it('are loops only when they repeat, and the names of the iterations do not count the scopes', () => {
        const { d } = regionDesign(false);
        expect(emitCpp(lower(d).program).text).toMatch(/for \(size_t iteration = 0; iteration < 3;/);
        const { d: nested, id } = make([['a'], ['b']], [['f', 'Map']], [['a', 'f', 'elwise'], ['f', 'b', 'out']], [['Scope'], ['Loop']]);
        nested.loops[0].scope = true; nested.loops[1].parent = nested.loops[0].id; nested.skeletons[0].parent = nested.loops[1].id;
        expect(emitCpp(lower(nested).program).text).toMatch(/for \(size_t iteration = 0;/);
    });

    it('have the data in the region of the call that writes it: each update is in the region of its own writer', () => {
        const { d, id, group } = regionDesign();
        let r = lower(d);
        expect(r.dataRegion).toEqual({ [id.a]: null, [id.b]: group.id, [id.c]: group.id, [id.d]: null });      // d is written by h, which is not in the region
        d.skeletons.find(s => s.id === id.h).parent = null;
        const other = { id: 'lx', label: 'Other', count: 1, scope: true, parent: null };
        d.loops.push(other);
        d.skeletons.push(defaultSkeleton('Map', 'sk', 'k'));
        d.positions.sk = { x: 0, y: 900 };
        d.containers.push({ ...defaultContainer('b2', 'b'), object: id.b });        // another update of b, written by k (outside the region)
        d.edges.push({ id: 'extra', from: 'sk', to: 'b2', role: 'out' });
        r = lower(d);
        expect(r.dataRegion[id.b]).toBe(group.id);
        expect(r.dataRegion.b2).toBeNull();
        expect(r.warnings.map(w => w.msg).join()).toMatch(/Other has no calls/);
    });

    it('nest, and a region with a call in it is not empty', () => {
        const { d, id, group } = regionDesign();
        d.loops.push({ id: 'inner', label: 'Inner', count: 1, scope: true, parent: group.id });
        d.skeletons.find(s => s.id === id.g).parent = 'inner';
        const r = lower(d);
        expect(r.ok).toBe(true);
        expect(r.warnings.map(w => w.msg).join()).not.toMatch(/has no calls/);
        expect(r.program.steps[0].steps.map(s => s.type)).toEqual(['call', 'loop']);
    });
});

describe('duplicating nodes', () => {
    const chain = () => make([['a', 'Vector', [8]], ['b', 'Vector', [8]], ['c', 'Vector', [8]]], [['f', 'Map'], ['g', 'Map']],
        [['a', 'f', 'elwise'], ['f', 'b', 'out'], ['b', 'g', 'elwise'], ['g', 'c', 'out']]);

    it('makes a call that reads what the original reads and writes data of its own, in the format of the original', () => {
        const { d, id } = chain();
        d.containers.find(c => c.id === id.b).auto = true;
        const made = duplicateNodes(d, [id.f]);
        expect(made).toHaveLength(2);
        const r = lower(d);
        expect(r.errors).toEqual([]);
        const [f, copy] = [r.program.calls.find(c => c.name === 'f'), r.program.calls.find(c => c.name === 'f_copy')];
        expect(copy.elwise.map(c => c.name)).toEqual(['a']);                // the same input
        expect(copy.out.name).toBe('b_copy');                                // an output of its own
        expect(copy.out.size).toEqual([8]);
        expect(f.out.name).toBe('b');
        expect(d.containers.find(c => c.label === 'b_copy').auto).toBe(true);
        expect(d.positions[made[0]]).toEqual({ x: d.positions[id.f].x + 40, y: d.positions[id.f].y + 40 });
    });

    it('keeps what is between copied nodes between the copies', () => {
        const { d, id } = chain();
        duplicateNodes(d, [id.f, id.b, id.g]);
        const r = lower(d);
        expect(r.errors).toEqual([]);
        const copy = r.program.calls.find(c => c.name === 'g_copy');
        expect(copy.elwise.map(c => c.name)).toEqual(['b_copy']);           // g_copy reads the copy of b, which f_copy writes
        expect(r.program.calls.find(c => c.name === 'f_copy').out.name).toBe('b_copy');
        expect(d.containers.map(c => c.label).sort()).toEqual(['a', 'b', 'b_copy', 'c', 'c_copy']);
    });

    it('brings what is in a copied region, into the region of the original when it is in one', () => {
        const { d, id } = chain();
        d.loops.push({ id: 'lo', label: 'Outer', count: 2, scope: false, parent: null }, { id: 'li', label: 'Inner', count: 1, scope: true, parent: 'lo' });
        d.skeletons.find(s => s.id === id.f).parent = 'li';
        const made = duplicateNodes(d, ['lo']);
        const loops = d.loops.filter(l => made.includes(l.id));
        const outer = loops.find(l => l.label === 'Outer_copy'), inner = loops.find(l => l.label === 'Inner_copy');
        expect(inner.parent).toBe(outer.id);
        expect(d.skeletons.find(s => s.label === 'f_copy').parent).toBe(inner.id);
        expect(outer.parent).toBeNull();
        expect(d.skeletons.find(s => s.label === 'g_copy')).toBeUndefined();   // g is not in the region
        expect(lower(d).errors).toEqual([]);
    });

    it('gives the label of a copy of a copy, a random stream its own copy when selected and shared when not, and a copied container its format', () => {
        const { d, id } = chain();
        d.containers.push(defaultPrng('p1', 'noise'));
        d.edges.push({ id: 'er', from: 'p1', to: id.f, role: 'random' });
        duplicateNodes(d, [id.f]);
        expect(d.edges.filter(e => e.role === 'random').map(e => e.from)).toEqual(['p1', id.f]);        // shared: the copy continues the chain
        duplicateNodes(d, [id.f, 'p1']);
        expect(d.containers.filter(c => c.kind === 'PRNG').map(c => c.label)).toEqual(['noise', 'noise_copy']);
        expect(d.skeletons.map(s => s.label)).toContain('f_copy2');
        d.containers.find(c => c.id === id.b).auto = true;
        const [b] = duplicateNodes(d, [id.b]).map(x => d.containers.find(c => c.id === x));
        expect(b).toMatchObject({ label: 'b_copy3', auto: false, size: [8], kind: 'Vector' });
    });
});

describe('random designs', () => {
    const nodes = d => d.containers.length + d.skeletons.length + d.loops.length;
    const depthRatio = d => {                      // the longest chain of calls by data, over the number of calls
        const r = lower(d), level = new Map();
        let best = 0;
        for (const c of r.program.calls) {
            const reads = [...c.elwise, ...c.proxies, c.vertical, c.horizontal, ...(c.reads ?? [])].filter(Boolean);
            const l = 1 + Math.max(0, ...reads.map(x => level.get(x.id) ?? 0));
            best = Math.max(best, l);
            for (const w of [c.out, ...(c.writes ?? [])].filter(Boolean)) level.set(w.id, l);
        }
        return best / r.program.calls.length;
    };
    const mean = (list, f) => list.reduce((a, x) => a + f(x), 0) / list.length;
    const many = (settings, n = 25) => Array.from({ length: n }, (_, k) => randomDesign({ ...settings, seed: `test${k}` }).design);

    it('make valid designs, whatever the settings: no problems, no warnings, and a program in both languages', () => {
        for (const size of SIZES) for (const arity of ARITIES) for (const shape of SHAPES)
            for (const d of many({ size, arity, shape }, 8)) {
                const r = lower(d);
                expect([size, arity, shape, r.errors.map(e => e.msg), r.warnings.map(w => w.msg)]).toEqual([size, arity, shape, [], []]);
                expect(emitCpp(r.program).text).toContain('int main');
                expect(emitPython(r.program).text).toContain('def main');
                const trace = synthesizeTrace(r.program, { dir: '/x', file: 'a.cpp', lines: [], sites: [] });
                if (!trace.truncated) expect(compareTrace(r.program, trace.events)).toEqual([]);      // (a trace that has the first iterations of the loops is not all of the program)
            }
    });

    it('are the same for the same settings and seed, and different for another seed', () => {
        const a = randomDesign({ size: 'medium', seed: 'abc' }), b = randomDesign({ size: 'medium', seed: 'abc' });
        expect(JSON.stringify(a.design)).toBe(JSON.stringify(b.design));
        expect(JSON.stringify(randomDesign({ size: 'medium', seed: 'abd' }).design)).not.toBe(JSON.stringify(a.design));
        expect(JSON.stringify(randomDesign({ size: 'medium', shape: 'wide', seed: 'abc' }).design)).not.toBe(JSON.stringify(a.design));
    });

    it('make a seed when there is none, and return it', () => {
        const a = randomDesign({ seed: '' }), b = randomDesign({ seed: '  ' });
        expect(a.seed).toMatch(/^[a-z0-9]{3,}$/);
        expect(randomDesign({ seed: a.seed }).design).toEqual(randomDesign({ seed: a.seed }).design);
        expect(a.seed === b.seed && JSON.stringify(a.design) === JSON.stringify(b.design)).toBe(false);
    });

    it('have about the nodes that the size says', () => {
        const count = size => many({ size }).map(nodes);
        expect(Math.max(...count('small'))).toBeLessThan(25);
        expect(Math.min(...count('medium'))).toBeGreaterThan(20);
        expect(Math.max(...count('medium'))).toBeLessThan(80);
        expect(Math.min(...count('large'))).toBeGreaterThan(70);
        expect(mean(many({ size: 'large' }), nodes)).toBeLessThan(160);
        expect(mean(many({ size: 'large' }), nodes)).toBeGreaterThan(mean(many({ size: 'medium' }), nodes) * 1.8);
    });

    it('are deeper for a deep shape and wider for a wide one', () => {
        const ratio = shape => mean(many({ size: 'large', shape }), depthRatio);
        expect(ratio('deep')).toBeGreaterThan(ratio('balanced'));
        expect(ratio('balanced')).toBeGreaterThan(ratio('wide'));
        const sources = shape => mean(many({ size: 'large', shape }), d => { const written = new Set(d.edges.filter(e => e.role === 'out').map(e => e.to)); return d.containers.filter(c => c.kind !== 'PRNG' && !written.has(c.id)).length / d.skeletons.length; });
        expect(sources('wide')).toBeGreaterThan(sources('deep') * 2);
    });

    it('have more inputs for a large arity', () => {
        const inputs = arity => mean(many({ size: 'large', arity }), d => d.edges.filter(e => e.role !== 'out' && e.role !== 'random').length / d.skeletons.length);
        expect(inputs('large')).toBeGreaterThan(inputs('small') * 1.5);
    });

    it('reuse names: calls write new updates of data that exists, in place for a Map, and the streams are chains', () => {
        let outputs = 0, updates = 0, inPlace = 0, chains = 0, streams = 0;
        for (const d of many({ size: 'large' }, 20)) {
            const key = c => c.object ?? c.id;
            const nodes = new Map(d.containers.map(c => [c.id, c]));
            for (const e of d.edges.filter(e => e.role === 'out')) {
                outputs++;
                const c = nodes.get(e.to);
                if (d.containers.some(x => x !== c && key(x) === key(c))) updates++;
                if (d.edges.some(r => r.to === e.from && r.role === 'elwise' && key(nodes.get(r.from)) === key(c))) inPlace++;
            }
            const random = d.edges.filter(e => e.role === 'random');
            chains += random.filter(e => e.from.startsWith('s')).length;
            streams += d.containers.filter(c => c.kind === 'PRNG').length;
            for (const from of new Set(random.map(e => e.from))) expect(random.filter(e => e.from === from)).toHaveLength(1);      // one edge out of a stream or a Map
        }
        expect(updates / outputs).toBeGreaterThan(0.1);
        expect(updates / outputs).toBeLessThan(0.6);
        expect(inPlace).toBeGreaterThan(0);
        expect(chains).toBeGreaterThan(0);
        expect(streams).toBeGreaterThan(0);
    });

    it('use every kind of node when they are large: all skeletons, externals, streams, loops and regions', () => {
        const seen = { patterns: new Set(), prng: 0, loop: 0, scope: 0, nested: 0, backends: new Set(), proxies: 0 };
        for (const d of many({ size: 'large', arity: 'large' }, 20)) {
            d.skeletons.forEach(s => { seen.patterns.add(s.pattern); if (s.backend) seen.backends.add(s.backend); });
            seen.prng += d.containers.filter(c => c.kind === 'PRNG').length;
            seen.loop += d.loops.filter(l => !l.scope).length; seen.scope += d.loops.filter(l => l.scope).length;
            seen.nested += d.loops.filter(l => l.parent).length;
            seen.proxies += d.edges.filter(e => e.role === 'proxy').length;
        }
        expect([...seen.patterns].sort()).toEqual(['External', 'IO', 'Map', 'MapOverlap', 'MapPairs', 'MapPairsReduce', 'MapReduce', 'Reduce', 'Scan']);
        expect(seen.prng).toBeGreaterThan(0); expect(seen.loop).toBeGreaterThan(0); expect(seen.scope).toBeGreaterThan(0);
        expect(seen.nested).toBeGreaterThan(0); expect(seen.proxies).toBeGreaterThan(0);
        expect([...seen.backends].sort()).toEqual(['cpu', 'opencl', 'openmp']);
    });
});

describe('user functions as lambdas', () => {
    const lambdas = { userFunctions: 'lambdas' };

    it('are in the declaration of the skeleton in C++, with the type that they return, and there is no function before main', () => {
        const text = emitCpp(lower(sampleDesign()).program, lambdas).text;
        expect(text).not.toMatch(/^float \w+\(/m);
        expect(text).toMatch(/auto smooth = skepu::MapOverlap\(\[\]\(skepu::Region1D<float> r\) -> float\n\t\{\n\t\t\/\/ TODO: the function of smooth\n\t\treturn r\(0\);\n\t\}\);/);
        expect(text).toMatch(/auto scale = skepu::Map<1>\(\[\]\(float a, float factor\) -> float/);
        // a MapReduce has two: the map and the reduction
        expect(text).toMatch(/skepu::MapReduce<1>\(\[\]\(float a\) -> float\n\t\{[^}]*\},\n\t\[\]\(float a, float b\) -> float\n\t\{[^}]*\}\);/);
        expect(text).toContain('smooth.setOverlap(1);');            // what is set after the declaration is as before
    });

    it('have the body that the user gave, and their lines are the skeleton\'s', () => {
        const d = sampleDesign();
        d.skeletons.find(s => s.id === 's2').bodies = { _f: { cpp: 'float x = a * factor;\nreturn x;' } };
        const r = lower(d), cpp = emitCpp(r.program, lambdas);
        expect(cpp.text).toContain('\t\tfloat x = a * factor;\n\t\treturn x;');
        const line = cpp.text.split('\n').findIndex(l => l.includes('float x = a * factor;'));
        expect(cpp.lines[line]).toContain('s2');
        expect(cpp.sites.find(s => s.id === 's2').line).toBeGreaterThan(line);       // the call is after the declaration
    });

    it('are lambdas in Python when the body is one line, and functions when it is not', () => {
        const d = sampleDesign();
        d.skeletons.find(s => s.id === 's2').bodies = { _f: { py: 'x = 2.0\nreturn a * x' } };            // two lines: stays a function
        d.skeletons.find(s => s.id === 's1').bodies = { _f: { py: '# a note\n\nreturn r[0] + r[1]  ' } };  // one line, with a note and a blank: a lambda
        const text = emitPython(lower(d).program, lambdas).text;
        expect(text).toContain('smooth = skepu.MapOverlap(lambda r: r[0] + r[1])');
        expect(text).toContain('def scale_f(a, factor):\n    x = 2.0\n    return a * x');
        expect(text).toContain('scale = skepu.Map(scale_f)');
        expect(text).toContain('energy = skepu.MapReduce(lambda a: a, lambda a, b: a + b)');
        expect(text).not.toContain('def smooth_f');
        expect(text).not.toContain('def energy_map');
    });

    it('are functions in Python when a line has a comment, and the free-standing form is the default', () => {
        const d = sampleDesign();
        d.skeletons.find(s => s.id === 's2').bodies = { _f: { py: 'return a * factor  # scaled' } };
        expect(emitPython(lower(d).program, lambdas).text).toContain('def scale_f(a, factor):');
        const plain = emitPython(lower(sampleDesign()).program).text;
        expect(plain).toContain('def smooth_f(r):');
        expect(plain).not.toContain('lambda');
        expect(emitCpp(lower(sampleDesign()).program).text).toBe(emitCpp(lower(sampleDesign()).program, { userFunctions: 'functions' }).text);
    });

    it('have the same externals, random streams and setters as the functions', () => {
        const r = lower(allSkeletonsDesign());
        const strip = t => t.replace(/\n/g, ' ');
        for (const emit of [emitCpp, emitPython]) {
            const a = emit(r.program).text, b = emit(r.program, lambdas).text;
            for (const marker of ['setPRNG', 'set_prng', 'skepu::external', 'skepu.external', 'setBackend', 'set_backend', 'setOverlap', 'set_overlap'])
                expect([marker, a.split(marker).length]).toEqual([marker, b.split(marker).length]);
        }
    });
});

describe('where the containers are declared', () => {
    const firstUse = { containerPlacement: 'first-use' };
    // a -> f -> b -> g -> c, with f and g in a loop or a scope, and an unused container and a random stream
    const design = (kind = 'none') => {
        const { d, id } = make([['a'], ['b'], ['c'], ['unused'], ['t']], [['f', 'Map'], ['g', 'Map'], ['h', 'Map']],
            [['a', 'f', 'elwise'], ['f', 'b', 'out'], ['b', 'g', 'elwise'], ['g', 'c', 'out'], ['c', 'h', 'elwise'], ['h', 't', 'out']], kind === 'none' ? [] : [['Group']]);
        if (kind !== 'none') {
            d.loops[0].scope = kind === 'scope'; d.loops[0].count = kind === 'scope' ? 1 : 3;
            for (const x of ['f', 'g']) d.skeletons.find(s => s.id === id[x]).parent = d.loops[0].id;
        }
        d.containers.push(defaultPrng('p', 'noise'));
        d.edges.push({ id: 'er', from: 'p', to: id.f, role: 'random' });
        return { d, id };
    };
    const declarations = text => text.split('\n').flatMap((l, i) => /^\s*skepu::(Vector|Matrix|PRNG)/.test(l) ? [[i, l.match(/\b(\w+)\(/g).pop().slice(0, -1)]] : []);

    it('is grouped by default, as before', () => {
        const { d } = design();
        const text = emitCpp(lower(d).program).text;
        expect(text).toContain('// containers');
        expect(text).toBe(emitCpp(lower(d).program, { containerPlacement: 'grouped' }).text);
        const names = declarations(text).map(x => x[1]);
        expect(names).toEqual(['a', 'b', 'c', 'unused', 't', 'noise']);
        expect(Math.max(...declarations(text).map(x => x[0]))).toBeLessThan(text.split('\n').findIndex(l => l.includes('// skeletons')));
    });

    it('is before the first use, with the random streams and what is not used at the start', () => {
        const { d } = design();
        const text = emitCpp(lower(d).program, firstUse).text, lines = text.split('\n');
        const at = name => lines.findIndex(l => new RegExp(`skepu::\\w+(<float>)? ${name}\\(`).test(l));
        const use = name => lines.findIndex(l => new RegExp(`^\\s+${name}\\(`).test(l));
        expect(lines.slice(at('a'), use('f')).every(l => /skepu::/.test(l))).toBe(true);       // a and b are declared just before the call of f
        expect(at('b')).toBeLessThan(use('f'));
        expect(at('c')).toBeLessThan(use('g'));
        expect(at('c')).toBeGreaterThan(use('f'));
        expect(at('t')).toBeLessThan(use('h'));
        expect(at('noise')).toBeLessThan(at('a'));
        expect(at('unused')).toBeLessThan(at('a'));
        expect(text).toContain('// random streams, and data that no call uses');
    });

    it('does not put a container in a loop, but in a scope that has all its uses', () => {
        const before = (lines, name) => lines.findIndex(l => new RegExp(`skepu::Vector<float> ${name}\\(`).test(l));
        const loop = emitCpp(lower(design('loop').d).program, firstUse).text.split('\n');
        const forAt = loop.findIndex(l => l.includes('for (size_t'));
        for (const name of ['a', 'b', 'c']) expect(before(loop, name)).toBeLessThan(forAt);          // (a loop makes its own again for each iteration)

        const lines = emitCpp(lower(design('scope').d).program, firstUse).text.split('\n');
        const open = lines.findIndex(l => l.includes('SKEPU_TRACE_SCOPE'));
        expect(before(lines, 'a')).toBeGreaterThan(open);      // only f, in the scope, uses a and b: they are in it
        expect(before(lines, 'b')).toBeGreaterThan(open);
        expect(before(lines, 'c')).toBeLessThan(open);         // g, in the scope, and h, after it, use c: before the scope
    });

    it('is the same in Python, and the placement is a map of the steps', () => {
        const { d } = design();
        const r = lower(d);
        const py = emitPython(r.program, firstUse).text.split('\n');
        expect(py.findIndex(l => l.startsWith('    b = skepu.Vector'))).toBeLessThan(py.findIndex(l => l.startsWith('    f(b')));
        expect(py).toContain('    # random streams, and data that no call uses');
        const { before, start } = containerPlacement(r.program);
        expect(start.map(c => c.label).sort()).toEqual(['noise', 'unused']);
        expect([...before.values()].flat().map(c => c.label).sort()).toEqual(['a', 'b', 'c', 't']);
    });

    it('is where the trace allocates the containers, and where it frees them', () => {
        const { d } = design('scope');
        const r = lower(d), cpp = emitCpp(r.program, firstUse);
        const trace = synthesizeTrace(r.program, { dir: '/x', file: 'a.cpp', lines: cpp.lines, sites: cpp.sites, containerPlacement: 'first-use' });
        const order = trace.events.map(e => e.type === 'allocation' ? `+${e.label}` : e.type === 'deallocation' ? `-${e.label}` : e.type === 'region' ? `{${e.label}` : e.type === 'skeleton_call' ? e.label : null).filter(Boolean);
        expect(order.slice(0, 2).sort()).toEqual(['+noise', '+unused']);
        expect(order.indexOf('+a')).toBeGreaterThan(order.indexOf('{Group'));            // a and b are in the scope
        expect(order.indexOf('+b')).toBeGreaterThan(order.indexOf('{Group'));
        expect(order.indexOf('+c')).toBeLessThan(order.indexOf('{Group'));               // c is before it
        expect(order.indexOf('-b')).toBeLessThan(order.indexOf('h'));                     // a and b are freed at the end of the scope
        expect(order.slice(-2)).toEqual(['-noise', '-unused']);        // (freed in the reverse of the order that they were made)
        expect(compareTrace(r.program, trace.events)).toEqual([]);
        const grouped = synthesizeTrace(r.program, { dir: '/x', file: 'a.cpp', lines: cpp.lines, sites: cpp.sites }).events.filter(e => e.type === 'allocation');
        expect(grouped).toHaveLength(6);
    });
});

describe('where the skeletons are declared', () => {
    const first = { skeletonPlacement: 'first-use' };
    // a -> f -> b -> g -> c -> h, with f and g in a loop or a scope
    const design = kind => {
        const { d, id } = make([['a'], ['b'], ['c'], ['t']], [['f', 'Map'], ['g', 'Map'], ['h', 'Map']],
            [['a', 'f', 'elwise'], ['f', 'b', 'out'], ['b', 'g', 'elwise'], ['g', 'c', 'out'], ['c', 'h', 'elwise'], ['h', 't', 'out']], kind ? [['Group']] : []);
        if (kind) {
            d.loops[0].scope = kind === 'scope'; d.loops[0].count = kind === 'scope' ? 1 : 3;
            for (const x of ['f', 'g']) d.skeletons.find(s => s.id === id[x]).parent = d.loops[0].id;
        }
        return d;
    };
    const at = (lines, re) => lines.findIndex(l => re.test(l));

    it('are grouped before the calls by default, as before', () => {
        const d = design();
        const text = emitCpp(lower(d).program).text;
        expect(text).toContain('// skeletons');
        expect(text).toBe(emitCpp(lower(d).program, { skeletonPlacement: 'grouped' }).text);
        const lines = text.split('\n');
        expect(Math.max(...['f', 'g', 'h'].map(n => at(lines, new RegExp(`auto ${n} =`))))).toBeLessThan(at(lines, /^\tf\(/));
    });

    it('are before the call of each when that is asked, with no section of their own', () => {
        const lines = emitCpp(lower(design()).program, first).text.split('\n');
        expect(lines).not.toContain('\t// skeletons');
        for (const n of ['f', 'g', 'h']) {
            expect(at(lines, new RegExp(`auto ${n} = skepu::Map`))).toBeLessThan(at(lines, new RegExp(`^\\t${n}\\(`)));
            expect(at(lines, new RegExp(`^\\t${n}\\(`)) - at(lines, new RegExp(`auto ${n} = skepu::Map`))).toBeLessThanOrEqual(2);
        }
        expect(at(lines, /auto g =/)).toBeGreaterThan(at(lines, /^\tf\(/));          // g is declared after the call of f
        const py = emitPython(lower(design()).program, first).text.split('\n');
        expect(at(py, /    g = skepu\.Map/)).toBeGreaterThan(at(py, /^ {4}f\(/));
        expect(py).not.toContain('    # skeletons');
    });

    it('are not declared inside a loop, but inside a scope that has the call', () => {
        const loop = emitCpp(lower(design('loop')).program, first).text.split('\n');
        const forAt = at(loop, /for \(size_t/);
        expect(at(loop, /auto f =/)).toBeLessThan(forAt);
        expect(at(loop, /auto g =/)).toBeLessThan(forAt);               // (both are in the loop: before it)
        const scope = emitCpp(lower(design('scope')).program, first).text.split('\n');
        const open = at(scope, /SKEPU_TRACE_SCOPE/);
        expect(at(scope, /auto f =/)).toBeGreaterThan(open);
        expect(at(scope, /auto g =/)).toBeGreaterThan(open);
        expect(at(scope, /auto h =/)).toBeGreaterThan(at(scope, /^\t}/));       // h is after the scope
    });

    it('keep the lines of their nodes and work with the lambdas and the containers before their first use', () => {
        const r = lower(design());
        for (const options of [{ ...first, userFunctions: 'lambdas' }, { ...first, containerPlacement: 'first-use', whitespace: 'readable' }]) {
            const cpp = emitCpp(r.program, options);
            expect(cpp.lines).toHaveLength(cpp.text.split('\n').length - 1);
            expect(cpp.lines[at(cpp.text.split('\n'), /auto g =/)]).toContain(r.program.calls[1].skeleton.id);
            expect(cpp.sites.every(s => cpp.text.split('\n')[s.line - 1].trim() !== '')).toBe(true);
            expect(cpp.text).not.toMatch(/\n\n\n/);
        }
    });
});

describe('whitespace', () => {
    const readable = { whitespace: 'readable' }, compact = { whitespace: 'compact' };
    const program = () => lower(allSkeletonsDesign()).program;
    const comments = (text, mark) => text.split('\n').map(l => l.trim()).filter(l => l.startsWith(mark) && !l.startsWith('#include'));

    it('is balanced by default: nothing changes', () => {
        expect(emitCpp(program()).text).toBe(emitCpp(program(), { whitespace: 'balanced' }).text);
        expect(emitPython(program()).text).toBe(emitPython(program(), { whitespace: 'balanced' }).text);
    });

    it('has no comments of its own when it is compact, and keeps what the user wrote', () => {
        const d = allSkeletonsDesign();
        d.skeletons.find(s => s.id === 's3').bodies = { _f: { cpp: '// mixing\nreturn a + b;', py: '# mixing\nreturn a + b' } };
        const r = lower(d);
        const cpp = emitCpp(r.program, compact).text, py = emitPython(r.program, compact).text;
        expect(comments(cpp, '//')).toEqual(['// mixing']);
        expect(comments(py, '#')).toEqual(['# mixing']);
        expect(cpp).not.toMatch(/TODO|made by the SkeVU designer/);
        expect(py).not.toMatch(/TODO|made by the SkeVU designer/);
        expect(py).toMatch(/with skepu\.external\(read=prefix, write=summary, label="inspect"\):\n {8}pass\n/);       // (an empty body needs a pass)
        expect(cpp).toMatch(/skepu::external\("inspect", skepu::read\(prefix\), \[&\]\n\t\{\n\t\}, skepu::write\(summary\)\);/);
        expect(cpp.split('\n').length).toBeLessThan(emitCpp(r.program).text.split('\n').length);
    });

    it('has the same code when it is compact, apart from the comments and the blank lines', () => {
        const strip = text => text.split('\n').filter(l => l.trim() && !l.trim().startsWith('//') && !l.trim().startsWith('#include')).join('\n');
        const r = program();
        expect(strip(emitCpp(r, compact).text)).toBe(strip(emitCpp(r).text).replace(/^\t\/\/ .*\n/gm, ''));
    });

    it('is readable: a blank line between the skeletons, the calls and the regions, and no more than one', () => {
        const text = emitCpp(program(), readable).text, lines = text.split('\n');
        expect(text).not.toMatch(/\n\n\n/);
        expect(text).not.toMatch(/\{\n\n/);                                       // none after the opening of a block
        const at = re => lines.findIndex(l => re.test(l));
        expect(lines[at(/auto ramp_up =/) + 1]).toBe('');                       // the skeletons are apart
        expect(lines[at(/smooth\.setPad/) + 1]).toBe('');                       // and what is set of one stays with it
        expect(lines[at(/make_noise\.setPRNG/) - 1]).toMatch(/make_noise =/);
        expect(lines[at(/for \(size_t/) - 1]).toBe('');                         // a region is apart from what comes before it
        const closing = lines.findIndex((l, i) => i > at(/for \(size_t/) && l === '\t}');
        expect(lines[closing + 1]).toBe('');                                    // and from what comes after
        expect(lines[at(/dot_result = dot/) + 1]).toMatch(/io::cout/);       // the print of a result stays with the call
    });

    it('keeps the containers before their first use with the call when it is readable', () => {
        const text = emitCpp(program(), { whitespace: 'readable', containerPlacement: 'first-use' }).text, lines = text.split('\n');
        const at = re => lines.findIndex(l => re.test(l));
        expect(lines[at(/Vector<float> ramp\(/) - 1]).toBe('\t// the program');           // (the first fragment follows the name of the section)
        expect(lines[at(/Vector<float> noise\(/) - 1]).toBe('');                          // the next is apart
        expect(lines[at(/Vector<float> ramp\(/) + 1]).toMatch(/^\tramp_up\(ramp\);/);       // the container is with the call that needs it
        expect(lines[at(/Vector<float> table\(/) + 1]).toMatch(/Vector<float> values\(/);   // two containers for one call: together
        expect(text).not.toMatch(/\n\n\n/);
    });

    it('is the same in Python', () => {
        const text = emitPython(program(), readable).text, lines = text.split('\n');
        expect(text).not.toMatch(/\n\n\n/);
        expect(text).not.toMatch(/:\n\n/);                                        // none after a line that opens a block
        const at = re => lines.findIndex(l => re.test(l));
        expect(lines[at(/    make_noise\.set_prng/) - 1]).toMatch(/make_noise = /);
        expect(lines[at(/    mix = /) - 1]).toBe('');
        expect(lines[at(/^ {12}damp\(/) - 1]).toBe('');                       // the calls in a region are apart too
    });

    it('works with the lambdas and the placement of the containers, and its blank lines are not nodes', () => {
        const r = program();
        for (const options of [{ userFunctions: 'lambdas' }, { containerPlacement: 'first-use' }, { userFunctions: 'lambdas', containerPlacement: 'first-use' }]) {
            const a = emitCpp(r, { ...options, whitespace: 'readable' });
            expect(a.text).not.toMatch(/\n\n\n/);
            expect(a.lines).toHaveLength(a.text.split('\n').length - 1);
            expect(a.sites.every(s => a.text.split('\n')[s.line - 1].trim() !== '')).toBe(true);
        }
    });
});

describe('where the data is', () => {
    const run = unset => {
        const { d, id } = make([['a'], ['b'], ['c']], [['f', 'Map', { backend: 'opencl' }], ['g', 'Map'], ['h', 'Map', { backend: 'cpu' }]],
            [['a', 'f', 'elwise'], ['f', 'b', 'out'], ['b', 'g', 'elwise'], ['g', 'c', 'out'], ['c', 'h', 'elwise']]);
        const program = lower(d).program, residency = new Residency(program, unset);
        return { id, moves: program.calls.map(call => residency.step(call).map(m => `${m.container.label} ${m.direction}`)), program };
    };

    it('says what a call needs moved, and is one place for the graph and for the trace', () => {
        const { moves, program } = run('host');
        expect(moves).toEqual([['a host-to-device'], ['b device-to-host'], []]);       // the call without a backend runs on the host
        expect(transfers(program).map(t => t.direction)).toEqual(['host-to-device']);  // the graph does not follow it
    });

    it('does not follow what has no backend when that is not known', () => {
        expect(run(undefined).moves).toEqual([['a host-to-device'], [], []]);
        const { program } = run(undefined);
        expect(placeOf(program.calls[1])).toBeUndefined();
        expect(placeOf(program.calls[1], 'host')).toBe('host');
        expect(placeOf(program.calls[0])).toBe('device');
    });
});

describe('objects: data nodes that are updates of the same data', () => {
    const chain = () => make([['a', 'Vector', [8]], ['b', 'Vector', [8]], ['c', 'Vector', [8]]], [['f', 'Map'], ['g', 'Map']],
        [['a', 'f', 'elwise'], ['f', 'b', 'out'], ['b', 'g', 'elwise'], ['g', 'c', 'out']]);
    const node = (d, id) => d.containers.find(c => c.id === id);

    it('are renamed together, and a name that another object has makes them one object (the other one keeps its format)', () => {
        const { d, id } = make([['a', 'Vector', [8]], ['a2', 'Vector', [8], 'a'], ['b', 'Vector', [16]]], [], []);
        expect(renameObject(d, node(d, id.a), 'x')).toEqual({});
        expect([id.a, id.a2].map(i => node(d, i).label)).toEqual(['x', 'x']);
        expect(renameObject(d, node(d, id.b), 'x')).toEqual({ merged: 'x' });          // b joins x: x's format
        expect(new Set(d.containers.map(objectKey)).size).toBe(1);
        expect(node(d, id.b)).toMatchObject({ label: 'x', size: [8] });
    });

    it('are never a random stream and data', () => {
        const d = emptyDesign();
        d.containers.push(defaultContainer('c1', 'data'), defaultPrng('c2', 'noise'));
        expect(renameObject(d, d.containers[0], 'noise').refused).toMatch(/noise is a random stream/);
        expect(joinObjects(d, d.containers[1], d.containers[0]).refused).toMatch(/not the same object/);
        expect(d.containers.map(objectKey)).toEqual(['c1', 'c2']);
    });

    it('are joined by the connection of two data nodes (the first one\'s name), and detached to a name of their own', () => {
        const { d, id } = chain();
        expect(joinObjects(d, node(d, id.a), node(d, id.c))).toEqual({ merged: 'a' });
        expect(node(d, id.c)).toMatchObject({ label: 'a', object: id.a });
        expect(joinObjects(d, node(d, id.a), node(d, id.c)).refused).toMatch(/is an update of a already/);
        expect(membersOf(d, id.a).map(c => c.id)).toEqual([id.a, id.c]);
        expect(detachNode(d, node(d, id.c))).toBe('a_2');
        expect(node(d, id.c)).toMatchObject({ label: 'a_2' });
        expect(objectKey(node(d, id.c))).not.toBe(objectKey(node(d, id.a)));
        expect(detachNode(d, node(d, id.c))).toBeNull();                                 // (it is one already)
    });

    it('have one format: what is edited is the format of all the nodes', () => {
        const { d, id } = make([['a', 'Vector', [8]], ['a2', 'Vector', [8], 'a']], [], []);
        Object.assign(node(d, id.a), { size: [32], type: 'int', init: '3' });
        syncObject(d, node(d, id.a));
        expect(node(d, id.a2)).toMatchObject({ size: [32], type: 'int', init: '3', label: 'a' });
    });

    it('are copied together: the copies of several nodes of an object are an object, a copy of one is a new one', () => {
        const { d, id } = make([['a', 'Vector', [8]], ['a2', 'Vector', [8], 'a']], [], []);
        const both = duplicateNodes(d, [id.a, id.a2]).map(i => node(d, i));
        expect(both.map(c => c.label)).toEqual(['a_copy', 'a_copy']);
        expect(new Set(both.map(objectKey)).size).toBe(1);
        expect(objectKey(both[0])).not.toBe(id.a);
        const [one] = duplicateNodes(d, [id.a]).map(i => node(d, i));
        expect(one.label).toBe('a_copy2');
        expect(one.object).toBeUndefined();
    });

    it('make a program with one declaration, one allocation in the trace, and the transfers of the object', () => {
        const { d, id } = make([['a'], ['a2', 'Vector', undefined, 'a'], ['a3', 'Vector', undefined, 'a'], ['z']], [['f', 'Map', { backend: 'opencl' }], ['g', 'Map', { backend: 'cpu' }], ['h', 'Map']],
            [['a', 'f', 'elwise'], ['f', 'a2', 'out'], ['a2', 'g', 'elwise'], ['g', 'a3', 'out'], ['a3', 'h', 'elwise'], ['h', 'z', 'out']]);
        const r = lower(d);
        expect(r.errors).toEqual([]);
        const cpp = emitCpp(r.program);
        expect(cpp.text.match(/skepu::Vector<float> a\(/g)).toHaveLength(1);
        expect(cpp.lines[cpp.text.split('\n').findIndex(l => l.includes('Vector<float> a('))]).toEqual([id.a, id.a2, id.a3]);
        const trace = synthesizeTrace(r.program, { dir: '/x', file: 'a.cpp', lines: cpp.lines, sites: cpp.sites });
        expect(trace.events.filter(e => e.type === 'allocation').map(e => e.label)).toEqual(['a', 'z']);
        expect(trace.events.filter(e => e.type === 'transfer').map(e => [e.label, e.direction])).toEqual([['a', 'host-to-device'], ['a', 'device-to-host']]);
        expect(transfers(r.program).map(t => [t.container, t.consumer])).toEqual([[id.a, id.f], [id.a2, id.g]]);      // (by the node that is read)
        expect(compareTrace(r.program, trace.events)).toEqual([]);
    });
});
