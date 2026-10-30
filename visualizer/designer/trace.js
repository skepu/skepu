// trace.js — the trace a design's program could produce (TRACE_FORMAT.md), synthesized from a model of SkePU without running it: calls in order with
// their containers and parameters, host–device transfers, call times from backend and size, and loop regions. Times are estimates; order, ids, labels,
// elements and transfers follow the design. No DOM. A call without a backend runs on the CPU.
'use strict';
import { PATTERNS, readsOf, writesOf, sizeOf, Residency, dimsOf, overlapOf, poolOf, strideOf, stridesOf } from './model.js';
import { containerPlacement, anonymousLabel, isAnonymous } from './codegen.js';

/** Per backend: launch time, time per element (ns) and thread count. */
const COST = { CPU: { launch: 2, ns: 6, threads: 1 }, OpenMP: { launch: 15, ns: 1.2, threads: 8 }, OpenCL: { launch: 250, ns: 0.15, threads: 8 } };
const BACKEND = { cpu: 'CPU', openmp: 'OpenMP', opencl: 'OpenCL' };
const START_ID = 1000;

const iterationsOf = (steps, cap) => steps.reduce((n, step) => n + (step.type === 'call' ? 1 : Math.min(step.count, cap) * iterationsOf(step.steps, cap)), 0);

/** The iteration cap per loop that keeps the trace within `budget` events (no cap while the calls are few). */
function capFor(steps, budget) {
    let cap = Infinity;
    while (cap > 1 && iterationsOf(steps, cap === Infinity ? 1e9 : cap) > budget) cap = cap === Infinity ? 64 : Math.floor(cap / 2);
    return cap;
}

const number = v => { const x = parseFloat(String(v ?? '').trim()); return Number.isFinite(x) ? x : null; };

/**
 * @param program  the lowered design (model.js lower())
 * @param where    { dir, file, lines, sites, containerPlacement, skeletonInstancing }: the C++ path and name, and the `lines` and `sites` of codegen.js emitCpp() (events carry
 *                 their lines); with 'first-use' placement, containers are allocated there and freed at the end of their block
 * @returns { events, ids, truncated }  ids[i]: the design node ids of event i (the first is the main one); truncated: the iteration cap applied to loops (null: none)
 */
export function synthesizeTrace(program, where) {
    const path = `${where.dir}/${where.file}`;
    const site = new Map((where.sites ?? []).map(s => [s.id, s.line]));
    const declared = id => { const i = (where.lines ?? []).findIndex(ids => ids.includes(id)); return i < 0 ? undefined : i + 1; };

    const events = [], ids = [];
    const push = (event, nodes) => { events.push(event); ids.push(nodes); return event; };
    let clock = 100;

    // the objects: skeleton instances first, then the data in the order of the program (as ids are made in the program)
    const instance = new Map();
    program.calls.forEach((call, i) => instance.set(call.skeleton.id, START_ID + i));
    let nextId = START_ID + program.calls.length;
    const object = new Map();
    const regions = [];
    const scope = () => (regions.length ? { region: regions[regions.length - 1] } : {});
    // the updates of an object are one object of the trace: one allocation, one id
    const allocate = c => {
        object.set(c.object, nextId++);
        push({ type: 'allocation', label: c.label, object_id: object.get(c.object), time: ++clock, ...(declared(c.id) ? { line: declared(c.id) } : {}), ...scope() }, c.nodes);
    };
    const placement = where.containerPlacement === 'first-use' ? containerPlacement(program) : null;
    const atStart = placement ? placement.start : program.containers;
    atStart.forEach(allocate);
    const idOf = c => object.get(c.object);
    const count = c => sizeOf(c);

    // where the data is (calls without a backend run on the CPU)
    const residency = new Residency(program, 'host');

    const transfer = (c, direction, nodes) => {
        const elements = count(c);
        const start = clock, end = clock + Math.max(3, Math.round(20 + elements * 4 / 12000));       // about 12 GB/s plus a latency
        clock = end + 1;
        push({ type: 'transfer', object_id: idOf(c), label: c.label, elements, direction, backend: 'OpenCL', start, end, time: end, ...scope() }, [c.id, ...nodes]);
    };

    const patternOf = call => {
        const s = call.skeleton;
        if (s.pattern === 'Reduce') return call.partial ? 'Reduce 1D-Partial' : call.second ? 'Reduce 2D' : 'Reduce 1D-Full';
        if (s.pattern === 'MapPool') return `MapPool ${dimsOf(s)}D`;
        if (s.pattern === 'MapOverlap') return `MapOverlap ${dimsOf(s)}D`;
        return s.pattern;
    };
    const elementsOf = call => {
        const first = call.elwise[0];
        switch (call.pattern) {
            case 'Reduce':               // Full: one element count; otherwise rows and columns (swapped when along the columns)
                return call.partial || call.second ? (call.skeleton.params.mode === 'ColWise' ? [first.size[1], first.size[0]] : [...first.size]) : [count(first)];
            case 'MapPool': return [...(call.out ?? first).size];
            case 'MapOverlap': return [...(dimsOf(call.skeleton) >= 2 ? (call.out ?? first) : first).size];
            case 'MapPairs': case 'MapPairsReduce': return [call.horizontal.size[0], call.vertical.size[0]];
            case 'MapReduce': return [Math.floor(count(first) / stridesOf(call.skeleton, call.elwise.length)[0])];
            case 'Map': return [Math.floor(count(call.out ?? first) / stridesOf(call.skeleton, 1 + call.elwise.length)[0])];          // strided: the output elements that are computed
            default: return [count(call.out ?? first)];
        }
    };
    const parametersOf = call => {
        const s = call.skeleton, p = {};
        if (call.pattern === 'MapOverlap') {
            p.overlap = overlapOf(s);
            if (s.params.edge !== 'Duplicate') p.edge = s.params.edge;
            if (s.params.edge === 'Pad' && number(s.params.pad) !== null) p.pad = number(s.params.pad);
            if (s.params.update && s.params.update !== 'Normal') p.update_mode = s.params.update;
        }
        if (call.pattern === 'Map' || call.pattern === 'MapReduce') {
            const strides = stridesOf(s, call.pattern === 'Map' ? 1 + call.elwise.length : call.elwise.length);
            if (strides.some(x => x !== 1)) p.strides = strides;
        }
        if (call.pattern === 'MapReduce' && number(s.params.start)) p.start_value = number(s.params.start);
        if (call.pattern === 'MapPool') { p.pool_size = poolOf(s); p.stride = strideOf(s); p.pool = true; }
        if (call.pattern === 'Reduce' && (call.partial || call.second)) {
            if (s.params.mode === 'ColWise') p.mode = 'ColWise';
            if (number(s.params.start)) p.start_value = number(s.params.start);
        }
        if (call.pattern === 'MapPairsReduce') {
            if (s.params.mode === 'ColWise') p.mode = 'ColWise';
            if (number(s.params.start)) p.start_value = number(s.params.start);
        }
        if (call.pattern === 'Scan') {
            if (s.params.mode === 'Exclusive') p.mode = 'Exclusive';
            if (number(s.params.start)) p.start_value = number(s.params.start);
        }
        return Object.keys(p).length ? p : null;
    };

    const scalarOf = new Map();           // call -> id of the scalar of its latest run (read by the printing I/O block)
    const io = (call, line, at, input, nodes) => {           // an I/O block (skepu::io::cout): an external event reading what is printed
        const t0 = clock;
        clock += 6;
        push({ type: 'external', label: 'I/O', file: path, ...(line ? { line: line + at } : {}), start: t0, end: t0 + 5, outputs: [], proxy_inputs: input === null ? [] : [input],
               elwise_inputs: [], scalar_inputs: [], ...scope() }, nodes);
    };

    const visitCall = call => {
        const s = call.skeleton, line = site.get(s.id);
        const nodes = [s.id];
        for (const m of residency.step(call)) transfer(m.container, m.direction, nodes);

        if (call.io) {                    // one event per printed item
            const prints = [...call.reads.map(c => [idOf(c), [s.id, c.id]]), ...call.scalars.filter(c => scalarOf.has(c.skeleton.id)).map(c => [scalarOf.get(c.skeleton.id), [s.id, c.skeleton.id]])];
            if (!prints.length) io(call, line, 0, null, nodes);
            prints.forEach(([input, ids], k) => io(call, line, k, input, ids));
            return;
        }

        const label = call.host ? s.label : isAnonymous(s, where) ? anonymousLabel(s, call, where.file, line) : call.name;          // anonymous skeletons are named by their call
        if (call.host) {
            const start = clock, end = clock + Math.max(3, Math.round(10 + readsOf(call).concat(writesOf(call)).reduce((n, c) => n + count(c), 0) * 0.002));
            clock = end + 1;
            push({ type: 'external', label, file: path, ...(line ? { line } : {}), start, end,
                   outputs: writesOf(call).map(idOf), proxy_inputs: call.reads.map(idOf), elwise_inputs: [], scalar_inputs: [], ...scope() }, nodes);
            return;
        }

        const backend = BACKEND[s.backend] ?? 'CPU', cost = COST[backend];
        const elements = elementsOf(call);
        const work = elements.reduce((a, b) => a * b, 1);
        const start = clock, end = clock + Math.max(2, Math.round(cost.launch + work * cost.ns / 1000));
        clock = end + 1;
        const scalar = call.scalar ? nextId++ : null;
        const event = { type: 'skeleton_call', object_id: instance.get(s.id), label, pattern: patternOf(call), backend, file: path, ...(line ? { line } : {}), elements,
            elwise_inputs: (call.pattern === 'MapPairs' || call.pattern === 'MapPairsReduce' ? [call.vertical, call.horizontal] : call.elwise).map(idOf), proxy_inputs: call.proxies.map(idOf), scalar_inputs: [],
            outputs: scalar !== null ? [scalar] : call.out ? [idOf(call.out)] : [], threads: cost.threads, start, end, ...scope() };
        if (backend === 'OpenCL') event.devices = 1;
        if (call.random) event.prng = idOf(call.random);
        const parameters = parametersOf(call);
        if (parameters) event.parameters = parameters;
        push(event, call.random ? [s.id, call.random.id] : nodes);
        if (call.pattern === 'MapOverlap' && s.params.update === 'RedBlack') {          // two passes: red, then black
            const second = { ...event, start: clock, end: clock + (event.end - event.start) };
            clock = second.end + 1;
            push(second, nodes);
        }

        if (scalar !== null) scalarOf.set(s.id, scalar);
    };

    const cap = capFor(program.steps, 400);
    const free = declared => { for (const c of [...declared].reverse()) push({ type: 'deallocation', label: c.label, object_id: idOf(c), time: ++clock, ...scope() }, c.nodes); };
    const run = (steps, depth, declared) => {
        for (const step of steps) {
            for (const c of placement?.before.get(step) ?? []) { allocate(c); declared.push(c); }
            if (step.type === 'call') { visitCall(step.call); continue; }
            const line = site.get(step.loop.id);
            for (let i = 0; i < Math.min(step.count, cap); i++) {
                push({ type: 'region', label: step.label, region_depth: depth + 1, file: path, ...(line ? { line } : {}), time: ++clock, ...scope() }, [step.loop.id]);
                regions.push(step.label);
                const inner = [];
                run(step.steps, depth + 1, inner);
                free(inner);
                regions.pop();
            }
        }
    };
    const top = [...atStart];
    run(program.steps, 0, top);
    free(top);
    return { events, ids, truncated: cap === Infinity ? null : cap };
}
