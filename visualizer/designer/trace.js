// trace.js — the trace that the program of a design could have (TRACE_FORMAT.md): what SKEPU_TRACING would write if the program was built
// and run, made from a small model of SkePU, not by running it: the calls in order with the containers they use and their parameters,
// the transfers that the backends make between host and device, the time of each call from its backend and size, the regions of the
// loops. The times are made up (a rough cost of each backend); the order, ids, labels, elements and transfers are what the design says.
// Pure: no DOM. A call or block that has no backend of its own is run as the program is started: CPU.
'use strict';
import { readsOf, writesOf, sizeOf, Residency, dimsOf, overlapOf } from './model.js';
import { containerPlacement } from './codegen.js';

/** The cost of a call: a launch, and a time for each element (nanoseconds) on each backend. */
const COST = { CPU: { launch: 2, ns: 6, threads: 1 }, OpenMP: { launch: 15, ns: 1.2, threads: 8 }, OpenCL: { launch: 250, ns: 0.15, threads: 8 } };
const BACKEND = { cpu: 'CPU', openmp: 'OpenMP', opencl: 'OpenCL' };
const START_ID = 1000;

const iterationsOf = (steps, cap) => steps.reduce((n, step) => n + (step.type === 'call' ? 1 : Math.min(step.count, cap) * iterationsOf(step.steps, cap)), 0);

/** The most iterations of a loop to simulate, so that the trace is a size that can be read: all of them while the calls are few. */
function capFor(steps, budget) {
    let cap = Infinity;
    while (cap > 1 && iterationsOf(steps, cap === Infinity ? 1e9 : cap) > budget) cap = cap === Infinity ? 64 : Math.floor(cap / 2);
    return cap;
}

const number = v => { const x = parseFloat(String(v ?? '').trim()); return Number.isFinite(x) ? x : null; };

/**
 * @param program  the lowered design (model.js lower())
 * @param where    { dir, file, lines, sites }: the path and name of the C++ file, and what codegen.js emitCpp() returned for it
 *                 (`lines`: the ids of the nodes of each line, `sites`: the lines of the calls and loops), so that the events have the lines
 *                 of the program that was made); `containerPlacement`: 'first-use' if the containers are declared before their first use (codegen.js),
 *                 so that they are allocated there, and freed at the end of the block that has them
 * @returns { events, ids, truncated }  ids[i]: the ids of the design nodes that event i is about (the first is the main one); truncated: the
 *          number of iterations that the loops were cut to (null: they are all there)
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
    // (the updates of an object are one object of the trace: one allocation, one id)
    const allocate = c => {
        object.set(c.object, nextId++);
        push({ type: 'allocation', label: c.label, object_id: object.get(c.object), time: ++clock, ...(declared(c.id) ? { line: declared(c.id) } : {}), ...scope() }, c.nodes);
    };
    const placement = where.containerPlacement === 'first-use' ? containerPlacement(program) : null;
    const atStart = placement ? placement.start : program.containers;
    atStart.forEach(allocate);
    const idOf = c => object.get(c.object);
    const count = c => sizeOf(c);

    // where the data is (a call that has no backend of its own runs as the program is started: on the CPU)
    const residency = new Residency(program, 'host');

    const transfer = (c, direction, nodes) => {
        const elements = count(c);
        const start = clock, end = clock + Math.max(3, Math.round(20 + elements * 4 / 12000));       // about 12 GB/s, and a latency
        clock = end + 1;
        push({ type: 'transfer', object_id: idOf(c), label: c.label, elements, direction, backend: 'OpenCL', start, end, time: end, ...scope() }, [c.id, ...nodes]);
    };

    const patternOf = call => {
        const s = call.skeleton;
        if (s.pattern === 'Reduce') return 'Reduce 1D-Full';
        if (s.pattern === 'MapOverlap') return `MapOverlap ${dimsOf(s)}D`;
        return s.pattern;
    };
    const elementsOf = call => {
        const first = call.elwise[0];
        switch (call.pattern) {
            case 'Reduce': return [...first.size];
            case 'MapOverlap': return [...(dimsOf(call.skeleton) >= 2 ? (call.out ?? first) : first).size];
            case 'MapPairs': case 'MapPairsReduce': return [call.horizontal.size[0], call.vertical.size[0]];
            case 'MapReduce': return [count(first)];
            default: return [count(call.out ?? first)];
        }
    };
    const parametersOf = call => {
        const s = call.skeleton, p = {};
        if (call.pattern === 'MapOverlap') {
            p.overlap = overlapOf(s);
            if (s.params.edge !== 'Duplicate') p.edge = s.params.edge;
            if (s.params.edge === 'Pad' && number(s.params.pad) !== null) p.pad = number(s.params.pad);
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

    const scalarOf = new Map();           // the call -> the id of the scalar that its latest run made (what an I/O block that prints it reads)
    const io = (call, line, at, input, nodes) => {           // an I/O block of the program (skepu::io::cout): an external block that reads what is printed
        const t0 = clock;
        clock += 6;
        push({ type: 'external', label: 'I/O', file: path, ...(line ? { line: line + at } : {}), start: t0, end: t0 + 5, outputs: [], proxy_inputs: input === null ? [] : [input],
               elwise_inputs: [], scalar_inputs: [], ...scope() }, nodes);
    };

    const visitCall = call => {
        const s = call.skeleton, line = site.get(s.id);
        const nodes = [s.id];
        for (const m of residency.step(call)) transfer(m.container, m.direction, nodes);

        if (call.io) {                    // one event for each thing that it prints, as skepu::io::cout makes them
            const prints = [...call.reads.map(c => [idOf(c), [s.id, c.id]]), ...call.scalars.filter(c => scalarOf.has(c.skeleton.id)).map(c => [scalarOf.get(c.skeleton.id), [s.id, c.skeleton.id]])];
            if (!prints.length) io(call, line, 0, null, nodes);
            prints.forEach(([input, ids], k) => io(call, line, k, input, ids));
            return;
        }

        const label = call.host ? s.label : call.name;
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
        const scalar = call.pattern === 'Reduce' || call.pattern === 'MapReduce' ? nextId++ : null;
        const event = { type: 'skeleton_call', object_id: instance.get(s.id), label, pattern: patternOf(call), backend, file: path, ...(line ? { line } : {}), elements,
            elwise_inputs: (call.pattern === 'MapPairs' || call.pattern === 'MapPairsReduce' ? [call.vertical, call.horizontal] : call.elwise).map(idOf), proxy_inputs: call.proxies.map(idOf), scalar_inputs: [],
            outputs: scalar !== null ? [scalar] : call.out ? [idOf(call.out)] : [], threads: cost.threads, start, end, ...scope() };
        if (backend === 'OpenCL') event.devices = 1;
        if (call.random) event.prng = idOf(call.random);
        const parameters = parametersOf(call);
        if (parameters) event.parameters = parameters;
        push(event, call.random ? [s.id, call.random.id] : nodes);

        if (scalar !== null) {
            scalarOf.set(s.id, scalar);
            if (call.printed) io(call, line, 1, scalar, nodes);           // the program prints the result: an I/O block that reads the scalar
        }
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
                free(inner);                                 // (what a scope declared is freed at the end of it)
                regions.pop();
            }
        }
    };
    const top = [...atStart];
    run(program.steps, 0, top);
    free(top);
    return { events, ids, truncated: cap === Infinity ? null : cap };
}
