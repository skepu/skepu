// test_communication_nodes.js — the collective operations of an MPI program are nodes of the graph: they read the
// containers they move and are what uses the containers afterwards depends on.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { DirectedGraph } from '../../script/backend/graph.js';
import { CommunicationNode } from '../../script/backend/node.js';
import { buildDisplayGraph, graphSettings } from '../../script/backend/pipeline.js';
import { mergeMpiTraces } from '../../script/backend/mpi.js';
import { parseTrace } from '../../script/backend/trace.js';
import { builder, trace } from '../event_builders.js';
import { testSettings } from '../helpers.js';
import { TraceDataProvider } from '../../script/TraceDataProvider.js';

const DIR = join(dirname(fileURLToPath(import.meta.url)), '../fixtures/mpi/trace_map_3');
const merged = mergeMpiTraces(['skepu_trace.json', 'skepu_trace.rank1.json', 'skepu_trace.rank2.json']
    .map(name => ({ name, text: readFileSync(join(DIR, name), 'utf-8') })));

const nodesOf = (g, type) => g.getAllNodes().filter(n => n.type === type);
const edgesBetween = (g, a, b) => g.getAllEdges?.().filter(e => e.source === a && e.target === b) ?? [];
const sourcesOf = n => n.getIncomingEdges().map(e => e.source);
// (the container updates that follow a call are between it and what reads its results)
const reads = (n, producer) => sourcesOf(n).some(s => s === producer || sourcesOf(s).includes(producer));

function graphOf(events, tweak = () => {}) {
    const s = testSettings();
    tweak(s);
    return DirectedGraph.fromEvents(trace(events), s);
}

// A collective that is part of a call is connected to it, also when the data of the containers does not do that
describe('the collectives of a call', () => {
    const b = builder();
    const external = { ...b.external('fill', { outs: [1] }), seq: 1 };
    const events = [
        b.alloc(1, 'A'),
        b.comm([1], { operation: 'allgather', seq: 1, phase: 'before', comm_seq: 1 }),   // (the container is gathered before the code writes it)
        b.comm([1], { operation: 'scatter', reason: 'skepu::external: the container is given to its owners', seq: 1, phase: 'after', comm_seq: 2 }),
        external,
        b.comm([], { operation: 'allgather', reason: 'the partial results are combined', seq: 2, phase: 'after', comm_seq: 3 }),
        { ...b.call('reduce', { ins: [1] }), seq: 2 },
    ];

    it('before the call feed it, and after it are fed by it', () => {
        const g = graphOf(events);
        const [gather, scatter, partials] = nodesOf(g, 'communication');
        const fill = g.getAllNodes().find(n => n.label === 'fill');
        const reduce = g.getAllNodes().find(n => n.label === 'reduce');
        expect(sourcesOf(fill)).toContain(gather);
        expect(sourcesOf(scatter)).toContain(fill);
        expect(sourcesOf(partials)).toContain(reduce);
        expect(sourcesOf(reduce)).toContain(scatter);
    });

    it('do not make a second edge where the data made one', () => {
        const c = builder();
        const g = graphOf([c.alloc(1, 'A'), c.comm([1], { seq: 1, phase: 'before' }), { ...c.call('use', { ins: [1] }), seq: 1 }]);
        const use = g.getAllNodes().find(n => n.label === 'use');
        expect(sourcesOf(use).filter(n => n.type === 'communication')).toHaveLength(1);
    });
});

describe('a communication node', () => {
    const b = builder();
    const events = [
        b.alloc(1, 'A'), b.alloc(2, 'B'), b.alloc(3, 'C'),
        b.call('produce', { outs: [1] }),
        b.comm([1], { operation: 'allgather' }),
        b.call('consume', { proxy: [1], outs: [2] }),          // (random access: it reads the gathered copy)
        b.call('elementwise', { ins: [1], outs: [3] }),        // (it reads the tiles, which the gather did not change)
    ];

    it('reads the container and is what the readers of the gathered copy read it from', () => {
        const g = graphOf(events);
        const [comm] = nodesOf(g, 'communication');
        const produce = g.getAllNodes().find(n => n.label === 'produce');
        const consume = g.getAllNodes().find(n => n.label === 'consume');
        expect(comm).toBeInstanceOf(CommunicationNode);
        expect(comm.label).toBe('allgather: A');
        expect(sourcesOf(comm).some(n => n === produce || sourcesOf(n).includes(produce))).toBe(true);
        expect(sourcesOf(consume).includes(comm)).toBe(true);        // (not on the producer directly)
        expect(comm.start).toBeLessThan(comm.end);
        expect(comm.totalOrder).toBeGreaterThan(produce.totalOrder);
        expect(comm.totalOrder).toBeLessThan(consume.totalOrder);
    });

    it('is not what the elements of a skeleton are read from: those are in the tiles, which a gather does not change', () => {
        const g = graphOf(events);
        const produce = g.getAllNodes().find(n => n.label === 'produce');
        const elementwise = g.getAllNodes().find(n => n.label === 'elementwise');
        expect(reads(elementwise, produce)).toBe(true);
        expect(sourcesOf(elementwise).some(n => n.type === 'communication')).toBe(false);
    });

    it('a write of the container makes the gathered copy out of date: what reads it after reads the tiles', () => {
        const c = builder();
        const g = graphOf([c.alloc(1, 'A'), c.call('first', { outs: [1] }), c.comm([1]), c.call('second', { ins: [1], outs: [1] }),
            c.call('third', { proxy: [1], outs: [2] })]);
        const second = g.getAllNodes().find(n => n.label === 'second');
        const third = g.getAllNodes().find(n => n.label === 'third');
        expect(reads(third, second)).toBe(true);
        expect(sourcesOf(third).some(n => n.type === 'communication')).toBe(false);
    });

    it('an skepu::external call writes the gathered copy of the first process; the scatter gives it to the tiles', () => {
        const c = builder();
        const events2 = [
            c.alloc(1, 'A'), c.alloc(2, 'B'),
            { ...c.external('fill'), outputs: [1] },
            c.comm([1], { operation: 'scatter', reason: 'the container is given to its owners', seq: 5, phase: 'before', comm_seq: 1 }),
            { ...c.call('use', { ins: [1], outs: [2] }), seq: 5 },
        ];
        const g = graphOf(events2);
        const fill = g.getAllNodes().find(n => n.label === 'fill');
        const [scatter] = nodesOf(g, 'communication');
        const use = g.getAllNodes().find(n => n.label === 'use');
        expect(reads(scatter, fill)).toBe(true);
        expect(sourcesOf(use).includes(scatter) || sourcesOf(use).some(n => sourcesOf(n).includes(scatter))).toBe(true);
    });

    it('is not there when communications are not shown, and the calls are then joined directly', () => {
        const g = graphOf(events, s => { s.communications = false; });
        expect(nodesOf(g, 'communication')).toHaveLength(0);
        const consume = g.getAllNodes().find(n => n.label === 'consume');
        const produce = g.getAllNodes().find(n => n.label === 'produce');
        expect(sourcesOf(consume).some(n => n === produce || sourcesOf(n).includes(produce))).toBe(true);
    });

    it('serializes what the views show', () => {
        const g = graphOf(events);
        const [comm] = nodesOf(g, 'communication');
        const d = comm.toCytoscape();
        expect(d).toMatchObject({ type: 'communication', operation: 'allgather', bytes_sent: 64, bytes_received: 32, ranks: 4 });
        expect(d.duration).toBe(comm.end - comm.start);
        const tl = comm.toTimeline();
        expect(tl).toMatchObject({ type: 'communication', start: comm.start, end: comm.end, operation: 'allgather' });
        const info = comm.infoData();
        expect(info.Operation).toBe('allgather');
        expect(info.Reason).toMatch(/host access/);
        expect(info).not.toHaveProperty('File');
        expect(info['Bytes sent (rank 0)']).toBe(64);
    });

    it('is between the calls in the graph the views get, and the critical path goes through it', () => {
        const { graph } = buildDisplayGraph(trace(events), { container_allocations: true });
        const comm = nodesOf(graph, 'communication')[0];
        expect(comm.isCriticalPath).toBe(true);
        expect(comm.depth).toBeGreaterThan(graph.getAllNodes().find(n => n.label === 'produce').depth);
        const [cyNodes, cyEdges] = graph.toCytoscape();
        const node = cyNodes.find(n => n.data.type === 'communication');
        expect(node.data.operation).toBe('allgather');
        expect(cyEdges.some(e => e.data.target === node.data.id)).toBe(true);   // (the edges of the views too)
    });

    it('is a setting of the views: show_communications', () => {
        expect(graphSettings({}).communications).toBe(true);       // (a program that has none is not changed by it)
        expect(graphSettings({ show_communications: false }).communications).toBe(false);
        const off = buildDisplayGraph(trace(events), { show_communications: false }).graph;
        expect(nodesOf(off, 'communication')).toHaveLength(0);
    });
});

describe('a collective of a call', () => {
    const b = builder();
    // The collective that gives the results back is recorded before the call (the call is recorded when it ends), but it is after its work
    const make = phase => {
        const bb = builder();
        return [
            bb.alloc(1, 'In'), bb.alloc(2, 'Out'),
            { ...bb.comm([2], { operation: 'alltoallv', reason: 'layout: the results go back to their owners', seq: 7, phase }), },
            { ...bb.call('map', { ins: [1], outs: [2] }), seq: 7, mpi: { rank: 0 } },
            bb.call('later', { ins: [2], outs: [1] }),
        ];
    };

    it('after the work reads what the call wrote', () => {
        const g = graphOf(make('after'));
        const comm = nodesOf(g, 'communication')[0];
        const map = g.getAllNodes().find(n => n.label === 'map');
        expect(comm.totalOrder).toBeGreaterThan(map.totalOrder);
        expect(sourcesOf(comm).some(n => n === map || sourcesOf(n).includes(map))).toBe(true);
        expect(sourcesOf(g.getAllNodes().find(n => n.label === 'later')).includes(comm)).toBe(true);
        expect(comm.callSeq).toBe(7);
        expect(comm.phase).toBe('after');
    });

    it('before the work is read by the call', () => {
        const g = graphOf(make('before'));
        const comm = nodesOf(g, 'communication')[0];
        const map = g.getAllNodes().find(n => n.label === 'map');
        expect(comm.totalOrder).toBeLessThan(map.totalOrder);
        expect(comm.phase).toBe('before');
    });

    it('whose call is not in the trace is still a node', () => {
        const events = make('after').filter(e => e.type !== 'skeleton_call' || e.label !== 'map');
        expect(nodesOf(graphOf(events), 'communication')).toHaveLength(1);
    });
});

describe('the merged trace of an MPI program', () => {
    it('has a node for each collective, with what every rank did', () => {
        const comms = merged.events.filter(e => e.type === 'communication');
        const g = DirectedGraph.fromEvents(merged.events, testSettings());
        const nodes = nodesOf(g, 'communication');
        expect(nodes).toHaveLength(comms.length);
        for (const n of nodes) {
            expect(n.rankSummary).not.toBeNull();
            expect(n.ranks).toBe(3);
            const info = n.infoData();
            expect(info['Bytes moved (all ranks)']).toBe(n.rankSummary.bytes_sent);
            expect(info).toHaveProperty('Longest wait (us)');
            expect(n.toCytoscape().max_wait_us).toBe(n.rankSummary.max_wait_us);
        }
    });

    it('has the collectives that give the results back after the call that made them', () => {
        const g = DirectedGraph.fromEvents(merged.events, testSettings());
        const backs = nodesOf(g, 'communication').filter(n => n.phase === 'after');
        expect(backs.length).toBeGreaterThan(0);
        const isCall = n => n.type === 'skeleton_call' || n.type === 'external';
        for (const back of backs) {
            const call = sourcesOf(back).flatMap(n => isCall(n) ? [n] : sourcesOf(n)).find(isCall);
            expect(call, back.label).toBeDefined();
            expect(back.totalOrder).toBeGreaterThan(call.totalOrder);
        }
    });
});

describe('the example of the splash page (Coulombic on 4 MPI processes)', () => {
    const dir = join(dirname(fileURLToPath(import.meta.url)), '../../examples/coulombic/mpi');
    const files = ['trace_coulombic_mpi.json', 'trace_coulombic_mpi.rank1.json', 'trace_coulombic_mpi.rank2.json', 'trace_coulombic_mpi.rank3.json']
        .map(name => ({ name, text: readFileSync(join(dir, name), 'utf-8') }));
    const m = mergeMpiTraces(files);
    const g = DirectedGraph.fromEvents(m.events, testSettings());

    it('has the print of the result as an external node, which depends on the gather of the result', () => {
        const io = g.getAllNodes().find(n => n.type === 'external');
        expect(io.label).toBe('I/O');
        const comm = sourcesOf(io).find(n => n.type === 'communication');
        expect(comm).toBeDefined();
        expect(comm.operation).toBe('gather');                      // (to the process that runs the code: the others do not need it)
        expect(comm.reason).toMatch(/skepu::external/);
        expect(sourcesOf(comm).some(n => n.type === 'skeleton_call' || sourcesOf(n).some(p => p.type === 'skeleton_call'))).toBe(true);
    });

    it('has the gather of the atoms before the call that reads them at random', () => {
        const call = g.getAllNodes().find(n => n.label === 'columbicPotential');
        const comm = sourcesOf(call).find(n => n.type === 'communication');
        expect(comm.reason).toMatch(/random access/);
        expect(comm.phase).toBe('before');
    });
});

describe('the imbalance of the calls of an MPI program', () => {
    const g = DirectedGraph.fromEvents(merged.events, testSettings());
    const calls = g.getAllNodes().filter(n => n.type === 'skeleton_call');
    const externals = g.getAllNodes().filter(n => n.type === 'external');

    it('is on the nodes the views get, with the slowest rank and the communication', () => {
        for (const n of [...calls, ...externals]) {
            const d = n.toCytoscape();
            if (d.imbalance !== null) expect(d.imbalance).toBeGreaterThanOrEqual(1);       // (null: no work was measured: it took less than a microsecond)
            expect(d.slowest_rank).toBeGreaterThanOrEqual(0);
            expect(d).toHaveProperty('comm_bytes');
        }
        const last = calls.at(-1).toCytoscape();                    // (the MatRow call: 4, 1 and 0 elements)
        expect(last.imbalance).toBeGreaterThan(1.2);
        expect(externals[0].toCytoscape().imbalance).toBeCloseTo(3, 6);               // (the code of an external runs on one rank)
    });

    it('is in the info panel with the work of every rank', () => {
        const info = calls.at(-1).infoData();
        expect(info['Imbalance (longest work / mean over ranks)']).toBeGreaterThan(1.2);
        expect(info['Slowest rank']).toBeGreaterThanOrEqual(0);
        expect(info['Work per rank']).toMatch(/^0: .*; 1: .*; 2: none$/);
        expect(info['Elements per rank']).toBe('0: 4; 1: 1; 2: 0');
    });

    it('is not on a call of a single node', () => {
        const b = builder();
        const single = DirectedGraph.fromEvents(trace([b.alloc(1, 'A'), b.call('produce', { outs: [1] })]), testSettings());
        const call = single.getAllNodes().find(n => n.label === 'produce');
        expect(call.toCytoscape()).not.toHaveProperty('imbalance');
        expect(call.infoData()).not.toHaveProperty('Slowest rank');
    });

    it('colours the nodes: the mode is a measure of the nodes that have it, and the default colour for the others', () => {
        const data = [...calls, ...externals].map(n => n.toCytoscape());
        const provider = new TraceDataProvider({ fetchGraph: () => Promise.reject(new Error('none')), settings: { 'node-color-call': 'imbalance' } });
        const mm = { imbalance: { min: 1, max: 3 }, 'comm-bytes': { min: 0, max: 100 } };
        expect(provider.normFromData('node-color-call', { imbalance: 3 }, mm)).toBe(1);
        expect(provider.normFromData('node-color-call', { imbalance: 2 }, mm)).toBeCloseTo(0.5, 6);
        const degenerate = provider.normFromData('node-color-call', {}, mm);
        expect(degenerate).not.toBe(1);
        provider.settings['node-color-call'] = 'comm-bytes';
        expect(provider.normFromData('node-color-call', { comm_bytes: 50 }, mm)).toBeCloseTo(0.5, 6);
        expect(data.some(d => d.imbalance != null)).toBe(true);
    });
});

describe('the graph of every MPI example', () => {
    const dir = join(dirname(fileURLToPath(import.meta.url)), '../../examples');
    const examples = JSON.parse(readFileSync(join(dir, 'examples.json'), 'utf-8')).filter(ex => ex.trace.mpi);

    const components = (nodes, edges) => {
        const parent = new Map(nodes.map(n => [n.data.id, n.data.id]));
        const find = x => parent.get(x) === x ? x : (parent.set(x, find(parent.get(x))), parent.get(x));
        for (const e of edges)
            if (parent.has(e.data.source) && parent.has(e.data.target)) parent.set(find(e.data.source), find(e.data.target));
        return new Set(nodes.map(n => find(n.data.id))).size;
    };

    it('is in as few pieces as the graph of the same program on a single node (a collective is connected to the call that it is part of)', () => {
        expect(examples.length).toBeGreaterThan(0);
        for (const ex of examples) {
            const m = mergeMpiTraces(ex.trace.mpi.map(rel => ({ name: rel.split('/').pop(), text: readFileSync(join(dir, rel), 'utf-8') })));
            const { graph } = buildDisplayGraph(trace(m.events), {});
            const [nodes, edges] = graph.toCytoscape();
            const single = parseTrace(readFileSync(join(dir, ex.trace.openmp), 'utf-8')).events;
            const [snodes, sedges] = buildDisplayGraph(trace(single), {}).graph.toCytoscape();
            expect(components(nodes, edges), ex.title).toBeLessThanOrEqual(components(snodes, sedges));
        }
    });
});
