// test_graph_keys.js — identifiers of nodes: the keys are the same whenever a graph is made from
// the same events (the ids are random UUIDs), and the region numbers belong to the graph.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseTrace } from '../../script/backend/trace.js';
import { DirectedGraph, GraphSettings } from '../../script/backend/graph.js';
import { graphSettings } from '../../script/backend/pipeline.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const events = parseTrace(readFileSync(join(ROOT, 'examples/dnn_cifar10/trace_dnn_cifar10_opencl.json'), 'utf8')).events;
const all = { container_allocations: true, container_deallocations: true, container_transfers: true, show_scalars: true, show_regions: true };

describe('node keys', () => {
    const keys = g => g.getAllNodes().map(n => n.key);

    it('are unique within a graph', () => {
        const k = keys(DirectedGraph.fromEvents(events, graphSettings(all)));
        expect(new Set(k).size).toBe(k.length);
    });

    it('are the same when the graph is made again, unlike the ids', () => {
        const a = DirectedGraph.fromEvents(events, graphSettings(all));
        const b = DirectedGraph.fromEvents(events, graphSettings(all));
        expect(keys(a)).toEqual(keys(b));
        expect(a.getAllNodes()[3].id).not.toBe(b.getAllNodes()[3].id);
    });

    it('name the event of a node, and the further nodes of an event with a suffix', () => {
        const g = DirectedGraph.fromEvents(events, graphSettings({ ...all }));
        const call = g.getAllNodes().find(n => n.type === 'skeleton_call');
        expect(call.key).toMatch(/^e\d+(\.\d+)?$/);
        expect(call.traceIndex).toBe(Number(call.key.slice(1).split('.')[0]));
        expect(g.getNodeById(call.id)).toBe(call);
    });

    it('name the synthesized root region root', () => {
        const g = DirectedGraph.fromEvents(events, graphSettings({ ...all, mode: 'program-tree' }));
        expect(g.getRootRegion().key).toBe('root');
        expect(keys(g).filter(k => k === 'root')).toHaveLength(1);
    });
});

describe('region numbers', () => {
    const pids = g => g.getAllRegions().map(r => r.persistentRegionId);

    it('count from 1 in every graph, and do not change when another graph is made', () => {
        const a = DirectedGraph.fromEvents(events, graphSettings(all));
        const before = pids(a);
        const b = DirectedGraph.fromEvents(events, graphSettings(all));
        expect(before[0]).toBe('region_1');
        expect(pids(b)).toEqual(before);
        expect(pids(a)).toEqual(before);
    });

    it('are consecutive', () => {
        const p = pids(DirectedGraph.fromEvents(events, graphSettings(all)));
        expect(p).toEqual(p.map((_, i) => 'region_' + (i + 1)));
    });
});
