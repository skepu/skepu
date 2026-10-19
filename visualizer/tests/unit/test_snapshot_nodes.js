// test_snapshot_nodes.js — the snapshots of containers as nodes of the graph ("yellow stars"): a node for each snapshot event if the snapshots are
// shown, hanging on the node that made the data, with an entry in the timeline, a badge at the line of the call that took it, and the contents of
// the snapshot in its info. (The snapshot is a part of the node that made the data too, shown or not: test_snapshot_transfers.js.)
import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

import { DirectedGraph, GraphSettings } from '../../script/backend/graph.js';
import { SnapshotNode, UpdateNode } from '../../script/backend/node.js';
import { parseTrace } from '../../script/backend/trace.js';
import { collectBadges } from '../../script/badges.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');

const events = [
    { type: 'allocation', label: 'A', object_id: 1, time: 1 },
    { type: 'region', label: 'Step', region_depth: 1, time: 2, file: '/p/prog.cpp', line: 7 },
    { type: 'skeleton_call', label: 'k', object_id: 100, pattern: 'Map', backend: 'CPU', elements: [4], region: 'Step',
      elwise_inputs: [], proxy_inputs: [], scalar_inputs: [], outputs: [1], start: 3, end: 4, file: '/p/prog.cpp', line: 9 },
    { type: 'snapshot', label: 'A', object_id: 1, data_file: 'snap/A@1.txt', file: '/p/prog.cpp', line: 10, start: 5, end: 6, region: 'Step' },
    { type: 'region', label: 'Step', region_depth: 1, time: 7, file: '/p/prog.cpp', line: 7 },
    { type: 'skeleton_call', label: 'k', object_id: 100, pattern: 'Map', backend: 'CPU', elements: [4], region: 'Step',
      elwise_inputs: [1], proxy_inputs: [], scalar_inputs: [], outputs: [1], start: 8, end: 9, file: '/p/prog.cpp', line: 9 },
    { type: 'snapshot', label: 'A', object_id: 1, data_file: 'snap/A@2.txt', file: '/p/prog.cpp', line: 10, start: 10, end: 11, region: 'Step' },
];

function graph(snapshots, extra = {}) {
    const s = new GraphSettings();
    s.snapshots = snapshots;
    Object.assign(s, extra);
    return DirectedGraph.fromEvents(events, s);
}
const snapshotNodes = g => g.getAllNodes().filter(n => n instanceof SnapshotNode);

describe('the snapshot nodes', () => {
    it('are there only if the snapshots are shown', () => {
        expect(snapshotNodes(graph(true)).length).toBe(2);
        expect(snapshotNodes(graph(false)).length).toBe(0);
        // (the data of the update nodes has them either way)
        expect(graph(false).getAllNodes().filter(n => n instanceof UpdateNode && n.hasSnapshot).length).toBe(2);
    });

    it('hang on the node that made the container, and are in the region of the call that took them', () => {
        const [first] = snapshotNodes(graph(true));
        expect(first.getParentNodes().map(n => n.label)).toEqual(['A']);
        expect(first.getParentNodes()[0].type).toBe('container_update');
        expect(first.region?.label).toBe('Step');
    });

    it('have the place of the call in the node, the timeline and the badge', () => {
        const [first] = snapshotNodes(graph(true));
        const node = first.toCytoscape();
        expect(node).toMatchObject({ type: 'snapshot', label: 'A', file: '/p/prog.cpp', line: 10, data_file: 'snap/A@1.txt' });
        expect(first.toTimeline()).toMatchObject({ type: 'snapshot', label: 'A', timestamp: 5 });
        expect(first.toSourceBadge()).toMatchObject({ type: 'snapshot', file: '/p/prog.cpp', line: 10 });
    });

    it('have a badge at their line, which the viewer places like that of a call', () => {
        const g = graph(true);
        const badgeNodes = g.toTimelineAndBadges([], true, [])[1];
        const badges = collectBadges(badgeNodes, []).badges;
        expect(badges['/p/prog.cpp:10']).toMatchObject({ count: 2, onlySnapshots: true });
        expect(badges['/p/prog.cpp:9']).toMatchObject({ count: 2, onlySnapshots: false });
    });

    it('are one node with the iterations together when the iterations are coalesced', () => {
        const g = graph(true);
        g.coalesceIterations();
        const nodes = snapshotNodes(g);
        expect(nodes.length).toBe(1);
        expect(nodes[0].iterationCount).toBe(2);
        expect(nodes[0].snapshotDataFiles.map(([, f]) => f)).toEqual(['snap/A@1.txt', 'snap/A@2.txt']);
    });
});

// The contents of a snapshot are in the info of its node (the backend is a web worker)
describe('the info of a snapshot node', () => {
    const replies = {};
    let seq = 0;
    async function call(type, payload = {}) {
        const id = String(++seq);
        await globalThis.self.onmessage({ data: { id, type, ...payload } });
        const reply = replies[id];
        if (reply.type === 'error') throw new Error(reply.error);
        return reply;
    }
    let snapshotEvents = [];
    const TRACE = join(ROOT, 'examples/cellular_automaton/trace_cellular_automaton_openmp.json');

    beforeAll(async () => {
        globalThis.self = {
            postMessage: (m) => { replies[m.id] = m; },
            location: { href: 'file://' + join(ROOT, 'script/backend/visualizer.js') },
            addEventListener() {},
        };
        await import(join(ROOT, 'script/backend/visualizer.js'));
        const parsed = parseTrace(readFileSync(TRACE, 'utf-8')).events;
        snapshotEvents = parsed.filter(e => e.type === 'snapshot');
        const snapshotFiles = Object.fromEntries(snapshotEvents.map(e => [basename(e.data_file), 'contents of ' + basename(e.data_file)]));
        await call('load', { traceJson: readFileSync(TRACE, 'utf-8'), traceFilename: 'trace.json', cppFiles: {}, snapshotFiles });
    });

    it('has the contents of its snapshot, and the graph has a star for each snapshot', async () => {
        const graph = await call('graph', { container_snapshots: true });
        const stars = graph.nodes.map(n => n.data ?? n).filter(d => d.type === 'snapshot');
        expect(snapshotEvents.length).toBeGreaterThan(0);
        expect(stars.length).toBe(snapshotEvents.length);
        const info = await call('getData', { nodeId: stars[0].id });
        expect(info.content_snapshots).toEqual(['contents of ' + basename(stars[0].data_file)]);
        expect(graph.timeline_nodes.filter(t => t.data.type === 'snapshot').length).toBe(snapshotEvents.length);
    });

    it('is not in the graph when the snapshots are not shown', async () => {
        const graph = await call('graph', { container_snapshots: false });
        expect(graph.nodes.map(n => n.data ?? n).filter(d => d.type === 'snapshot').length).toBe(0);
    });
});
