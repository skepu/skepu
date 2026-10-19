// test_snapshot_transfers.js — a snapshot on a device: it reads the container, so the transfer to the host comes just before the
// snapshot event. The snapshot is of what the transfer moved, so it belongs to the node that made the data, with or without
// the transfer nodes in the graph.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { DirectedGraph, GraphSettings } from '../../script/backend/graph.js';
import { UpdateNode } from '../../script/backend/node.js';
import { parseTrace } from '../../script/backend/trace.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');

function graphOf(events, transfers) {
    const settings = new GraphSettings();
    settings.synthesizeRootRegion = false;
    settings.transfers = transfers;
    return DirectedGraph.fromEvents(events, settings);
}

const withSnapshots = g => g.getAllNodes().filter(n => n instanceof UpdateNode && n.hasSnapshot);

// A map that makes the container 1, which is moved to the host and snapshotted (twice: a second transfer is not needed, the data
// is on the host), then updated again, moved and snapshotted
const events = [
    { type: 'allocation', label: 'A', object_id: 1, time: 1 },
    { type: 'skeleton_call', label: 'k', object_id: 100, pattern: 'Map', backend: 'OpenCL', devices: 1, elements: [4],
      elwise_inputs: [], proxy_inputs: [], scalar_inputs: [], outputs: [1], start: 2, end: 3 },
    { type: 'transfer', label: 'A', object_id: 1, direction: 'device-to-host', backend: 'OpenCL', elements: 4, start: 4, end: 5, time: 5 },
    { type: 'snapshot', label: 'A', object_id: 1, data_file: 'snap/A@1.txt', file: '', line: 0, start: 4, end: 6 },
    { type: 'snapshot', label: 'A', object_id: 1, data_file: 'snap/A@2.txt', file: '', line: 0, start: 7, end: 8 },
    { type: 'skeleton_call', label: 'k', object_id: 100, pattern: 'Map', backend: 'OpenCL', devices: 1, elements: [4],
      elwise_inputs: [1], proxy_inputs: [], scalar_inputs: [], outputs: [1], start: 9, end: 10 },
    { type: 'transfer', label: 'A', object_id: 1, direction: 'device-to-host', backend: 'OpenCL', elements: 4, start: 11, end: 12, time: 12 },
    { type: 'snapshot', label: 'A', object_id: 1, data_file: 'snap/A@3.txt', file: '', line: 0, start: 11, end: 13 },
];

describe('snapshots after a transfer', () => {
    for (const transfers of [true, false]) {
        it(`are attached to the node that made the data (transfers ${transfers ? 'shown' : 'hidden'})`, () => {
            const nodes = withSnapshots(graphOf(events, transfers));
            expect(nodes.length).toBe(2);
            expect(nodes[0].snapshotDataFiles.map(([, f]) => f)).toEqual(['snap/A@1.txt', 'snap/A@2.txt']);
            expect(nodes[1].snapshotDataFiles.map(([, f]) => f)).toEqual(['snap/A@3.txt']);
        });
    }

    for (const example of ['cellular_automaton/trace_cellular_automaton_opencl.json']) {
        it(`are the same with and without the transfers in the trace of ${example}`, () => {
            const trace = parseTrace(readFileSync(join(ROOT, 'examples', example), 'utf-8')).events;
            const shown = withSnapshots(graphOf(trace, true));
            const hidden = withSnapshots(graphOf(trace, false));
            expect(shown.length).toBeGreaterThan(0);
            expect(shown.length).toBe(hidden.length);
            expect(shown.flatMap(n => n.snapshotDataFiles.map(([, f]) => f)))
                .toEqual(hidden.flatMap(n => n.snapshotDataFiles.map(([, f]) => f)));
        });
    }
});
