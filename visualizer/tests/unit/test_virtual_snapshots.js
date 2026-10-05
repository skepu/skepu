// test_virtual_snapshots.js — the snapshots of a virtual data set (the "A" and "B" buffers of
// a double buffered computation) in the node info: the stepper must have all of them, in
// execution order. With the iterations coalesced and the regions shown, the container A is
// updated by two different nodes (its initialisation, outside the loop's region, and the
// updates in the loop) and its snapshots are spread over both.
import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT  = join(dirname(fileURLToPath(import.meta.url)), '../..');
const TRACE = join(ROOT, 'examples/cellular_automaton_skepy/trace_cellular_automaton_openmp.json');

// The backend is a web worker: it talks through self.postMessage and self.onmessage.
const replies = {};
let seq = 0;
async function call(type, payload = {}) {
    const id = String(++seq);
    await globalThis.self.onmessage({ data: { id, type, ...payload } });
    const reply = replies[id];
    if (reply.type === 'error') throw new Error(reply.error);
    return reply;
}

const events = JSON.parse(readFileSync(TRACE, 'utf-8'));
const snapshotEvents = events.filter(e => e.type === 'snapshot');
// (the contents are not needed, only that every snapshot file is there: its name will do)
const snapshotFiles = Object.fromEntries(
    snapshotEvents.map(e => [basename(e.data_file), 'contents of ' + basename(e.data_file)]));

beforeAll(async () => {
    globalThis.self = {
        postMessage: (m) => { replies[m.id] = m; },
        location: { href: 'file://' + join(ROOT, 'script/backend/visualizer.js') },
        addEventListener() {},
    };
    await import(join(ROOT, 'script/backend/visualizer.js'));
    await call('load', { traceJson: readFileSync(TRACE, 'utf-8'), traceFilename: 'trace.json',
                         cppFiles: {}, snapshotFiles });
});

// The info of the update nodes of the data set that stand for several iterations.
async function aggregatedInfo(settings) {
    const graph = await call('graph', { collapse_iteration: true, show_virtual_alias: true, ...settings });
    const infos = [];
    for (const node of graph.nodes) {
        const data = node.data ?? node;
        if ((data.iteration_count ?? 1) > 1) {
            const info = await call('getData', { nodeId: data.id });
            if (info.content_snapshot_labels) infos.push(info);
        }
    }
    return infos;
}

// A@1 B@1 A@2 B@2 ...: the snapshots in the order of the run (the trace is in that order too)
function expectedLabels() {
    const count = {};
    return snapshotEvents.map(e => `${e.label}@${count[e.label] = (count[e.label] ?? 0) + 1}`);
}

describe('TestVirtualDatasetSnapshots', () => {

    it('the trace has the snapshots of both containers', () => {
        expect(snapshotEvents.length).toBeGreaterThan(4);
        expect(new Set(snapshotEvents.map(e => e.label)).size).toBe(2);
    });

    for (const [name, settings] of [
        ['iterations coalesced', {}],
        ['regions shown', { show_regions: true, expand_all_regions: true }],
    ]) {
        it(`the stepper has every snapshot in order (${name})`, async () => {
            const infos = await aggregatedInfo(settings);
            expect(infos.length).toBeGreaterThan(0);
            for (const info of infos) {
                expect(info.content_snapshot_virtual_label).toBe('Domain');
                expect(info.content_snapshot_labels).toEqual(expectedLabels());
                expect(info.content_snapshots).toEqual(
                    snapshotEvents.map(e => 'contents of ' + basename(e.data_file)));
            }
        });
    }
});
