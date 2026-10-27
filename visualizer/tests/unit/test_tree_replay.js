// test_tree_replay.js — program tree mode: visiting the tree depth first, children in the order their edges are drawn, replays the trace. A nested
// region ends the chain of events in its parent, so the events after it start a new chain. The root of the tree is never collapsed.
import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const examples = JSON.parse(readFileSync(join(ROOT, 'examples/examples.json'), 'utf-8'));

const replies = {};
let seq = 0;
async function call(type, payload = {}) {
    const id = String(++seq);
    await globalThis.self.onmessage({ data: { id, type, ...payload } });
    const reply = replies[id];
    if (reply.type === 'error') throw new Error(reply.error);
    return reply;
}

function tree(graph) {
    const byId = new Map(graph.nodes.map(({ data }) => [data.id, data]));
    const children = new Map();
    for (const { data } of graph.edges) children.set(data.source, [...(children.get(data.source) ?? []), data.target]);
    const hasParent = new Set(graph.edges.map(({ data }) => data.target));
    const roots = graph.nodes.filter(({ data }) => !hasParent.has(data.id)).map(({ data }) => data.id);
    return { byId, children, roots };
}

// where a node is in the order of the trace (the first, for a node that iterations are merged into; for a trace of MPI ranks it is the order of the merged trace)
const first = data => Math.min(...(data.total_orders?.length ? data.total_orders : [data.total_order]));

beforeAll(async () => {
    globalThis.self = {
        postMessage: m => { replies[m.id] = m; },
        location: { href: 'file://' + join(ROOT, 'script/backend/visualizer.js') },
        addEventListener() {},
    };
    await import(join(ROOT, 'script/backend/visualizer.js?tree-replay'));
});

for (const example of examples) {
    for (const [backend, entry] of Object.entries(example.trace)) {
        const files = Array.isArray(entry) ? entry : [entry];
        if (!existsSync(join(ROOT, 'examples', files[0]))) continue;
        it(`is the trace when it is visited depth first (${example.title}, ${backend})`, async () => {
            if (Array.isArray(entry))
                await call('load', { traceFiles: entry.map(rel => ({ name: rel.split('/').pop(), text: readFileSync(join(ROOT, 'examples', rel), 'utf-8') })) });
            else
                await call('load', { traceJson: readFileSync(join(ROOT, 'examples', entry), 'utf-8'), traceFilename: 't.json', cppFiles: {}, snapshotFiles: {} });
            const graph = await call('graph', { mode: 'program-tree', show_regions: true, expand_all_regions: true, collapse_iteration: false });
            const { byId, children, roots } = tree(graph);
            expect(roots).toHaveLength(1);
            const order = [];
            const visit = id => { order.push(first(byId.get(id))); for (const child of children.get(id) ?? []) visit(child); };
            visit(roots[0]);
            expect(order).toEqual([...order].sort((a, b) => a - b));
        });
    }
}

describe('the tree', () => {
    it('has a chain of its own for the events after a region (Horner)', async () => {
        await call('load', { traceJson: readFileSync(join(ROOT, 'examples/horner/trace_horner_openmp.json'), 'utf-8'), traceFilename: 't.json', cppFiles: {}, snapshotFiles: {} });
        const graph = await call('graph', { mode: 'program-tree', show_regions: true, expand_all_regions: true, collapse_iteration: false });
        const { byId, children, roots } = tree(graph);
        const labels = id => (children.get(id) ?? []).map(child => byId.get(child).label);
        // the chains (an event, and what follows it) and the regions, in the order of the program
        expect(labels(roots[0])).toEqual(['Read coeffs', 'generate', 'I/O', 'horner_eval', 'I/O']);
        // (each chain is one event here: no event follows another before a region)
        for (const id of children.get(roots[0])) if (byId.get(id).type !== 'tree-region') expect(children.get(id)?.length ?? 0).toBeLessThanOrEqual(1);
    });

    it('never has its root collapsed, whatever is asked', async () => {
        await call('load', { traceJson: readFileSync(join(ROOT, 'examples/horner/trace_horner_openmp.json'), 'utf-8'), traceFilename: 't.json', cppFiles: {}, snapshotFiles: {} });
        // (nothing is expanded: every region but the root is collapsed)
        const graph = await call('graph', { mode: 'program-tree', show_regions: true, expand_all_regions: false, expanded_regions: '' });
        const root = graph.nodes.find(({ data }) => data.persistent_region_id === 'region_global').data;
        expect(root.collapsed).toBeFalsy();
        const { children, roots } = tree(graph);
        expect(roots).toEqual([root.id]);
        expect(children.get(root.id).length).toBeGreaterThan(0);
    });
});
