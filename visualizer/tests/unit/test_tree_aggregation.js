// test_tree_aggregation.js — the program tree mode merges the iterations of a loop into the same nodes as the dependence DAG does (what differs is the root
// region, the edges, and how the nodes are shown): the merged nodes, each as its type, label and the trace events that it joins, are the same set. A node that
// is under different regions is not merged with another (the nested regions of a region belong to it only: the root region does not have them too).
import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const examples = JSON.parse(readFileSync(join(ROOT, 'examples/examples.json'), 'utf-8'));
const VARIANTS = {
    defaults:    {},
    data:        { container_allocations: true, container_deallocations: true, show_scalars: true },
    transfers:   { container_allocations: true, container_deallocations: true, container_transfers: true, show_scalars: true, anti_deps: true },
    by_location: { aggregate_by: 'name-location', container_allocations: true, show_scalars: true },
    fusion:      { fusion_analysis: true, container_allocations: true },
};

const replies = {};
let seq = 0;
async function call(type, payload = {}) {
    const id = String(++seq);
    await globalThis.self.onmessage({ data: { id, type, ...payload } });
    const reply = replies[id];
    if (reply.type === 'error') throw new Error(reply.error);
    return reply;
}

// the merged nodes: "type|label|the trace events it joins", without the root region of the tree and the numbering of the regions of one label
function nodes(graph) {
    const found = new Map();
    for (const { data } of graph.nodes) {
        if (data.label === '[global]' && data.type === 'tree-region') continue;
        const events = (data.trace_indices?.length ? data.trace_indices : [data.trace_index]).slice().sort((a, b) => a - b);
        const key = `${data.type === 'tree-region' ? 'region' : data.type}|${String(data.label).replace(/[₀-₉]/g, '')}|${events.join(',')}`;
        found.set(key, (found.get(key) ?? 0) + 1);
    }
    return found;
}

beforeAll(async () => {
    globalThis.self = {
        postMessage: m => { replies[m.id] = m; },
        location: { href: 'file://' + join(ROOT, 'script/backend/visualizer.js') },
        addEventListener() {},
    };
    await import(join(ROOT, 'script/backend/visualizer.js?tree-aggregation'));
});

for (const example of examples) {
    for (const [backend, entry] of Object.entries(example.trace)) {
        const files = Array.isArray(entry) ? entry : [entry];
        if (!existsSync(join(ROOT, 'examples', files[0]))) continue;
        describe(`${example.title}, ${backend}`, () => {
            for (const [name, variant] of Object.entries(VARIANTS)) {
                it(`merges the same nodes in both modes (${name})`, async () => {
                    if (Array.isArray(entry))
                        await call('load', { traceFiles: entry.map(rel => ({ name: rel.split('/').pop(), text: readFileSync(join(ROOT, 'examples', rel), 'utf-8') })) });
                    else
                        await call('load', { traceJson: readFileSync(join(ROOT, 'examples', entry), 'utf-8'), traceFilename: 't.json', cppFiles: {}, snapshotFiles: {} });
                    const settings = { show_regions: true, expand_all_regions: true, collapse_iteration: true, ...variant };
                    const dag = nodes(await call('graph', { ...settings, mode: 'dependence-dag' }));
                    const tree = nodes(await call('graph', { ...settings, mode: 'program-tree' }));
                    expect([...tree].sort()).toEqual([...dag].sort());
                });
            }
        });
    }
}

describe('the tree', () => {
    it('has one parent for each node, and the regions in the regions that they are in', async () => {
        await call('load', { traceJson: readFileSync(join(ROOT, 'examples/horner/trace_horner_openmp.json'), 'utf-8'), traceFilename: 't.json', cppFiles: {}, snapshotFiles: {} });
        const graph = await call('graph', { mode: 'program-tree', show_regions: true, expand_all_regions: true, collapse_iteration: true, container_allocations: true });
        const parents = new Map();
        for (const { data } of graph.edges) parents.set(data.target, [...(parents.get(data.target) ?? []), data.source]);
        for (const list of parents.values()) expect(list).toHaveLength(1);
        const label = new Map(graph.nodes.map(({ data }) => [data.id, data.label]));
        const roots = graph.nodes.filter(({ data }) => !parents.has(data.id));
        expect(roots.map(({ data }) => data.label)).toEqual(['[global]']);
        const parentOf = name => label.get(parents.get(graph.nodes.find(({ data }) => data.label === name).data.id)[0]);
        expect(parentOf('horner_eval')).toBe('[global]');
        expect(parentOf('Iteration')).toBe('horner_eval');
        expect(parentOf('muladd')).toBe('Iteration');   // (the first event of the chain of the region: the region itself is above it)
    });
});

describe('the ids of the regions', () => {
    // what the viewer keeps whether a region is collapsed by: the same region has the same id in both modes (and with the iterations merged or not)
    for (const rel of ['examples/horner/trace_horner_openmp.json', 'examples/ode_lorenz/trace_ode_lorenz_openmp.json', 'examples/pso/trace_pso_openmp.json']) {
        it(`are the same in the dependence graph and the program tree (${rel.split('/')[1]})`, async () => {
            await call('load', { traceJson: readFileSync(join(ROOT, rel), 'utf-8'), traceFilename: 't.json', cppFiles: {}, snapshotFiles: {} });
            for (const collapse_iteration of [false, true]) {
                const regions = async mode => {
                    const graph = await call('graph', { mode, show_regions: true, expand_all_regions: true, collapse_iteration });
                    return new Map(graph.nodes.filter(({ data }) => data.persistent_region_id).map(({ data }) => [data.persistent_region_id, String(data.label).replace(/[₀-₉]/g, '')]));
                };
                const dag = await regions('dependence-dag'), tree = await regions('program-tree');
                expect(tree.get('region_global')).toBe('[global]');
                expect(dag.has('region_global')).toBe(false);
                tree.delete('region_global');
                expect([...tree].sort()).toEqual([...dag].sort());
            }
        });
    }

    it('of the regions that iterations are merged into are all in the node', async () => {
        await call('load', { traceJson: readFileSync(join(ROOT, 'examples/horner/trace_horner_openmp.json'), 'utf-8'), traceFilename: 't.json', cppFiles: {}, snapshotFiles: {} });
        const iterations = async collapse_iteration => (await call('graph', { show_regions: true, expand_all_regions: true, collapse_iteration })).nodes
            .filter(({ data }) => String(data.label).startsWith('Iteration')).map(({ data }) => data);
        const merged = await iterations(true), separate = await iterations(false);
        expect(merged).toHaveLength(1);
        expect(merged[0].persistent_region_ids.slice().sort()).toEqual(separate.map(d => d.persistent_region_id).sort());
        expect(merged[0].persistent_region_id).toBe(separate[0].persistent_region_id);   // (the first of them is the node)
    });
});
