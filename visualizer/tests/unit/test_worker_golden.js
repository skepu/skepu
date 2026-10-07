// test_worker_golden.js — characterization of the backend worker: the result of its 'graph'
// message on the shipped example traces with several settings must not change by accident.
// (Safety net for restructuring the backend; the ids of nodes and edges are random UUIDs and
// are replaced by their order of appearance.)
//
// Per case a file in tests/golden/worker/ has the counts and a hash of each part of the result,
// so that a difference says which part changed. After an intended change of the output:
//   UPDATE_GOLDEN=1 npm test
import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT   = join(dirname(fileURLToPath(import.meta.url)), '../..');
const GOLDEN = join(ROOT, 'tests/golden/worker');
const UPDATE = !!process.env.UPDATE_GOLDEN;

const TRACES = {
    dotproduct:    'examples/dotproduct/trace_dotproduct_openmp.json',
    horner:        'examples/horner/trace_horner_openmp.json',
    coulombic:     'examples/coulombic/trace_coulombic_opencl.json',
    heat:          'examples/heat_diffusion/trace_heat_diffusion_openmp.json',
    cellular:      'examples/cellular_automaton/trace_cellular_automaton_openmp.json',
    cellular_py:   'examples/cellular_automaton_skepy/trace_cellular_automaton_openmp.json',
    dnn_cifar10:   'examples/dnn_cifar10/trace_dnn_cifar10_opencl.json',
};

const ALL = { container_allocations: true, container_deallocations: true, container_transfers: true,
              show_scalars: true, anti_deps: true, show_regions: true, expand_all_regions: true, show_virtual_alias: true };
const SETTINGS = {
    defaults: {},
    full:     ALL,
    fusion:   { fusion_analysis: true, show_regions: true, expand_all_regions: true },
    fusion_collapse: { fusion_analysis: true, collapse_iteration: true, show_regions: true, expand_all_regions: true },
    collapse: { collapse_iteration: true, show_regions: true, show_virtual_alias: true },
    tree:     { mode: 'program-tree', show_regions: true },
    edges:    { data_as_edges: true, container_transfers: true },
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

// Replace every UUID by the order of its first appearance.
function normalize(value) {
    const ids = new Map();
    return JSON.stringify(value).replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/g,
        id => { if (!ids.has(id)) ids.set(id, 'id' + ids.size); return ids.get(id); });
}

const PARTS = ['nodes', 'edges', 'timeline_nodes', 'badge_nodes', 'fusion_hints', 'all_region_pids'];
const sha = text => createHash('sha256').update(text).digest('hex').slice(0, 16);

function digest(graph) {
    const normalized = JSON.parse(normalize(Object.fromEntries(PARTS.map(p => [p, graph[p]]))));
    return {
        counts: { nodes: graph.nodes.length, edges: graph.edges.length, timeline: graph.timeline_nodes.length,
                  badges: graph.badge_nodes.length, fusion_hints: graph.fusion_hints.length,
                  events: graph.event_count, snapshots: graph.snapshot_count },
        hashes: Object.fromEntries(PARTS.map(p => [p, sha(JSON.stringify(normalized[p]))])),
    };
}

beforeAll(async () => {
    globalThis.self = {
        postMessage: (m) => { replies[m.id] = m; },
        location: { href: 'file://' + join(ROOT, 'script/backend/visualizer.js') },
        addEventListener() {},
    };
    await import(join(ROOT, 'script/backend/visualizer.js'));
});

for (const [traceName, rel] of Object.entries(TRACES)) {
    describe(`worker graph: ${traceName}`, () => {
        const text = readFileSync(join(ROOT, rel), 'utf-8');
        const snapshotFiles = Object.fromEntries(JSON.parse(text).filter?.(e => e.type === 'snapshot')
            .map(e => [basename(e.data_file), 'contents of ' + basename(e.data_file)]) ?? []);

        for (const [settingsName, settings] of Object.entries(SETTINGS)) {
            it(settingsName, async () => {
                await call('load', { traceJson: text, traceFilename: 'trace.json', cppFiles: {}, snapshotFiles });
                const got  = digest(await call('graph', settings));
                const file = join(GOLDEN, `${traceName}.${settingsName}.json`);
                if (UPDATE || !existsSync(file)) {
                    if (!UPDATE) throw new Error(`no golden file ${file}; run UPDATE_GOLDEN=1 npm test`);
                    writeFileSync(file, JSON.stringify(got, null, 2) + '\n');
                    return;
                }
                expect(got).toEqual(JSON.parse(readFileSync(file, 'utf-8')));
            });
        }
    });
}
