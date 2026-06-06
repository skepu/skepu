// helpers.js — Shared fixtures and helpers for the JS test suite.
// Mirrors tests/conftest.py.

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

import { DirectedGraph, GraphSettings } from '../backend-js/graph.js';

const __dirname   = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(__dirname, '../tests/fixtures');
const EXAMPLES_DIR = join(__dirname, '../examples');

// ── File helpers ──────────────────────────────────────────────────────────────

export function loadFixture(name) {
    return JSON.parse(readFileSync(join(FIXTURES_DIR, name), 'utf-8'));
}

export function graphFromFile(path, settings = null) {
    const events = JSON.parse(readFileSync(path, 'utf-8'));
    return DirectedGraph.fromEvents(events, settings);
}

export function graphFromFixture(name, settings = null) {
    return DirectedGraph.fromEvents(loadFixture(name), settings);
}

// ── Named fixture builders ────────────────────────────────────────────────────

export function simpleGraph()    { return graphFromFixture('simple.json'); }
export function mapChainGraph()  { return graphFromFixture('map_chain.json'); }
export function fanoutGraph()    { return graphFromFixture('fanout.json'); }
export function mapReduceGraph() { return graphFromFixture('map_reduce.json'); }

export function fusedChainGraph() {
    const g = mapChainGraph();
    g.findSerialFusions();
    return g;
}

// ── Inline event lists (mirrors conftest.py) ──────────────────────────────────

export const EXTERNAL_EVENTS = [
    { type: 'allocation', label: 'A', object_id: 1001, time: 0, line: 1 },
    { type: 'external', label: 'I/O', start: 0, end: 100,
      file: 'io.cpp', line: 5,
      proxy_inputs: [1001], elwise_inputs: [], scalar_inputs: [], outputs: [] },
];

export const TRANSFER_EVENTS = [
    { type: 'skeleton_call', label: 'compute', pattern: 'Map', elements: [512],
      elwise_inputs: [], proxy_inputs: [], scalar_inputs: [], outputs: [1001],
      backend: 'OpenCL', start: 0, end: 100, file: 'ex.cpp', line: 1 },
    { type: 'transfer', label: 'data', object_id: 1001,
      direction: 'device-to-host', backend: 'OpenCL', start: 100, end: 200 },
];

export const REGION_EVENTS = [
    { type: 'region', label: 'main_loop', region_depth: 0,
      file: 'ex.cpp', line: 10 },
    { type: 'skeleton_call', label: 'map1', pattern: 'Map', elements: [1024],
      elwise_inputs: [], proxy_inputs: [], scalar_inputs: [], outputs: [1001],
      backend: 'OpenMP', start: 100, end: 200,
      file: 'ex.cpp', line: 20, region: 'main_loop' },
];

export const ANTIDEP_EVENTS = [
    { type: 'allocation', label: 'A', object_id: 1001, time: 0, line: 1 },
    { type: 'allocation', label: 'B', object_id: 1002, time: 1, line: 2 },
    { type: 'skeleton_call', label: 'map1', pattern: 'Map', elements: [1024],
      elwise_inputs: [1001], proxy_inputs: [], scalar_inputs: [], outputs: [1002],
      backend: 'OpenMP', start: 100, end: 200, file: 'ex.cpp', line: 10 },
    { type: 'skeleton_call', label: 'map2', pattern: 'Map', elements: [1024],
      elwise_inputs: [1001], proxy_inputs: [], scalar_inputs: [], outputs: [1002],
      backend: 'OpenMP', start: 300, end: 400, file: 'ex.cpp', line: 20 },
];

export const PRNG_EVENTS = [
    { type: 'allocation', label: 'A',   object_id: 1001, time: 0, line: 1 },
    { type: 'allocation', label: 'RNG', object_id: 1002, time: 1, line: 2 },
    { type: 'skeleton_call', label: 'sample', pattern: 'Map', elements: [1024],
      elwise_inputs: [1001], proxy_inputs: [], scalar_inputs: [], outputs: [1001],
      prng: 1002,
      backend: 'OpenMP', start: 100, end: 200, file: 'ex.cpp', line: 10 },
];

export const VIRTUAL_ALIAS_EVENTS = [
    { type: 'virtual_container', virtual_id: 9001,
      label: 'double_buf', object_ids: [1001, 1002] },
    { type: 'skeleton_call', label: 'write_A', pattern: 'Map', elements: [512],
      elwise_inputs: [], proxy_inputs: [], scalar_inputs: [], outputs: [1001],
      backend: 'OpenMP', start: 0, end: 100, file: 'ex.cpp', line: 1 },
    { type: 'skeleton_call', label: 'write_B', pattern: 'Map', elements: [512],
      elwise_inputs: [], proxy_inputs: [], scalar_inputs: [], outputs: [1002],
      backend: 'OpenMP', start: 100, end: 200, file: 'ex.cpp', line: 2 },
];

export function externalGraph()      { return DirectedGraph.fromEvents(EXTERNAL_EVENTS); }
export function transferGraph()      { return DirectedGraph.fromEvents(TRANSFER_EVENTS); }
export function regionGraph()        { return DirectedGraph.fromEvents(REGION_EVENTS); }
export function antidepGraph() {
    const s = new GraphSettings(); s.antideps = true;
    return DirectedGraph.fromEvents(ANTIDEP_EVENTS, s);
}
export function prngGraph()          { return DirectedGraph.fromEvents(PRNG_EVENTS); }
export function virtualAliasGraph()  {
    const g = DirectedGraph.fromEvents(VIRTUAL_ALIAS_EVENTS);
    g.createVirtualAliasEdges(true);
    return g;
}

// ── Example trace collection ──────────────────────────────────────────────────

function walkDir(dir, predicate, results = []) {
    let entries;
    try { entries = readdirSync(dir); } catch { return results; }
    for (const name of entries) {
        const full = join(dir, name);
        try {
            if (statSync(full).isDirectory()) walkDir(full, predicate, results);
            else if (predicate(name)) results.push(full);
        } catch { /* skip unreadable */ }
    }
    return results;
}

export function collectExampleTraces() {
    return walkDir(
        EXAMPLES_DIR,
        name => name.startsWith('trace_') && name.endsWith('.json')
    ).sort();
}

export { DirectedGraph, GraphSettings };
