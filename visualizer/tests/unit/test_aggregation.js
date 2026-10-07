// test_aggregation.js — what iterations are aggregated by (View > Advanced > Aggregate by: Name | Name and
// Location), and the nesting of regions and fusion nodes that the recursive collapse and expand use.
import { describe, it, expect, beforeAll } from 'vitest';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { descendantsOf } from '../../script/nesting.js';
import { builder } from '../event_builders.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const replies = {};
let seq = 0;
async function call(type, payload = {}) {
    const id = String(++seq);
    await globalThis.self.onmessage({ data: { id, type, ...payload } });
    const reply = replies[id];
    if (reply.type === 'error') throw new Error(reply.error);
    return reply;
}
beforeAll(async () => {
    globalThis.self = {
        postMessage: (m) => { replies[m.id] = m; },
        location: { href: 'file://' + join(ROOT, 'script/backend/visualizer.js') },
        addEventListener() {},
    };
    await import(join(ROOT, 'script/backend/visualizer.js'));
});

// n iterations of a region; in each, the skeleton instance `sum` is called from the given lines
function loop(lines, n) {
    const b = builder();
    const events = [b.alloc(1, 'X'), b.alloc(2, 'Y')];
    for (let i = 0; i < n; i++) {
        events.push({ type: 'region', label: 'Iteration', region_depth: 1, time: 0, file: 'prog.cpp', line: 3 });
        for (const line of lines) events.push({ ...b.call('sum', { ins: [1], outs: [2] }), region: 'Iteration', line, file: 'prog.cpp' });
    }
    return events;
}
const graph = async (events, opts) => {
    await call('load', { traceJson: JSON.stringify(events), traceFilename: 't.json' });
    const g = await call('graph', { show_regions: true, expand_all_regions: true, collapse_iteration: true, ...opts });
    const nodes = g.nodes.map(n => n.data ?? n);
    return { ...g, nodes, calls: nodes.filter(n => n.type === 'skeleton_call'), badges: g.badge_nodes.map(n => n.data) };
};

describe('aggregate by name', () => {
    it('is the default and makes one node of an instance called from several lines', async () => {
        const g = await graph(loop([10, 20], 3), {});
        expect(g.calls).toHaveLength(1);
        expect(g.calls[0].iteration_count).toBe(6);
        expect(g.badges.find(b => b.type === 'skeleton_call').locations).toHaveLength(6);
        expect((await graph(loop([10, 20], 3), { aggregate_by: 'name' })).calls).toHaveLength(1);
    });
});

describe('aggregate by name and location', () => {
    it('keeps the call sites apart, each aggregated over the iterations', async () => {
        const g = await graph(loop([10, 20], 3), { aggregate_by: 'name-location' });
        expect(g.calls).toHaveLength(2);
        expect(g.calls.map(c => c.iteration_count)).toEqual([3, 3]);
        const lines = g.badges.filter(b => b.type === 'skeleton_call').map(b => b.line).sort();
        expect(lines).toEqual([10, 20]);
        for (const b of g.badges.filter(b => b.type === 'skeleton_call')) expect(b).not.toHaveProperty('locations');   // all iterations at its own line
    });

    it('does not change what has one call site', async () => {
        const a = await graph(loop([10], 3), { aggregate_by: 'name' });
        const b = await graph(loop([10], 3), { aggregate_by: 'name-location' });
        expect(b.calls.map(c => [c.label, c.iteration_count])).toEqual(a.calls.map(c => [c.label, c.iteration_count]));
        expect(b.nodes.length).toBe(a.nodes.length);
    });

    it('keeps regions of one name apart when they begin at different lines', async () => {
        const b = builder();
        const events = [b.alloc(1, 'X'), b.alloc(2, 'Y')];
        for (const line of [3, 3, 8, 8])
            events.push({ type: 'region', label: 'Phase', region_depth: 1, time: 0, file: 'prog.cpp', line },
                { ...b.call('f', { ins: [1], outs: [2] }), region: 'Phase', line: line + 1, file: 'prog.cpp' });
        const byName = await graph(events, {});
        const byLocation = await graph(events, { aggregate_by: 'name-location' });
        expect(byName.nodes.filter(n => n.type === 'region')).toHaveLength(1);
        expect(byLocation.nodes.filter(n => n.type === 'region')).toHaveLength(2);
    });

    it('does not apply when iterations are not aggregated', async () => {
        const g = await graph(loop([10, 20], 3), { aggregate_by: 'name-location', collapse_iteration: false });
        expect(g.calls).toHaveLength(6);
    });
});

describe('the nesting of regions and fusion nodes', () => {
    it('is reported for every region and fusion node', async () => {
        const b = builder();
        const events = [b.alloc(1, 'X'), b.alloc(2, 'C'), b.alloc(3, 'Y'),
            { type: 'region', label: 'Outer', region_depth: 1, time: 0 },
            { type: 'region', label: 'Inner', region_depth: 2, time: 0, region: 'Outer' },
            { ...b.call('A', { ins: [1], outs: [2] }), region: 'Inner' }, { ...b.call('B', { ins: [2], outs: [3] }), region: 'Inner' }, b.dealloc(2, 'C')];
        const g = await graph(events, { fusion_analysis: true, collapse_iteration: false });
        const outer = g.nodes.find(n => n.label === 'Outer' || n.label.startsWith('Outer')).persistent_region_id;
        const inner = g.nodes.find(n => n.label.startsWith('Inner')).persistent_region_id;
        const fusion = g.all_fusion_pids[0];
        expect(g.region_parents).toEqual({ [outer]: null, [inner]: outer, [fusion]: inner });
        expect(descendantsOf(g.region_parents, outer).sort()).toEqual([fusion, inner].sort());
        expect(descendantsOf(g.region_parents, inner)).toEqual([fusion]);
        expect(descendantsOf(g.region_parents, fusion)).toEqual([]);
    });
});

describe('descendantsOf', () => {
    const parents = { a: null, b: 'a', c: 'a', d: 'b', e: 'd', f: null };
    it('has everything inside at any depth', () => {
        expect(descendantsOf(parents, 'a').sort()).toEqual(['b', 'c', 'd', 'e']);
        expect(descendantsOf(parents, 'b').sort()).toEqual(['d', 'e']);
        expect(descendantsOf(parents, 'e')).toEqual([]);
        expect(descendantsOf(parents, 'x')).toEqual([]);
        expect(descendantsOf(undefined, 'a')).toEqual([]);
    });
});
