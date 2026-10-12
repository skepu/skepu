// test_badges.js — gutter badges of the source panes: a node that stands for several iterations (iterations
// coalesced) has a badge for each iteration at the line where that iteration ran, not all at the line of the
// first one. (backend: node.locations and the 'locations' of the badge data; frontend: collectBadges)
import { describe, it, expect, beforeAll } from 'vitest';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { collectBadges, badgeNodesToShow } from '../../script/badges.js';
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

// n iterations of a region; in each, the skeleton instance `label` is called from the given lines
function loop(lines, n = 2, label = 'sum') {
    const b = builder();
    const events = [b.alloc(1, 'X'), b.alloc(2, 'Y')];
    for (let i = 0; i < n; i++) {
        events.push({ type: 'region', label: 'Iteration', region_depth: 1, time: 0, file: 'prog.cpp', line: 3 });
        for (const line of lines) events.push({ ...b.call(label, { ins: [1], outs: [2] }), region: 'Iteration', line, file: 'prog.cpp' });
    }
    return events;
}
const badgesOf = async (events, opts = {}) => {
    await call('load', { traceJson: JSON.stringify(events), traceFilename: 't.json' });
    const g = await call('graph', { show_regions: true, expand_all_regions: true, collapse_iteration: true, ...opts });
    return g.badge_nodes.map(n => n.data);
};

describe('the badge data of a coalesced node', () => {
    it('has the line of each iteration when they are not all the same', async () => {
        const nodes = await badgesOf(loop([10, 20]));              // one instance called from lines 10 and 20 in each iteration
        const sum = nodes.find(n => n.type === 'skeleton_call');
        expect(sum.total_orders).toHaveLength(4);                  // 2 iterations x 2 call sites, one node
        expect(sum.locations).toEqual([['prog.cpp', 10], ['prog.cpp', 20], ['prog.cpp', 10], ['prog.cpp', 20]]);
    });

    it('has no locations when every iteration ran at the same line', async () => {
        const sum = (await badgesOf(loop([10], 3))).find(n => n.type === 'skeleton_call');
        expect(sum.total_orders).toHaveLength(3);
        expect(sum).not.toHaveProperty('locations');
        expect(sum).toMatchObject({ file: 'prog.cpp', line: 10 });
    });

    it('has none for a node of one iteration', async () => {
        const sum = (await badgesOf(loop([10, 20], 1), { collapse_iteration: false })).filter(n => n.type === 'skeleton_call');
        for (const n of sum) expect(n).not.toHaveProperty('locations');
    });
});

// The backend always has the regions in the graph (the graph view needs them for its layout, and draws them as invisible nodes
// when View > Regions is off): their gutter badges must not show then.
describe('the badges of the regions', () => {
    it('are not shown while the regions are not shown, and the others stay', async () => {
        const nodes = (await badgesOf(loop([10], 2), { collapse_iteration: false })).map(data => ({ data }));
        expect(nodes.some(n => n.data.type === 'region')).toBe(true);               // (the backend sends them)

        const hidden = badgeNodesToShow(nodes, false);
        expect(hidden.some(n => n.data.type === 'region')).toBe(false);
        expect(hidden.filter(n => n.data.type === 'skeleton_call')).toHaveLength(2);
        expect(Object.keys(collectBadges(hidden, []).badges)).toEqual(['prog.cpp:10']);          // (not prog.cpp:3, the line of the region)

        const shown = badgeNodesToShow(nodes, true);
        expect(shown).toHaveLength(nodes.length);
        expect(Object.keys(collectBadges(shown, []).badges).sort()).toEqual(['prog.cpp:10', 'prog.cpp:3']);
    });
});

describe('collectBadges', () => {
    const node = (data) => ({ data: { type: 'skeleton_call', trace_index: 0, ...data } });

    it('puts every iteration at its own line', () => {
        const { badges } = collectBadges([node({ id: 'a', file: 'p.cpp', line: 10, total_orders: [1, 2, 3], trace_indices: [4, 5, 6],
                                                 locations: [['p.cpp', 10], ['p.cpp', 20], ['p.cpp', 10]] })], []);
        expect(Object.keys(badges).sort()).toEqual(['p.cpp:10', 'p.cpp:20']);
        expect(badges['p.cpp:10']).toMatchObject({ count: 2, entries: [{ id: 'a', iterIdx: 0 }, { id: 'a', iterIdx: 2 }] });
        expect(badges['p.cpp:20']).toMatchObject({ count: 1, entries: [{ id: 'a', iterIdx: 1 }] });
    });

    it('counts hidden iterations at the line they ran', () => {
        const { badges } = collectBadges([node({ id: 'a', hidden: true, file: 'p.cpp', line: 10, total_orders: [1, 2], trace_indices: [4, 5],
                                                 locations: [['p.cpp', 10], ['q.cpp', 7]] }),
                                          node({ id: 'b', file: 'q.cpp', line: 7, total_order: 9 })], []);
        expect(badges['p.cpp:10']).toMatchObject({ count: 1, hiddenCount: 1 });
        expect(badges['q.cpp:7']).toMatchObject({ count: 2, hiddenCount: 1, entries: [{ id: 'a', iterIdx: 1 }, { id: 'b', iterIdx: 0 }] });
    });

    it('is what it was for a node without locations: all iterations at its line', () => {
        const { badges, orders } = collectBadges([node({ id: 'a', file: 'p.cpp', line: 5, total_orders: [3, 4, 5], trace_indices: [1, 2, 3] })], []);
        expect(badges['p.cpp:5']).toMatchObject({ count: 3 });
        expect(badges['p.cpp:5'].entries.map(e => e.iterIdx)).toEqual([0, 1, 2]);
        expect(orders.a).toEqual([3, 4, 5]);
    });

    it('leaves out nodes without a source line, and data nodes', () => {
        const { badges } = collectBadges([node({ id: 'a', file: '', line: 4, total_order: 1 }), node({ id: 'b', file: 'p.cpp', line: -1, total_order: 2 }),
                                          node({ id: 'c', type: 'container_update', file: 'p.cpp', line: 3, total_order: 3 })], []);
        expect(badges).toEqual({});
    });

    it('puts a badge in the trace file for each iteration at the line range of its event', () => {
        const ranges = [[1, 4], [5, 8], [9, 12]];
        const { traceBadges } = collectBadges([node({ id: 'a', file: 'p.cpp', line: 5, total_orders: [1, 2], trace_indices: [0, 2], locations: [['p.cpp', 5], ['p.cpp', 6]] })], ranges);
        expect(Object.keys(traceBadges)).toEqual(['0', '2']);
        expect(traceBadges[2].range).toEqual([9, 12]);
    });

    it('gives the order of each iteration, for the arrows between the badges', () => {
        const { orders } = collectBadges([node({ id: 'a', file: 'p.cpp', line: 5, total_orders: [7, 8], trace_indices: [1, 2] }), node({ id: 'b', file: 'p.cpp', line: 6, total_order: 9 })], []);
        expect(orders).toEqual({ a: [7, 8], b: [9] });
    });
});
