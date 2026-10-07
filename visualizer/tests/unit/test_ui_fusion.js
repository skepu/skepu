// test_ui_fusion.js — what the views get of the fusion analysis: compound 'fusion' nodes around the
// calls that fusion.js finds fusable, and for the other calls the reason why not (in the node info).
import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
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
const data = n => n.data ?? n;
const graph = async (events, opts) => {
    await call('load', { traceJson: JSON.stringify(events), traceFilename: 't.json' });
    const g = await call('graph', opts);
    const nodes = g.nodes.map(data);
    return { ...g, nodes, byLabel: l => nodes.filter(n => n.label === l), fusions: nodes.filter(n => n.type === 'fusion') };
};

beforeAll(async () => {
    globalThis.self = {
        postMessage: (m) => { replies[m.id] = m; },
        location: { href: 'file://' + join(ROOT, 'script/backend/visualizer.js') },
        addEventListener() {},
    };
    await import(join(ROOT, 'script/backend/visualizer.js'));
});

// X -> A -> C -> B -> Y, with C freed after B
function chain(extra = () => []) {
    const b = builder();
    return [b.alloc(1, 'X'), b.alloc(2, 'C'), b.alloc(3, 'Y'),
            b.call('A', { ins: [1], outs: [2] }), ...extra(b), b.call('B', { ins: [2], outs: [3] }), b.dealloc(2, 'C')];
}

describe('fusion nodes in the graph', () => {
    it('are drawn around the fused calls and the intermediate container', async () => {
        const g = await graph(chain(), { fusion_analysis: true });
        expect(g.fusions).toHaveLength(1);
        const f = g.fusions[0];
        const members = g.nodes.filter(n => n.parent === f.id).map(n => n.label).sort();
        expect(members).toEqual(['A', 'B', 'C']);                        // the two calls and the version of C that A writes
        expect(f.fusion_pattern).toBe('Map');
        expect(g.fusion_hints).toHaveLength(1);
    });

    it('are absent without the fusion analysis', async () => {
        const g = await graph(chain(), {});
        expect(g.fusions).toHaveLength(0);
        expect(g.fusion_hints).toHaveLength(0);
    });

    it('have an info pane that says what is fused and what is saved', async () => {
        const g = await graph(chain(), { fusion_analysis: true });
        const info = await call('getData', { nodeId: g.fusions[0].id });
        expect(info).toMatchObject({ 'Fusion kind': expect.stringContaining('serial'), Calls: 'A, B', 'Not written or read': 'C', 'Launches saved': 1 });
    });

    it('are made for independent calls too, with their element counts', async () => {
        const b = builder();
        const events = [b.alloc(1, 'X'), b.alloc(2, 'P'), b.alloc(3, 'Q'), b.call('M1', { ins: [1], outs: [2] }), b.call('M2', { ins: [1], outs: [3] })];
        const g = await graph(events, { fusion_analysis: true });
        expect(g.fusions).toHaveLength(1);
        expect(g.fusions[0]).toMatchObject({ uniform_elements: true });
        expect((await call('getData', { nodeId: g.fusions[0].id }))['Fusion kind']).toContain('parallel');
    });

    it('are not made across regions, which they would take the calls out of', async () => {
        const b = builder();
        const events = [b.alloc(1, 'X'), b.alloc(2, 'C'), b.alloc(3, 'Y'),
            { type: 'region', label: 'R1', region_depth: 1, time: 0 }, { ...b.call('A', { ins: [1], outs: [2] }), region: 'R1' },
            { type: 'region', label: 'R2', region_depth: 1, time: 0 }, { ...b.call('B', { ins: [2], outs: [3] }), region: 'R2' }];
        const g = await graph(events, { fusion_analysis: true, show_regions: true, expand_all_regions: true });
        expect(g.fusions).toHaveLength(0);
    });
});

describe('collapsing a fusion node', () => {
    const run = async (extra = {}) => {
        const g = await graph(chain(b => [b.call('Q', { ins: [1], outs: [4] })]), { fusion_analysis: true, ...extra });
        return g;
    };
    it('names the fusion nodes and offers their ids', async () => {
        const g = await run();
        expect(g.all_fusion_pids).toHaveLength(1);
        expect(g.fusions[0].persistent_fusion_id).toBe(g.all_fusion_pids[0]);
        expect(g.fusions[0].collapsed).toBeUndefined();               // they start expanded
    });
    it('hides the calls of a collapsed fusion node and keeps the edges that leave it', async () => {
        const open = await run();
        const pid = open.all_fusion_pids[0];
        const g = await run({ collapsed_fusions: pid });
        const f = g.fusions[0];
        expect(f.collapsed).toBe(true);
        const ids = new Set(g.nodes.map(n => n.id));
        expect(g.nodes.filter(n => n.parent === f.id)).toHaveLength(0);              // the members are not in the graph
        expect(g.nodes.map(n => n.label)).not.toContain('B');
        for (const e of g.edges.map(data)) { expect(ids.has(e.source)).toBe(true); expect(ids.has(e.target)).toBe(true); }
        expect(g.edges.map(data).some(e => e.source === f.id || e.target === f.id)).toBe(true);
        expect(g.timeline_nodes.filter(t => t.data.hidden).length).toBeGreaterThan(0);
        expect(g.node_count ?? g.nodes.length).toBeLessThan(open.nodes.length);
    });
    it('is the same graph as before when the fusion node is expanded again', async () => {
        const open = await run();
        const again = await run({ collapsed_fusions: '' });
        expect(again.nodes.map(n => n.label)).toEqual(open.nodes.map(n => n.label));
    });
    it('keeps the id of a group when the regions are shown or not', async () => {
        const a = await run();
        const b = await run({ show_regions: true, expand_all_regions: true });
        expect(b.all_fusion_pids).toEqual(a.all_fusion_pids);
    });
    it('collapses the fusion nodes of a loop that are one node', async () => {
        const b = builder();
        const events = [b.alloc(1, 'X'), b.alloc(2, 'C'), b.alloc(3, 'Y')];
        for (let i = 0; i < 3; i++)
            events.push({ type: 'region', label: 'Iteration', region_depth: 1, time: 0 },
                { ...b.call('A', { ins: [1], outs: [2] }), region: 'Iteration', line: 5 },
                { ...b.call('B', { ins: [2], outs: [3] }), region: 'Iteration', line: 6 }, b.dealloc(2, 'C'));
        const opts = { fusion_analysis: true, show_regions: true, expand_all_regions: true, collapse_iteration: true };
        const open = await graph(events, opts);
        const g = await graph(events, { ...opts, collapsed_fusions: open.all_fusion_pids.join(',') });
        expect(g.fusions[0].collapsed).toBe(true);
        expect(g.nodes.some(n => n.parent === g.fusions[0].id)).toBe(false);
    });
});

describe('the reasons for not fusing', () => {
    it('are in the info of the producer', async () => {
        const events = chain(b => [b.call('Q', { ins: [2], outs: [4] })]);   // a second reader of C
        const g = await graph(events, { fusion_analysis: true });
        const a = g.byLabel('A')[0];
        const info = await call('getData', { nodeId: a.id });
        expect(info['Not fused']).toContain('"C" is read by other calls too');
    });

    it('are not there without the analysis', async () => {
        const g = await graph(chain(b => [b.call('Q', { ins: [2], outs: [4] })]), {});
        expect(await call('getData', { nodeId: g.byLabel('A')[0].id })).not.toHaveProperty('Not fused');
    });
});

describe('iterations of a loop', () => {
    function loop(n) {
        const b = builder();
        const events = [b.alloc(1, 'X'), b.alloc(2, 'C'), b.alloc(3, 'Y')];
        for (let i = 0; i < n; i++)
            events.push({ type: 'region', label: 'Iteration', region_depth: 1, time: 0 },
                { ...b.call('A', { ins: [1], outs: [2] }), region: 'Iteration', line: 5 },
                { ...b.call('B', { ins: [2], outs: [3] }), region: 'Iteration', line: 6 }, b.dealloc(2, 'C'));
        return events;
    }
    it('give one fusion node each, or one in all when coalesced', async () => {
        const opts = { fusion_analysis: true, show_regions: true, expand_all_regions: true };
        expect((await graph(loop(3), opts)).fusions).toHaveLength(3);
        const c = await graph(loop(3), { ...opts, collapse_iteration: true });
        expect(c.fusions).toHaveLength(1);
        expect(c.fusions[0].iteration_count).toBe(3);
    });
    it('keep different groups apart when coalesced', async () => {
        const b = builder();
        const events = [b.alloc(1, 'X'), b.alloc(2, 'C'), b.alloc(3, 'Y'), b.alloc(4, 'D'), b.alloc(5, 'Z')];
        for (let i = 0; i < 2; i++)
            events.push({ type: 'region', label: 'It', region_depth: 1, time: 0 },
                { ...b.call('A', { ins: [1], outs: [2] }), region: 'It', line: 5 }, { ...b.call('B', { ins: [2], outs: [3] }), region: 'It', line: 6 },
                { ...b.call('P', { ins: [1], outs: [4] }), region: 'It', line: 8 }, { ...b.call('Q', { ins: [4], outs: [5] }), region: 'It', line: 9 },
                b.dealloc(2, 'C'), b.dealloc(4, 'D'));
        const c = await graph(events, { fusion_analysis: true, show_regions: true, expand_all_regions: true, collapse_iteration: true });
        expect(c.fusions).toHaveLength(2);
    });
});

describe('the DNN trace', () => {
    it('is drawn with fusion nodes that contain what the analysis says', async () => {
        const text = readFileSync(join(ROOT, 'examples/dnn_cifar10/trace_dnn_cifar10_opencl.json'), 'utf8');
        await call('load', { traceJson: text, traceFilename: 'trace.json' });
        const g = await call('graph', { fusion_analysis: true });
        const nodes = g.nodes.map(data);
        const fusions = nodes.filter(n => n.type === 'fusion');
        expect(fusions.length).toBeGreaterThan(50);
        for (const f of fusions) expect(nodes.filter(n => n.parent === f.id).length).toBeGreaterThanOrEqual(2);
    });
});
