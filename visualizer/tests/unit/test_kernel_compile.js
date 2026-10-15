// test_kernel_compile.js — the builds of OpenCL kernels (kernel_compile events): a node for each, a setup edge from it to the first call of
// the skeleton instance, nothing unless the setting is on, and a row of their own in the timeline.
import { describe, it, expect } from 'vitest';
import { builder, trace } from '../event_builders.js';
import { DirectedGraph } from '../../script/backend/graph.js';
import { CompileNode } from '../../script/backend/node.js';
import { SetupEdge } from '../../script/backend/edge.js';
import { testSettings, allEdges } from '../helpers.js';
import { layoutRows } from '../../script/timeline_layout.js';

function graphOf(events, compiles = true) {
    const settings = testSettings();
    settings.compiles = compiles;
    return DirectedGraph.fromEvents(trace(events), settings);
}
const nodesOf = g => g.getAllNodes().filter(n => n instanceof CompileNode);

describe('kernel builds in the graph', () => {
    it('makes a node of each build, and a setup edge from it to the first call of its instance', () => {
        const b = builder();
        const g = graphOf([
            b.alloc(1, 'a'),
            b.compile(7, 'scale'), b.compile(7, 'scale', { device: 'GPU 1' }),
            b.call('scale', { outs: [1], backend: 'OpenCL', instance: 7 }),
            b.call('scale', { outs: [1], backend: 'OpenCL', instance: 7 }),
        ]);
        const builds = nodesOf(g);
        expect(builds).toHaveLength(2);
        const setups = allEdges(g).filter(e => e instanceof SetupEdge);
        expect(setups).toHaveLength(2);
        for (const e of setups) { expect(builds).toContain(e.source); expect(e.target.type).toBe('skeleton_call'); }
        expect(setups[0].toCytoscape()).toMatchObject({ type: 'setup-dep', access_mode: 'setup', cross_backend: false });
    });

    it('has the label of the instance, and which kernel of it', () => {
        const b = builder();
        const g = graphOf([b.compile(null, 'matrix transpose'), b.compile(3, 'smooth', { qualifier: 'rows' })]);
        expect(nodesOf(g).map(n => n.label)).toEqual(['build: matrix transpose', 'build: smooth rows']);
    });

    it('has no edge for a kernel of the library, or for an instance that is not called', () => {
        const b = builder();
        const g = graphOf([b.compile(null, 'matrix transpose'), b.compile(9, 'never called')]);
        expect(nodesOf(g)).toHaveLength(2);
        expect(allEdges(g).filter(e => e instanceof SetupEdge)).toHaveLength(0);
    });

    it('does not link a build to a call of another instance', () => {
        const b = builder();
        const events = [b.compile(7, 'scale'), b.call('other', { backend: 'OpenCL', instance: 8 })];
        expect(allEdges(graphOf(events)).filter(e => e instanceof SetupEdge)).toHaveLength(0);
    });

    it('has the data of the build for the views', () => {
        const b = builder();
        const g = graphOf([b.compile(7, 'scale', { ok: false })]);
        const [n] = nodesOf(g);
        expect(n.toCytoscape()).toMatchObject({ type: 'compile', ok: false, device: 'GPU 0', duration: 5 });
        expect(n.toTimeline()).toMatchObject({ type: 'compile', start: 0, end: 5, ok: false });
        expect(n.infoData()['Build failed']).toBe('syntax error');
        expect(n.toSourceBadge()).toBeNull();
    });

    it('has nothing unless the setting is on', () => {
        const b = builder();
        const g = graphOf([b.compile(7, 'scale'), b.call('scale', { backend: 'OpenCL', instance: 7 })], false);
        expect(nodesOf(g)).toHaveLength(0);
        expect(allEdges(g).filter(e => e instanceof SetupEdge)).toHaveLength(0);
    });
});

describe('kernel builds in the timeline', () => {
    const heights = { regionH: 0, rowH: 28, headerH: 18, transferH: 0, commH: 0, eventH: 0, axisH: 36 };
    it('have a row between the transfers and the communication', () => {
        const L = layoutRows({ ...heights, ranks: 0, backends: ['OpenCL'], transferH: 28, compileH: 28, commH: 28, eventH: 28 });
        expect(L.compileY).toBe(L.transferY + 28);
        expect(L.commY).toBe(L.compileY + 28);
        expect(L.axisY).toBe(L.commY + 56);
    });
    it('have no row without builds', () => {
        const L = layoutRows({ ...heights, ranks: 0, backends: ['OpenCL'], transferH: 28, commH: 28 });
        expect(L.commY).toBe(L.transferY + 28);
    });
});
