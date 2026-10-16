// test_trace_facts.js — what the hover of an instance says about its calls in the trace (src/traceFacts.ts), also against the graph of a shipped trace.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { callsOfInstance, summarizeCalls, describeCalls, formatMicroseconds, callsMarkdown } from '../../src/traceFacts.ts';
import { buildDisplayGraph } from '../../script/backend/pipeline.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

describe('a time', () => {
    it('is in the unit that reads best', () => {
        expect(formatMicroseconds(850)).toBe('850 µs');
        expect(formatMicroseconds(1234)).toBe('1.23 ms');
        expect(formatMicroseconds(25000)).toBe('25 ms');
        expect(formatMicroseconds(3400000)).toBe('3.4 s');
    });
});

describe('the calls of an instance', () => {
    const nodes = [
        { id: 'a', type: 'skeleton_call', label: 'dotp', line: 40, duration: 100, backend: 'OpenMP', elements: 2048 },
        { id: 'b', type: 'skeleton_call', label: 'dotp', line: 41, duration: 300, backend: 'OpenMP' },
        { id: 'c', type: 'skeleton_call', label: 'other', line: 50, duration: 5 },
        { id: 'd', type: 'allocation', label: 'dotp' },
    ];

    it('are the call nodes with the name of the instance', () => {
        expect(callsOfInstance(nodes, 'dotp').map(n => n.id)).toEqual(['a', 'b']);
        expect(callsOfInstance(nodes, 'nothing')).toEqual([]);
    });

    it('are counted, timed and described', () => {
        const facts = summarizeCalls(callsOfInstance(nodes, 'dotp'));
        expect(facts).toEqual({ calls: 2, total: 400, mean: 200, max: 300, backends: ['OpenMP'], elements: '2048' });
        expect(describeCalls(facts)).toBe('2 calls · total 400 µs · mean 200 µs · longest 300 µs · OpenMP · 2048 elements');
    });

    it('of a node of coalesced iterations are the iterations', () => {
        const facts = summarizeCalls([{ id: 'x', type: 'skeleton_call', label: 'f', durations: [10, 20, 30], duration: 60, iteration_count: 3 }]);
        expect(facts.calls).toBe(3);
        expect(facts.total).toBe(60);
        expect(describeCalls({ ...facts, backends: [] })).toBe('3 calls · total 60 µs · mean 20 µs · longest 30 µs');
    });

    it('have nothing to say without times', () => {
        expect(summarizeCalls([{ id: 'x', type: 'skeleton_call', label: 'f' }])).toBeNull();
        expect(summarizeCalls([])).toBeNull();
    });

    it('have a link that shows them in the viewer, with the nodes as the argument', () => {
        const md = callsMarkdown('dotp', summarizeCalls(callsOfInstance(nodes, 'dotp')), { fileName: '/a.cpp', line: 39, nodeIDs: ['a', 'b'], sessionId: 2 });
        expect(md).toContain('**dotp**: 2 calls');
        expect(md).toContain('[Show in SkeVU](command:skevu.highlightNode?' + encodeURIComponent(JSON.stringify(['/a.cpp', 39, ['a', 'b'], 2])) + ')');
    });
});

describe('the graph of a shipped trace', () => {
    it('has call nodes that the hover finds by the name of the instance, with times', () => {
        const trace = JSON.parse(readFileSync(join(ROOT, 'examples/horner/trace_horner_openmp.json'), 'utf8'));
        const { graph } = buildDisplayGraph(trace.data ?? trace, {});
        const nodes = graph.getAllNodes().map(n => n.toCytoscape());
        const names = [...new Set(nodes.filter(n => n.type === 'skeleton_call').map(n => n.label))];
        expect(names.length).toBeGreaterThan(0);
        const facts = summarizeCalls(callsOfInstance(nodes, names[0]));
        expect(facts?.calls).toBeGreaterThan(0);
        expect(facts?.total).toBeGreaterThan(0);
    });
});
