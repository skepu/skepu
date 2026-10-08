// test_prng_allocation.js — the allocation of a random stream (an allocation that a call draws random numbers from) is marked, so that it is drawn
// in its own colour (graph_style.js), as the designer draws a random stream.
import { describe, it, expect } from 'vitest';
import { builder, trace } from '../event_builders.js';
import { buildDisplayGraph } from '../../script/backend/pipeline.js';

describe('the allocation of a random stream', () => {
    const b = builder();
    const events = trace([b.alloc(1, 'noise'), b.alloc(2, 'values'), b.alloc(3, 'other'), b.call('draw', { outs: [2], prng: 1 }), b.call('use', { ins: [2], outs: [3] })]);

    it('is marked in the graph, and the other allocations are not', () => {
        const { graph } = buildDisplayGraph(events, { container_allocations: true });
        const allocations = graph.getAllNodes().filter(n => n.type === 'allocation');
        expect(allocations.map(n => [n.label, n.toCytoscape().is_prng === true])).toEqual([['noise', true], ['values', false], ['other', false]]);
    });

    it('has the dependence to the call that draws from it', () => {
        const { graph } = buildDisplayGraph(events, { container_allocations: true });
        const edge = [...graph._dependenceEdges].find(e => e.toCytoscape().type === 'prng-dep');
        expect([edge.source.label, edge.target.label]).toEqual(['noise', 'draw']);
    });
});
