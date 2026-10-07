// test_graph_copy.js — a copy of a container (copy constructor, assignment) is host code that reads one container and writes another.
import { describe, it, expect } from 'vitest';
import { builder, trace } from '../event_builders.js';
import { buildDisplayGraph } from '../../script/backend/pipeline.js';
import { ExternalNode } from '../../script/backend/node.js';

describe('a copy of a container', () => {
    const b = builder();
    const events = trace([b.alloc(1, 'X'), b.alloc(2, 'Y'), b.call('f', { ins: [1], outs: [1] }), b.copy(1, 2), b.call('g', { ins: [2], outs: [2] })]);

    it('is a node between the call that wrote the container and the call that reads the copy', () => {
        const { graph } = buildDisplayGraph(events, { container_allocations: false });
        const copy = graph.getAllNodes().find(n => n instanceof ExternalNode);
        expect(copy.label).toBe('copy of X');
        const edges = [...graph._dependenceEdges].map(e => [e.source.label, e.target.label]);
        // (through the versions of the containers): f writes X, the copy reads X and writes Y, g reads Y
        expect(edges).toContainEqual(['X', 'copy of X']);
        expect(edges).toContainEqual(['copy of X', 'Y']);
        expect(edges).toContainEqual(['Y', 'g']);
    });
});
