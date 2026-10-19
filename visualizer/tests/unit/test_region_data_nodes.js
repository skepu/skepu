// test_region_data_nodes.js — the allocations and deallocations of containers in a region are nodes of the region (they were nodes of
// the root region, wherever the trace put them).
import { describe, it, expect } from 'vitest';

import { DirectedGraph, GraphSettings } from '../../script/backend/graph.js';
import { AllocationNode, DeallocationNode, RegionNode } from '../../script/backend/node.js';

const events = [
    { type: 'allocation', label: 'outside', object_id: 1, time: 1 },
    { type: 'region', label: 'Octave', region_depth: 1, time: 2 },
    { type: 'allocation', label: 'inside', object_id: 2, time: 3, region: 'Octave' },
    { type: 'skeleton_call', label: 'k', object_id: 100, pattern: 'Map', backend: 'CPU', elements: [4], region: 'Octave',
      elwise_inputs: [], proxy_inputs: [], scalar_inputs: [], outputs: [2], start: 4, end: 5 },
    { type: 'deallocation', label: 'inside', object_id: 2, time: 6, region: 'Octave' },
    { type: 'deallocation', label: 'outside', object_id: 1, time: 7 },
];

function graph(settings = {}) {
    const s = new GraphSettings();
    Object.assign(s, settings);
    return DirectedGraph.fromEvents(events, s);
}

const find = (g, cls, label) => g.getAllNodes().find(n => n instanceof cls && n.label === label);

describe('the allocations and deallocations in a region', () => {
    it('are nested in the region, as the call is', () => {
        const g = graph();
        expect(g.getAllNodes().some(n => n instanceof RegionNode && n.label === 'Octave')).toBe(true);
        // (the labels of the regions: a failed comparison of the nodes themselves would print the whole graph)
        expect(find(g, AllocationNode, 'inside').region?.label).toBe('Octave');
        expect(find(g, DeallocationNode, 'inside').region?.label).toBe('Octave');
        expect(g.getAllNodes().find(n => n.label === 'k').region?.label).toBe('Octave');
    });

    it('are not nested if the trace has them outside of the region', () => {
        const g = graph();
        expect(find(g, AllocationNode, 'outside').region?.label).not.toBe('Octave');
        expect(find(g, DeallocationNode, 'outside').region?.label).not.toBe('Octave');
    });
});
