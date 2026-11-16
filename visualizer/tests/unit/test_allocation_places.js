// test_allocation_places.js — where a container was made: an allocation event that has a file (the program or a frontend such as SkePY named the place)
// has it in its info and a badge at its line, which is not on the way of the arrows between the calls. Without a file, the line of the event is a
// line of SkePU's own headers: no place.
import { describe, it, expect } from 'vitest';

import { DirectedGraph, GraphSettings } from '../../script/backend/graph.js';
import { AllocationNode } from '../../script/backend/node.js';
import { collectBadges } from '../../script/badges.js';

const events = [
    { type: 'allocation', label: 'x', object_id: 1, time: 1, file: '/p/run.py', line: 7 },
    { type: 'allocation', label: 'y', object_id: 2, time: 2, line: 228 },
    { type: 'skeleton_call', label: 'k', object_id: 100, pattern: 'Map', backend: 'CPU', elements: [4],
      elwise_inputs: [1], proxy_inputs: [], scalar_inputs: [], outputs: [2], start: 3, end: 4, file: '/p/mod.py', line: 9 },
];

const graph = () => DirectedGraph.fromEvents(events, Object.assign(new GraphSettings(), { allocations: true }));
const allocations = g => g.getAllNodes().filter(n => n instanceof AllocationNode);

describe('the allocation nodes', () => {
    it('have the place of the container in their info and badge when the event has a file', () => {
        const [x, y] = allocations(graph());
        expect(x.infoData()).toMatchObject({ Container: 'x', File: '/p/run.py', Line: 7 });
        expect(x.toSourceBadge()).toMatchObject({ type: 'allocation', file: '/p/run.py', line: 7 });
        expect(y.infoData().File).toBeUndefined();
        expect(y.toSourceBadge().file).toBeUndefined();
    });

    it('have a badge at their line that is not on the way of the arrows', () => {
        const badgeNodes = graph().toTimelineAndBadges([], true, [])[1];
        const badges = collectBadges(badgeNodes, []).badges;
        expect(badges['/p/run.py:7']).toMatchObject({ count: 1, noArrows: true, onlySnapshots: false });
        expect(badges['/p/mod.py:9']).toMatchObject({ noArrows: false });
        expect(Object.keys(badges).some(k => k.endsWith(':228'))).toBe(false);
    });
});
