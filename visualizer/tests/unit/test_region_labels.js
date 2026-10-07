// test_region_labels.js — the instances of a region are numbered, a region that has one instance is not.
import { describe, it, expect } from 'vitest';
import { builder, trace } from '../event_builders.js';
import { buildDisplayGraph } from '../../script/backend/pipeline.js';
import { RegionNode } from '../../script/backend/node.js';

const region = (label, t, depth = 1) => ({ type: 'region', label, time: t, region_depth: depth, file: 'x.cpp', line: 1 });

describe('labels of regions', () => {
    const b = builder();
    const inRegion = (name, t) => [{ ...region(name, t), region: undefined }, { ...b.call('f_' + t, { ins: [1], outs: [1] }), region: name }];
    const events = trace([b.alloc(1, 'X'), ...inRegion('Once', 0), ...inRegion('Iteration', 10), ...inRegion('Iteration', 20), ...inRegion('Iteration', 30)]);
    const labels = buildDisplayGraph(events, { show_regions: true }).graph.getAllNodes().filter(n => n instanceof RegionNode).map(n => n.label);

    it('has no number on a region that ran once', () => {
        expect(labels).toContain('Once');
    });

    it('numbers the instances of a region that ran more than once', () => {
        expect(labels.filter(l => l.startsWith('Iteration'))).toEqual(['Iteration₁', 'Iteration₂', 'Iteration₃']);
    });
});
