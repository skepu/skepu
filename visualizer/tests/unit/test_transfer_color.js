// test_transfer_color.js — transfers are coloured by the call measures, on one scale with the calls and the external blocks.
import { describe, it, expect } from 'vitest';
import { TraceDataProvider, buildMinmax } from '../../script/TraceDataProvider.js';

const node = (type, data) => ({ data: { type, ...data } });
const nodes = [
    node('skeleton_call', { duration: 100, elements: [10, 10], dag_depth: 0 }),
    node('transfer', { duration: 400, elements: [20, 20], dag_depth: 1 }),
    node('transfer', { duration: 10, elements: [10, 10], dag_depth: 2 }),
    node('allocation', { duration: 99999 }),
];

describe('the colour scale of the transfers', () => {
    it('has the durations and the elements of the transfers in the scale of the calls, and nothing of the other nodes', () => {
        const mm = buildMinmax(nodes, []);
        expect(mm['duration-total']).toEqual({ min: 10, max: 400 });
        expect(mm['elements']).toEqual({ min: 100, max: 400 });
        expect(mm['duration-per-element'].max).toBeCloseTo(1, 6);          // call: 100 over 100 elements; transfers: 1 and 0.1
        expect(mm['dag-depth']).toEqual({ min: 0, max: 2 });
    });

    it('puts a transfer on the scale by its duration: the longest at the end of the spectrum', () => {
        const mm = buildMinmax(nodes, []);
        const provider = new TraceDataProvider({ fetchGraph: () => Promise.reject(new Error('none')), settings: { 'node-color-call': 'duration-total' } });
        const ele = data => ({ data: key => data[key] });
        expect(provider.normFromEle('node-color-call', ele(nodes[1].data), mm)).toBe(1);
        expect(provider.normFromEle('node-color-call', ele(nodes[2].data), mm)).toBe(0);
        expect(provider.normFromEle('node-color-call', ele(nodes[0].data), mm)).toBeCloseTo(90 / 390, 6);
    });
});

describe('the data of a transfer for the timeline and the graph', () => {
    it('has what the colour measures take: the duration, the elements, the depth, the order and the fan of it', async () => {
        const { DirectedGraph, GraphSettings } = await import('../../script/backend/graph.js');
        const { TransferNode } = await import('../../script/backend/node.js');
        const settings = new GraphSettings();
        settings.synthesizeRootRegion = false;
        settings.transfers = true;
        const g = DirectedGraph.fromEvents([
            { type: 'allocation', label: 'A', object_id: 1, time: 1 },
            { type: 'skeleton_call', label: 'k', object_id: 100, pattern: 'Map', backend: 'OpenCL', devices: 1, elements: [4], elwise_inputs: [], proxy_inputs: [], scalar_inputs: [], outputs: [1], start: 2, end: 3 },
            { type: 'transfer', label: 'A', object_id: 1, direction: 'device-to-host', backend: 'OpenCL', elements: 4, start: 4, end: 9, time: 9 },
        ], settings);
        const transfer = g.getAllNodes().find(n => n instanceof TransferNode);
        for (const data of [transfer.toTimeline(), transfer.toCytoscape()]) {
            expect(data).toMatchObject({ duration: 5, durations: [5], duration_cv: 0, elements: [4] });
            expect(data.dag_depth).toBeGreaterThan(0);
            expect(data.fan_in).toBe(1);
        }
        expect(transfer.toTimeline()).toHaveProperty('total_order');
    });
});
