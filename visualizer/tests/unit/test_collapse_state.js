// test_collapse_state.js — the collapse state of regions and fusion nodes, which the provider keeps and the
// Actions menu, the double click and the next fetch share (TraceDataProvider.js).
import { describe, it, expect } from 'vitest';
import { TraceDataProvider } from '../../script/TraceDataProvider.js';

// A provider whose backend answers with the given region and fusion ids, and records the requests.
function make(response) {
    const requests = [];
    const provider = new TraceDataProvider({
        fetchGraph: async params => { requests.push(params); return { nodes: [], edges: [], timeline_nodes: [], ...response() }; },
        settings: { 'graph-mode': 'dependence-dag', show_regions: true },
    });
    return { provider, requests };
}

describe('the collapse state', () => {
    it('starts with the regions of a large trace collapsed and the fusion nodes expanded', async () => {
        const { provider } = make(() => ({ all_region_pids: ['region_1', 'region_2'], all_fusion_pids: ['fusion_e5'], auto_expand_regions: false }));
        await provider.fetch();
        expect(provider.collapsedNodes).toEqual({ region_1: true, region_2: true, fusion_e5: false });
    });

    it('starts with the regions of a small trace expanded', async () => {
        const { provider } = make(() => ({ all_region_pids: ['region_1'], all_fusion_pids: [], auto_expand_regions: true }));
        await provider.fetch();
        expect(provider.collapsedNodes).toEqual({ region_1: false });
    });

    it('asks for the expanded regions and the collapsed fusion nodes', async () => {
        const { provider, requests } = make(() => ({ all_region_pids: ['region_1', 'region_2'], all_fusion_pids: ['fusion_e5', 'fusion_e9'], auto_expand_regions: false }));
        await provider.fetch();
        provider.collapsedNodes.region_2 = false;      // what the menu or a double click does
        provider.collapsedNodes.fusion_e9 = true;
        await provider.fetch();
        expect(requests[1].expanded_regions).toBe('region_2');
        expect(requests[1].collapsed_fusions).toBe('fusion_e9');
        expect(requests[0].collapsed_fusions).toBe('');
    });

    it('keeps what the user set when the next response has the same ids', async () => {
        const { provider } = make(() => ({ all_region_pids: ['region_1'], all_fusion_pids: ['fusion_e5'], auto_expand_regions: false }));
        await provider.fetch();
        provider.collapsedNodes.region_1 = false; provider.collapsedNodes.fusion_e5 = true;
        await provider.fetch();
        expect(provider.collapsedNodes).toEqual({ region_1: false, fusion_e5: true });
    });

    it('is forgotten when a new trace is loaded, in the same object that the other modules hold', async () => {
        const { provider } = make(() => ({ all_region_pids: ['region_1', 'region_2'], all_fusion_pids: ['fusion_e5'], auto_expand_regions: false }));
        await provider.fetch();
        const shared = provider.collapsedNodes;
        shared.region_1 = false; shared.fusion_e5 = true;
        provider.resetFirstLoad();
        expect(provider.collapsedNodes).toBe(shared);
        expect(shared).toEqual({});
        await provider.fetch();
        expect(shared).toEqual({ region_1: true, region_2: true, fusion_e5: false });   // not what the previous trace had
    });
});
