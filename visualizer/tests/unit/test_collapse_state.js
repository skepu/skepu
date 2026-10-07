// test_collapse_state.js — the collapse state of regions and fusion nodes, which the provider keeps and the
// Actions menu, the double click and the next fetch share (TraceDataProvider.js).
import { describe, it, expect } from 'vitest';
import { TraceDataProvider } from '../../script/TraceDataProvider.js';
import { drawnCollapsed } from '../../script/nesting.js';

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

describe('expanding one level', () => {
    // region_1 contains region_2 and region_3, which contains region_4; region_5 is on its own
    const parents = { region_1: null, region_2: 'region_1', region_3: 'region_1', region_4: 'region_3', region_5: null };
    const ids = Object.keys(parents);

    it('is the collapsed regions that are drawn: not those inside a collapsed region', () => {
        const all = { region_1: true, region_2: true, region_3: true, region_4: true, region_5: true };
        expect(drawnCollapsed(parents, all, ids)).toEqual(['region_1', 'region_5']);
    });

    it('goes one level further each time', () => {
        const state = { region_1: true, region_2: true, region_3: true, region_4: true, region_5: false };
        const once = () => drawnCollapsed(parents, state, ids).forEach(id => { state[id] = false; });
        once();
        expect(state).toEqual({ region_1: false, region_2: true, region_3: true, region_4: true, region_5: false });
        once();
        expect(state).toEqual({ region_1: false, region_2: false, region_3: false, region_4: true, region_5: false });
        once();
        expect(Object.values(state).every(c => c === false)).toBe(true);
    });

    it('does not open what is inside a collapsed fusion node', () => {
        const withFusion = { ...parents, fusion_e1: 'region_5', region_6: 'fusion_e1' };
        const state = { region_1: false, region_2: false, region_3: false, region_4: false, region_5: false, fusion_e1: true, region_6: true };
        expect(drawnCollapsed(withFusion, state, Object.keys(withFusion).filter(k => k.startsWith('region')))).toEqual([]);
    });
});

describe('the colours of backends and patterns', () => {
    const { provider } = make(() => ({}));

    it('are on the spectrum of the Colors menu, and change with it', () => {
        provider.colorSpectrum = 'default';
        const red = provider.backendColor('CPU'), pat = provider.patternColor('MapOverlap 2D');
        provider.colorSpectrum = 'grayscale';
        expect(provider.backendColor('CPU')).not.toBe(red);
        expect(provider.patternColor('MapOverlap 2D')).not.toBe(pat);
        expect(provider.backendColor('CPU')).toMatch(/^rgb\((\d+),\1,\1\)$/);     // gray
    });

    it('differ for each backend, are the same for the patterns of one skeleton, and are null for what is not known', () => {
        provider.colorSpectrum = 'viridis';
        const colors = ['CPU', 'OpenMP', 'OpenCL', 'CUDA', 'Hybrid'].map(b => provider.backendColor(b));
        expect(new Set(colors).size).toBe(5);
        expect(provider.patternColor('Reduce 1D-Full')).toBe(provider.patternColor('Reduce 2D'));
        expect(provider.backendColor('GPU')).toBeNull();
    });
});
