// test_serial_fusion.js — Port of tests/unit/test_serial_fusion.py
import { describe, it, expect } from 'vitest';

import { CallNode, UpdateNode, FusedNode, ScalarNode } from '../../backend-js/node.js';
import { mapChainGraph, fanoutGraph, mapReduceGraph } from '../helpers.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function callNodes(graph) {
    return graph.getAllNodes().filter(n => n instanceof CallNode);
}

function callNodeIds(graph) {
    return new Set(callNodes(graph).map(n => n.id));
}

function byLabel(graph) {
    const m = {};
    for (const n of graph.getAllNodes()) m[n.label] = n;
    return m;
}

function fusionPairsByLabel(graph, fusions) {
    const idToLabel = {};
    for (const n of graph.getAllNodes()) idToLabel[n.id] = n.label;
    return fusions.map(([d, u]) => [idToLabel[d], idToLabel[u]]);
}

// ── Linear chain (map1 → map2 → map3) ────────────────────────────────────────

describe('TestMapChain', () => {

    it('returns two fusion pairs', () => {
        const g = mapChainGraph();
        const fusions = g.findSerialFusions();
        expect(fusions.length).toBe(2);
    });

    it('each pair is a two-element array', () => {
        const g = mapChainGraph();
        for (const pair of g.findSerialFusions()) {
            expect(Array.isArray(pair)).toBe(true);
            expect(pair.length).toBe(2);
        }
    });

    it('pair IDs are CallNode IDs', () => {
        const g = mapChainGraph();
        const ids = callNodeIds(g);
        for (const [d, u] of g.findSerialFusions()) {
            expect(ids.has(d)).toBe(true);
            expect(ids.has(u)).toBe(true);
        }
    });

    it('correct downstream/upstream labels', () => {
        const g = mapChainGraph();
        const labeled = fusionPairsByLabel(g, g.findSerialFusions());
        expect(labeled).toContainEqual(['map3', 'map2']);
        expect(labeled).toContainEqual(['map2', 'map1']);
    });

    it('no duplicate pairs', () => {
        const g = mapChainGraph();
        const pairs = g.findSerialFusions().map(p => p.join('|'));
        expect(new Set(pairs).size).toBe(pairs.length);
    });

    it('exactly one FusedNode created (chain extension)', () => {
        const g = mapChainGraph();
        g.findSerialFusions();
        const fused = g.getAllNodes().filter(n => n instanceof FusedNode);
        expect(fused.length).toBe(1);
    });

    it('FusedNode fusion_pattern is Map', () => {
        const g = mapChainGraph();
        g.findSerialFusions();
        const fused = g.getAllNodes().find(n => n instanceof FusedNode);
        expect(fused.fusionPattern).toBe('Map');
    });

    it('map2 and map3 are stamped with a FusedNode', () => {
        const g = mapChainGraph();
        g.findSerialFusions();
        const nodes = byLabel(g);
        expect(nodes['map2'].fused).not.toBeNull();
        expect(nodes['map3'].fused).not.toBeNull();
    });

    it('map2 and map3 share the same FusedNode', () => {
        const g = mapChainGraph();
        g.findSerialFusions();
        const nodes = byLabel(g);
        expect(nodes['map2'].fused).toBe(nodes['map3'].fused);
    });

    it('map1 (head of chain) is not stamped', () => {
        const g = mapChainGraph();
        g.findSerialFusions();
        expect(byLabel(g)['map1'].fused).toBeNull();
    });

    it('toCytoscape includes the FusedNode', () => {
        const g = mapChainGraph();
        g.findSerialFusions();
        const [cyNodes] = g.toCytoscape();
        const types = cyNodes.map(n => n.data.type);
        expect(types).toContain('fusion');
    });

    it('fused members carry parent pointing to the FusedNode', () => {
        const g = mapChainGraph();
        g.findSerialFusions();
        const [cyNodes] = g.toCytoscape();
        const fusedId = cyNodes.find(n => n.data.type === 'fusion').data.id;
        const members = cyNodes.filter(n => n.data.parent === fusedId);
        expect(members.length).toBeGreaterThanOrEqual(2);
    });

    it('Cytoscape FusedNode carries fusion_pattern', () => {
        const g = mapChainGraph();
        g.findSerialFusions();
        const [cyNodes] = g.toCytoscape();
        const fusedData = cyNodes.find(n => n.data.type === 'fusion').data;
        expect(fusedData.fusion_pattern).toBe('Map');
    });
});

// ── Fan-out ───────────────────────────────────────────────────────────────────

describe('TestFanOut', () => {

    it('no fusions returned', () => {
        const g = fanoutGraph();
        expect(g.findSerialFusions()).toEqual([]);
    });

    it('no FusedNodes created', () => {
        const g = fanoutGraph();
        g.findSerialFusions();
        expect(g.getAllNodes().filter(n => n instanceof FusedNode)).toEqual([]);
    });

    it('no CallNode is stamped', () => {
        const g = fanoutGraph();
        g.findSerialFusions();
        for (const node of callNodes(g)) {
            expect(node.fused).toBeNull();
        }
    });

    it('UpdateNode for shared container has two consumers', () => {
        const g = fanoutGraph();
        const updates = g.getAllNodes().filter(n => n instanceof UpdateNode);
        const shared = updates.find(n => n.getOutgoingEdges().length === 2);
        expect(shared).toBeDefined();
    });
});

// ── Map → Reduce ──────────────────────────────────────────────────────────────

describe('TestMapReduce', () => {

    it('one fusion detected', () => {
        const g = mapReduceGraph();
        expect(g.findSerialFusions().length).toBe(1);
    });

    it('pair is [reduce1, map1]', () => {
        const g = mapReduceGraph();
        const labeled = fusionPairsByLabel(g, g.findSerialFusions());
        expect(labeled).toContainEqual(['reduce1', 'map1']);
    });

    it('FusedNode fusion_pattern is Reduce', () => {
        const g = mapReduceGraph();
        g.findSerialFusions();
        const fused = g.getAllNodes().filter(n => n instanceof FusedNode);
        expect(fused.length).toBe(1);
        expect(fused[0].fusionPattern).toBe('Reduce');
    });

    it('reduce1 CallNode is stamped', () => {
        const g = mapReduceGraph();
        g.findSerialFusions();
        expect(byLabel(g)['reduce1'].fused).not.toBeNull();
    });

    it('leaf is a ScalarNode (reason second pass is needed)', () => {
        const g = mapReduceGraph();
        const leaves = g.getLeafNodes();
        expect(leaves.length).toBe(1);
        expect(leaves[0]).toBeInstanceOf(ScalarNode);
    });
});
