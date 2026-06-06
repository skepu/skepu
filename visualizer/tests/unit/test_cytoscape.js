// test_cytoscape.js — Port of tests/unit/test_cytoscape.py
import { describe, it, expect } from 'vitest';

import {
    AllocationNode, CallNode, DeallocationNode, ExternalNode,
    FusedNode, RegionNode, ScalarNode, TransferNode, UpdateNode,
} from '../../backend-js/node.js';
import {
    AntiDepEdge, ElwiseEdge, ProxyEdge, PrngEdge, ScalarEdge, VirtualAliasEdge,
} from '../../backend-js/edge.js';
import {
    simpleGraph, externalGraph, transferGraph, regionGraph,
    antidepGraph, prngGraph, virtualAliasGraph, fusedChainGraph,
    mapReduceGraph,
} from '../helpers.js';

// ── Expected key sets ─────────────────────────────────────────────────────────

const BASE_NODE_KEYS = new Set([
    'id', 'label', 'total_order', 'type',
    'is_critical_path', 'dag_depth', 'iteration_count',
    'trace_index', 'nesting_level', 'fan_in', 'fan_out',
]);

const COMPUTATION_EXTRA_KEYS = new Set([
    'start', 'end', 'duration', 'durations', 'intervals',
    'file', 'line', 'backend', 'duration_cv',
]);

const BASE_EDGE_KEYS = new Set([
    'id', 'label', 'source', 'target',
    'is_critical_path', 'iteration_count', 'order_span', 'cross_backend',
]);

// ── Helpers ───────────────────────────────────────────────────────────────────

function first(graph, Type) {
    return graph.getAllNodes().find(n => n instanceof Type);
}

function firstEdge(graph, Type) {
    return graph._edges.find(e => e instanceof Type);
}

function assertKeys(data, required) {
    for (const key of required) {
        expect(data, `missing key '${key}'`).toHaveProperty(key);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Node types
// ══════════════════════════════════════════════════════════════════════════════

describe('TestAllocationNodeSchema', () => {
    it('has base node keys', () => {
        assertKeys(first(simpleGraph(), AllocationNode).toCytoscape(), BASE_NODE_KEYS);
    });
    it('type discriminator is allocation', () => {
        expect(first(simpleGraph(), AllocationNode).toCytoscape().type).toBe('allocation');
    });
    it('has timestamp key', () => {
        expect(first(simpleGraph(), AllocationNode).toCytoscape()).toHaveProperty('timestamp');
    });
    it('infoData has base keys', () => {
        assertKeys(first(simpleGraph(), AllocationNode).infoData(),
                   new Set(['internal', 'type', 'trace_index']));
    });
});

describe('TestDeallocationNodeSchema', () => {
    it('has base node keys', () => {
        assertKeys(first(simpleGraph(), DeallocationNode).toCytoscape(), BASE_NODE_KEYS);
    });
    it('type discriminator is deallocation', () => {
        expect(first(simpleGraph(), DeallocationNode).toCytoscape().type).toBe('deallocation');
    });
    it('has timestamp key', () => {
        expect(first(simpleGraph(), DeallocationNode).toCytoscape()).toHaveProperty('timestamp');
    });
    it('infoData has base keys', () => {
        assertKeys(first(simpleGraph(), DeallocationNode).infoData(),
                   new Set(['internal', 'type', 'trace_index']));
    });
});

describe('TestCallNodeSchema', () => {
    it('has base node keys', () => {
        assertKeys(first(simpleGraph(), CallNode).toCytoscape(), BASE_NODE_KEYS);
    });
    it('has computation extra keys', () => {
        assertKeys(first(simpleGraph(), CallNode).toCytoscape(), COMPUTATION_EXTRA_KEYS);
    });
    it('type discriminator is skeleton_call', () => {
        expect(first(simpleGraph(), CallNode).toCytoscape().type).toBe('skeleton_call');
    });
    it('has pattern key (non-empty string)', () => {
        const d = first(simpleGraph(), CallNode).toCytoscape();
        expect(d).toHaveProperty('pattern');
        expect(typeof d.pattern).toBe('string');
        expect(d.pattern.length).toBeGreaterThan(0);
    });
    it('has elements key (array)', () => {
        const d = first(simpleGraph(), CallNode).toCytoscape();
        expect(d).toHaveProperty('elements');
        expect(Array.isArray(d.elements)).toBe(true);
    });
    it('duration_cv is a number', () => {
        expect(typeof first(simpleGraph(), CallNode).toCytoscape().duration_cv).toBe('number');
    });
    it('infoData has required keys', () => {
        assertKeys(first(simpleGraph(), CallNode).infoData(), new Set([
            'internal', 'type', 'trace_index',
            'Duration', 'File', 'Line', 'Backend', 'durations',
            'Pattern', 'Elements',
        ]));
    });
});

describe('TestExternalNodeSchema', () => {
    it('has base node keys', () => {
        assertKeys(first(externalGraph(), ExternalNode).toCytoscape(), BASE_NODE_KEYS);
    });
    it('has computation extra keys', () => {
        assertKeys(first(externalGraph(), ExternalNode).toCytoscape(), COMPUTATION_EXTRA_KEYS);
    });
    it('type discriminator is external', () => {
        expect(first(externalGraph(), ExternalNode).toCytoscape().type).toBe('external');
    });
    it('has no pattern or elements key', () => {
        const d = first(externalGraph(), ExternalNode).toCytoscape();
        expect(d).not.toHaveProperty('pattern');
        expect(d).not.toHaveProperty('elements');
    });
    it('infoData has required keys', () => {
        assertKeys(first(externalGraph(), ExternalNode).infoData(), new Set([
            'internal', 'type', 'trace_index',
            'Duration', 'File', 'Line', 'Backend', 'durations',
        ]));
    });
});

describe('TestUpdateNodeSchema', () => {
    it('has base node keys', () => {
        assertKeys(first(simpleGraph(), UpdateNode).toCytoscape(), BASE_NODE_KEYS);
    });
    it('type discriminator is container_update', () => {
        expect(first(simpleGraph(), UpdateNode).toCytoscape().type).toBe('container_update');
    });
    it('has version key (integer)', () => {
        const d = first(simpleGraph(), UpdateNode).toCytoscape();
        expect(d).toHaveProperty('version');
        expect(Number.isInteger(d.version)).toBe(true);
    });
    it('infoData has required keys', () => {
        assertKeys(first(simpleGraph(), UpdateNode).infoData(),
                   new Set(['internal', 'type', 'trace_index', 'Version']));
    });
    it('infoData.internal has is_live', () => {
        expect(first(simpleGraph(), UpdateNode).infoData().internal).toHaveProperty('is_live');
    });
});

describe('TestScalarNodeSchema', () => {
    it('has base node keys', () => {
        assertKeys(first(mapReduceGraph(), ScalarNode).toCytoscape(), BASE_NODE_KEYS);
    });
    it('type discriminator is scalar', () => {
        expect(first(mapReduceGraph(), ScalarNode).toCytoscape().type).toBe('scalar');
    });
    it('has no computation keys', () => {
        const d = first(mapReduceGraph(), ScalarNode).toCytoscape();
        for (const key of ['start', 'end', 'duration', 'pattern', 'elements']) {
            expect(d).not.toHaveProperty(key);
        }
    });
    it('infoData has base keys', () => {
        assertKeys(first(mapReduceGraph(), ScalarNode).infoData(),
                   new Set(['internal', 'type', 'trace_index']));
    });
});

describe('TestTransferNodeSchema', () => {
    it('has base node keys', () => {
        assertKeys(first(transferGraph(), TransferNode).toCytoscape(), BASE_NODE_KEYS);
    });
    it('type discriminator is transfer', () => {
        expect(first(transferGraph(), TransferNode).toCytoscape().type).toBe('transfer');
    });
    it('has transfer-specific keys', () => {
        assertKeys(first(transferGraph(), TransferNode).toCytoscape(),
                   new Set(['direction', 'timestamp', 'start', 'end', 'duration', 'intervals']));
    });
    it('direction is a known value', () => {
        const dir = first(transferGraph(), TransferNode).toCytoscape().direction;
        expect(['device-to-host', 'host-to-device']).toContain(dir);
    });
    it('infoData has required keys', () => {
        assertKeys(first(transferGraph(), TransferNode).infoData(),
                   new Set(['internal', 'type', 'trace_index',
                            'Direction', 'Duration', 'Backend']));
    });
});

describe('TestRegionNodeSchema', () => {
    function regionNode() {
        const g = regionGraph();
        const n = first(g, RegionNode);
        n.computeExtent();
        return n;
    }

    it('has base node keys', () => {
        assertKeys(regionNode().toCytoscape(), BASE_NODE_KEYS);
    });
    it('type discriminator is region', () => {
        expect(regionNode().toCytoscape().type).toBe('region');
    });
    it('has region-specific keys', () => {
        assertKeys(regionNode().toCytoscape(), new Set([
            'region_depth', 'region_start', 'region_end',
            'region_intervals', 'persistent_region_id', 'file', 'line',
        ]));
    });
    it('persistent_region_id starts with "region_"', () => {
        const pid = regionNode().toCytoscape().persistent_region_id;
        expect(typeof pid).toBe('string');
        expect(pid.startsWith('region_')).toBe(true);
    });
    it('infoData has required keys', () => {
        assertKeys(regionNode().infoData(),
                   new Set(['internal', 'type', 'trace_index', 'File', 'Line']));
    });
});

describe('TestFusedNodeSchema', () => {
    it('has base node keys', () => {
        assertKeys(first(fusedChainGraph(), FusedNode).toCytoscape(), BASE_NODE_KEYS);
    });
    it('type discriminator is fusion', () => {
        expect(first(fusedChainGraph(), FusedNode).toCytoscape().type).toBe('fusion');
    });
    it('has fusion_pattern key (non-empty string)', () => {
        const d = first(fusedChainGraph(), FusedNode).toCytoscape();
        expect(d).toHaveProperty('fusion_pattern');
        expect(typeof d.fusion_pattern).toBe('string');
        expect(d.fusion_pattern.length).toBeGreaterThan(0);
    });
    it('fusion_pattern is a known value', () => {
        const p = first(fusedChainGraph(), FusedNode).toCytoscape().fusion_pattern;
        expect(['Map', 'Reduce', 'MapReduce', 'MapOverlap']).toContain(p);
    });
    it('infoData has required keys', () => {
        assertKeys(first(fusedChainGraph(), FusedNode).infoData(),
                   new Set(['internal', 'type', 'trace_index', 'Fusion pattern']));
    });
});

// ══════════════════════════════════════════════════════════════════════════════
// Edge types
// ══════════════════════════════════════════════════════════════════════════════

describe('TestElwiseEdgeSchema', () => {
    it('has base edge keys', () => {
        assertKeys(firstEdge(simpleGraph(), ElwiseEdge).toCytoscape(), BASE_EDGE_KEYS);
    });
    it('type=forward-dep, access_mode=elwise', () => {
        const d = firstEdge(simpleGraph(), ElwiseEdge).toCytoscape();
        expect(d.type).toBe('forward-dep');
        expect(d.access_mode).toBe('elwise');
    });
    it('source and target are non-empty strings', () => {
        const d = firstEdge(simpleGraph(), ElwiseEdge).toCytoscape();
        expect(typeof d.source).toBe('string');
        expect(d.source.length).toBeGreaterThan(0);
        expect(typeof d.target).toBe('string');
        expect(d.target.length).toBeGreaterThan(0);
    });
});

describe('TestProxyEdgeSchema', () => {
    it('has base edge keys', () => {
        assertKeys(firstEdge(externalGraph(), ProxyEdge).toCytoscape(), BASE_EDGE_KEYS);
    });
    it('type=forward-dep, access_mode=proxy', () => {
        const d = firstEdge(externalGraph(), ProxyEdge).toCytoscape();
        expect(d.type).toBe('forward-dep');
        expect(d.access_mode).toBe('proxy');
    });
});

describe('TestScalarEdgeSchema', () => {
    it('has base edge keys', () => {
        assertKeys(firstEdge(mapReduceGraph(), ScalarEdge).toCytoscape(), BASE_EDGE_KEYS);
    });
    it('type=forward-dep, access_mode=scalar', () => {
        const d = firstEdge(mapReduceGraph(), ScalarEdge).toCytoscape();
        expect(d.type).toBe('forward-dep');
        expect(d.access_mode).toBe('scalar');
    });
    it('cross_backend is always false', () => {
        expect(firstEdge(mapReduceGraph(), ScalarEdge).toCytoscape().cross_backend).toBe(false);
    });
});

describe('TestAntiDepEdgeSchema', () => {
    it('has base edge keys', () => {
        assertKeys(firstEdge(antidepGraph(), AntiDepEdge).toCytoscape(), BASE_EDGE_KEYS);
    });
    it('type=anti-dep, access_mode=proxy', () => {
        const d = firstEdge(antidepGraph(), AntiDepEdge).toCytoscape();
        expect(d.type).toBe('anti-dep');
        expect(d.access_mode).toBe('proxy');
    });
});

describe('TestPrngEdgeSchema', () => {
    it('has base edge keys', () => {
        assertKeys(firstEdge(prngGraph(), PrngEdge).toCytoscape(), BASE_EDGE_KEYS);
    });
    it('type=prng-dep, access_mode=proxy', () => {
        const d = firstEdge(prngGraph(), PrngEdge).toCytoscape();
        expect(d.type).toBe('prng-dep');
        expect(d.access_mode).toBe('proxy');
    });
});

describe('TestVirtualAliasEdgeSchema', () => {
    function aliasEdge() {
        const g = virtualAliasGraph();
        return g._decorative_edges.find(e => e instanceof VirtualAliasEdge);
    }

    it('has required keys', () => {
        assertKeys(aliasEdge().toCytoscape(), new Set([
            'id', 'label', 'source', 'target', 'type',
            'is_critical_path', 'iteration_count', 'virtual_label',
        ]));
    });
    it('type is virtual-alias', () => {
        expect(aliasEdge().toCytoscape().type).toBe('virtual-alias');
    });
    it('virtual_label matches label', () => {
        const d = aliasEdge().toCytoscape();
        expect(d.virtual_label).toBe(d.label);
    });
    it('has no access_mode', () => {
        expect(aliasEdge().toCytoscape()).not.toHaveProperty('access_mode');
    });
});

// ══════════════════════════════════════════════════════════════════════════════
// Cytoscape envelope
// ══════════════════════════════════════════════════════════════════════════════

describe('TestCytoscapeEnvelope', () => {

    it('nodes wrapped in {data: {...}}', () => {
        const [nodes] = simpleGraph().toCytoscape();
        for (const entry of nodes) {
            expect(Object.keys(entry)).toEqual(['data']);
            expect(entry.data).toHaveProperty('id');
        }
    });

    it('edges wrapped in {data: {...}}', () => {
        const [, edges] = simpleGraph().toCytoscape();
        for (const entry of edges) {
            expect(Object.keys(entry)).toEqual(['data']);
            expect(entry.data).toHaveProperty('id');
        }
    });

    it('all_region_pids is an array with at least one entry', () => {
        const [, , pids] = regionGraph().toCytoscape();
        expect(Array.isArray(pids)).toBe(true);
        expect(pids.length).toBeGreaterThanOrEqual(1);
    });
});
