// test_construction.js — Port of tests/unit/test_construction.py
import { describe, it, expect, beforeEach } from 'vitest';

import { DirectedGraph, GraphSettings } from '../../backend-js/graph.js';
import {
    AllocationNode, CallNode, DeallocationNode, UpdateNode,
} from '../../backend-js/node.js';
import { ElwiseEdge } from '../../backend-js/edge.js';
import {
    loadFixture, graphFromFile, simpleGraph, collectExampleTraces,
} from '../helpers.js';

// ── simple fixture ────────────────────────────────────────────────────────────

describe('TestSimpleConstruction', () => {

    it('parses without error', () => {
        const g = simpleGraph();
        expect(g).not.toBeNull();
    });

    it('node count with default settings', () => {
        // simple.json: 2 allocations + 1 CallNode + 1 UpdateNode + 2 deallocations = 6
        const g = simpleGraph();
        expect(g.getAllNodes().length).toBe(6);
    });

    it('contains one CallNode with correct label and pattern', () => {
        const g = simpleGraph();
        const calls = g.getAllNodes().filter(n => n instanceof CallNode);
        expect(calls.length).toBe(1);
        expect(calls[0].label).toBe('map1');
        expect(calls[0].pattern).toBe('Map');
    });

    it('contains one UpdateNode', () => {
        const g = simpleGraph();
        const updates = g.getAllNodes().filter(n => n instanceof UpdateNode);
        expect(updates.length).toBe(1);
    });

    it('contains two AllocationNodes', () => {
        const g = simpleGraph();
        const allocs = g.getAllNodes().filter(n => n instanceof AllocationNode);
        expect(allocs.length).toBe(2);
    });

    it('contains two DeallocationNodes', () => {
        const g = simpleGraph();
        const deallocs = g.getAllNodes().filter(n => n instanceof DeallocationNode);
        expect(deallocs.length).toBe(2);
    });

    it('edges are wired bidirectionally', () => {
        const g = simpleGraph();
        for (const edge of g._edges) {
            expect(edge.source.getOutgoingEdges()).toContain(edge);
            expect(edge.target.getIncomingEdges()).toContain(edge);
        }
    });

    it('elwise edge from AllocationNode to CallNode', () => {
        const g = simpleGraph();
        const call = g.getAllNodes().find(n => n instanceof CallNode);
        const elwiseIn = call.getIncomingEdges().filter(e => e instanceof ElwiseEdge);
        expect(elwiseIn.length).toBe(1);
        expect(elwiseIn[0].source).toBeInstanceOf(AllocationNode);
    });

    it('CallNode output goes to UpdateNode', () => {
        const g = simpleGraph();
        const call   = g.getAllNodes().find(n => n instanceof CallNode);
        const update = g.getAllNodes().find(n => n instanceof UpdateNode);
        const targets = call.getOutgoingEdges().map(e => e.target);
        expect(targets).toContain(update);
    });

    it('total_orders are unique', () => {
        const g = simpleGraph();
        const orders = g.getAllNodes().map(n => n.totalOrder);
        expect(new Set(orders).size).toBe(orders.length);
    });

    it('getNodeById round-trip', () => {
        const g = simpleGraph();
        for (const node of g.getAllNodes()) {
            expect(g.getNodeById(node.id)).toBe(node);
        }
    });
});

// ── Settings flags ────────────────────────────────────────────────────────────

describe('TestSettingsFlags', () => {
    const simpleEvents = () => loadFixture('simple.json');

    it('updates=false removes UpdateNodes', () => {
        const s = new GraphSettings(); s.updates = false;
        const g = DirectedGraph.fromEvents(simpleEvents(), s);
        expect(g.getAllNodes().filter(n => n instanceof UpdateNode).length).toBe(0);
    });

    it('updates=false produces fewer nodes', () => {
        const s = new GraphSettings(); s.updates = false;
        const g = DirectedGraph.fromEvents(simpleEvents(), s);
        expect(g.getAllNodes().length).toBe(5);
    });

    it('allocations=false removes AllocationNodes', () => {
        const s = new GraphSettings(); s.allocations = false;
        const g = DirectedGraph.fromEvents(simpleEvents(), s);
        expect(g.getAllNodes().filter(n => n instanceof AllocationNode).length).toBe(0);
    });

    it('deallocations=false removes DeallocationNodes', () => {
        const s = new GraphSettings(); s.deallocations = false;
        const g = DirectedGraph.fromEvents(simpleEvents(), s);
        expect(g.getAllNodes().filter(n => n instanceof DeallocationNode).length).toBe(0);
    });

    it('edges still bidirectional with updates=false', () => {
        const s = new GraphSettings(); s.updates = false;
        const g = DirectedGraph.fromEvents(simpleEvents(), s);
        for (const edge of g._edges) {
            expect(edge.source.getOutgoingEdges()).toContain(edge);
            expect(edge.target.getIncomingEdges()).toContain(edge);
        }
    });
});

// ── Analysis pipeline ─────────────────────────────────────────────────────────

describe('TestAnalysisPipeline', () => {

    it('computeDepths sets non-negative integer depth on all nodes', () => {
        const g = simpleGraph();
        g.computeDepths();
        for (const node of g.getAllNodes()) {
            expect(typeof node.depth).toBe('number');
            expect(node.depth).toBeGreaterThanOrEqual(0);
        }
    });

    it('findCriticalPath marks at least one node', () => {
        const g = simpleGraph();
        g.computeDepths();
        g.findCriticalPath();
        const critical = g.getAllNodes().filter(n => n.isCriticalPath);
        expect(critical.length).toBeGreaterThanOrEqual(1);
    });

    it('findCriticalPath marks at least one edge', () => {
        const g = simpleGraph();
        g.computeDepths();
        g.findCriticalPath();
        const critical = g._edges.filter(e => e.isCriticalPath);
        expect(critical.length).toBeGreaterThanOrEqual(1);
    });

    it('toCytoscape returns nodes and edges', () => {
        const g = simpleGraph();
        const [nodes, edges] = g.toCytoscape();
        expect(nodes.length).toBeGreaterThan(0);
        expect(edges.length).toBeGreaterThan(0);
    });

    it('toCytoscape node schema has required keys', () => {
        const g = simpleGraph();
        const [nodes] = g.toCytoscape();
        const required = new Set(['id', 'label', 'type', 'total_order']);
        for (const entry of nodes) {
            for (const key of required) {
                expect(entry.data).toHaveProperty(key);
            }
        }
    });

    it('toCytoscape edge schema has required keys', () => {
        const g = simpleGraph();
        const [, edges] = g.toCytoscape();
        const required = new Set(['id', 'source', 'target', 'type']);
        for (const entry of edges) {
            for (const key of required) {
                expect(entry.data).toHaveProperty(key);
            }
        }
    });
});

// ── Example traces ────────────────────────────────────────────────────────────

const examplePaths = collectExampleTraces();

describe('TestExampleTraces', () => {

    it.each(examplePaths)('parses %s without error', (path) => {
        const g = graphFromFile(path);
        expect(g).not.toBeNull();
    });

    it.each(examplePaths)('%s has nodes', (path) => {
        const g = graphFromFile(path);
        expect(g.getAllNodes().length).toBeGreaterThan(0);
    });

    it.each(examplePaths)('%s edges wired bidirectionally', (path) => {
        const g = graphFromFile(path);
        for (const edge of g._edges) {
            expect(edge.source.getOutgoingEdges()).toContain(edge);
            expect(edge.target.getIncomingEdges()).toContain(edge);
        }
    });

    it.each(examplePaths)('%s full pipeline (depth + critpath + toCytoscape)', (path) => {
        const g = graphFromFile(path);
        g.computeDepths();
        g.findCriticalPath();
        const [nodes] = g.toCytoscape();
        expect(nodes.length).toBeGreaterThan(0);
    });
});
