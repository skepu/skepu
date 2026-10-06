// graph_analysis.js — analyses that need the dependences between the events.
//
// Builds a DirectedGraph of the events and only reads it. (The graph's own findCriticalPath()
// counts hops; the path here is weighted by the time of the events.)
'use strict';

import { DirectedGraph, GraphSettings } from './graph.js';
import { duration } from './analysis.js';

/** A graph with what a dependence analysis needs: updates, transfers and scalars, no regions. */
export function buildAnalysisGraph(events) {
    const s = new GraphSettings();
    Object.assign(s, {
        updates: true, transfers: true, antideps: true, scalars: true,
        allocations: false, deallocations: false, regions: false, synthesizeRootRegion: false,
        mode: 'dependence-dag',
    });
    return DirectedGraph.fromEvents(events, s);
}

/**
 * The longest path in time through the dependences: the time the program could not go below
 * however many processors it had. `work_us` is the sum of the skeleton calls and
 * `parallelism` is work divided by the path. The path lists the nodes that take time.
 */
export function criticalPath(events) {
    const g = buildAnalysisGraph(events);
    const nodes = g.getAllNodes();
    const inDeg = new Map(nodes.map(n => [n, n.getIncomingEdges().length]));
    const dist = new Map(), prev = new Map();
    const queue = nodes.filter(n => inDeg.get(n) === 0);
    for (const n of queue) dist.set(n, n.duration ?? 0);

    for (let i = 0; i < queue.length; i++) {
        const n = queue[i];
        for (const edge of n.getOutgoingEdges()) {
            const c = edge.target;
            const through = dist.get(n) + (c.duration ?? 0);
            if (!dist.has(c) || through > dist.get(c)) { dist.set(c, through); prev.set(c, n); }
            inDeg.set(c, inDeg.get(c) - 1);
            if (inDeg.get(c) === 0) queue.push(c);
        }
    }

    let end = null;
    for (const n of nodes) if (dist.has(n) && (end === null || dist.get(n) > dist.get(end))) end = n;
    const chain = [];
    for (let n = end; n; n = prev.get(n)) chain.push(n);
    chain.reverse();

    const work = nodes.filter(n => n.type === 'skeleton_call').reduce((s, n) => s + (n.duration ?? 0), 0);
    const total = end ? dist.get(end) : 0;
    const timed = chain.filter(n => (n.duration ?? 0) > 0);
    return {
        total_us: total, hops: chain.length, work_us: work,
        parallelism: total ? +(work / total).toFixed(2) : null,
        path: timed.map(n => ({
            id: 'e' + n.traceIndex, type: n.type, label: n.label, pattern: n.pattern ?? undefined,
            duration_us: n.duration, share: total ? +(n.duration / total).toFixed(4) : 0,
            file: n.file || undefined, line: n.line ?? undefined,
        })),
        untimed_nodes_on_path: chain.length - timed.length,
        internal_transfers_in_graph: false,   // the graph leaves out the library's own transfers; see transfers()
    };
}
