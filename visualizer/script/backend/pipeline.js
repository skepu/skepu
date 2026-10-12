// pipeline.js — from the events of a trace to the analysed graph that the views show.
//
// What the worker does for a 'graph' request before it serializes the result, without any
// worker, I/O or state of its own: the same steps can be run in Node or in tests.
'use strict';

import { DirectedGraph, GraphSettings } from './graph.js';
import { RegionNode, FusedNode } from './node.js';
import { analyzeFusion, explainBlocked } from './fusion.js';
import { preprocessFilePaths } from './trace.js';

/**
 * The settings of a graph from the options of the views (the names are those of the query
 * string of the original Flask server):
 *   mode, container_allocations, container_deallocations, container_transfers, show_scalars,
 *   anti_deps, data_as_edges, show_regions
 */
export function graphSettings({
    mode                    = 'dependence-dag',
    container_allocations   = false,
    container_deallocations = false,
    container_transfers     = false,
    show_communications     = true,
    show_scalars            = false,
    anti_deps               = false,
    data_as_edges           = false,
    show_regions            = false,
} = {}) {
    const settings = new GraphSettings();
    settings.mode          = mode;
    settings.antideps      = anti_deps;
    settings.allocations   = container_allocations;
    settings.deallocations = container_deallocations;
    settings.transfers     = container_transfers;
    settings.communications = show_communications;
    settings.updates       = !data_as_edges;        // data_as_edges=true → updates=false (edges only)
    settings.scalars       = show_scalars;
    settings.regions       = show_regions;
    settings.synthesizeRootRegion = (mode == 'program-tree');
    return settings;
}

/**
 * Group the calls that fusion.js finds fusable, with the containers that fusion makes unnecessary, in
 * compound 'fusion' nodes of the graph, and note on each call why it was not fused with the consumers
 * of its output. A group whose calls are in different regions is not drawn (it would take them out
 * of their regions). Returns the hints [consumer node id, producer node id] of the fused pairs.
 */
export function applyFusion(graph, events) {
    const { ix, serial, parallel, blocked } = analyzeFusion(events.map((e, i) => ({ ...e, idx: i })));

    const calls = new Map(), updates = new Map();
    for (const n of graph.getAllNodes()) {
        if (n.type === 'skeleton_call') calls.set(n.traceIndex, n);
        else if (n.type === 'container_update') updates.set(`${n.traceIndex}:${n.objectId}`, n);
    }

    const hints = [];
    for (const g of [...serial, ...parallel]) {
        const cs = g.calls.map(i => calls.get(i));
        if (cs.some(c => !c) || cs.some(c => c.region !== cs[0].region)) continue;
        const ups = g.pairs.map(p => updates.get(`${p.A.idx}:${p.C}`)).filter(Boolean);

        const last = cs[cs.length - 1];
        const parent = cs[0].region instanceof RegionNode ? cs[0].region : null;
        const fused = new FusedNode(graph, g.kind === 'serial' ? 'Serial Fusion' : 'Parallel Fusion', cs[0].totalOrder, parent);
        fused.fusionPattern = last.pattern;
        fused.persistentFusionId = 'fusion_' + cs[0].key;   // names the group for collapsing it, the same on every build of the graph
        fused.fusionInfo = {
            kind: g.kind, placement: g.placement, calls: cs.map(c => c.label).join(', '),
            intermediates: g.pairs.map(p => ix.labelOf(p.C)), launchesSaved: cs.length - 1,
            recompute: g.pairs.filter(p => p.recompute).map(p => `${p.A.label} into ${p.B.label}: ${p.recompute.note}`),
        };
        if (g.kind === 'parallel') for (const c of cs) fused.elementSizes.push(c.elements);
        // The iterations of a loop are coalesced by key path, which has the label of the node and this: the
        // groups of two iterations are the same group, those of different calls are not.
        fused.keypathDiscriminator = cs.map(c => `${c.label}@${c.line ?? ''}`).join('+');

        for (const m of [...cs, ...ups]) {
            if (m.region instanceof RegionNode) {
                const i = m.region.nestedNodes.indexOf(m);
                if (i !== -1) m.region.nestedNodes.splice(i, 1);
            }
            m.fused  = fused;
            m.region = fused;
        }
        for (const p of g.pairs) hints.push([calls.get(p.B.idx).id, calls.get(p.A.idx).id]);
    }

    for (const b of blocked) {
        const producer = calls.get(b.producer);
        if (!producer || b.consumer == null) continue;
        (producer.fusionNotes ??= []).push({
            reason: b.reason, consumer: events[b.consumer].label,
            text: explainBlocked(b.reason, ix.labelOf(b.container), b.patterns),
        });
    }
    return hints;
}

/**
 * Build and analyse the graph of the events. The events' file paths are shortened in place
 * (idempotent). Options: those of graphSettings() and
 *   fusion_analysis    group the chains of calls that could be fused
 *   collapse_iteration merge the iterations of a loop into one node
 *   aggregate_by       what makes two nodes the same operation: 'name' (region, type and label) or
 *                      'name-location' (and the file and line)
 *   show_virtual_alias draw the edges between the members of a virtual data set
 * Returns { graph, fusionHints }.
 */
export function buildDisplayGraph(events, options = {}) {
    const { collapse_iteration = false, fusion_analysis = false, show_virtual_alias = false, aggregate_by = 'name' } = options;
    const byLocation = aggregate_by === 'name-location';

    preprocessFilePaths(events);
    const graph = DirectedGraph.fromEvents(events, graphSettings(options));
    graph.instances = options.instances ?? new Map();   // where the instances are defined: Map by instance id (headerInstances)

    graph.computeDepths();
    graph.findCriticalPath();
    graph.computeKeyPaths(byLocation);

    const fusionHints = fusion_analysis ? applyFusion(graph, events) : [];

    // Compute region extents before coalescing so each iteration gets its own span.
    if (graph.settings.regions) {
        for (const node of graph.getAllNodes()) {
            if (node instanceof RegionNode && !node.region)
                node.computeExtent();
        }
    }

    if (collapse_iteration) {
        graph.coalesceIterations(byLocation);
    } else {
        graph.numberRegionLabels();
    }

    if (graph.settings.regions) graph.pruneEmptyRegions();

    graph.createVirtualAliasEdges(show_virtual_alias);

    return { graph, fusionHints };
}
