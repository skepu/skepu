// pipeline.js — from the events of a trace to the analysed graph that the views show.
//
// What the worker does for a 'graph' request before it serializes the result, without any
// worker, I/O or state of its own: the same steps can be run in Node or in tests.
'use strict';

import { DirectedGraph, GraphSettings } from './graph.js';
import { RegionNode } from './node.js';
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
    settings.updates       = !data_as_edges;        // data_as_edges=true → updates=false (edges only)
    settings.scalars       = show_scalars;
    settings.regions       = show_regions;
    settings.synthesizeRootRegion = (mode == 'program-tree');
    return settings;
}

/**
 * Build and analyse the graph of the events. The events' file paths are shortened in place
 * (idempotent). Options: those of graphSettings() and
 *   fusion_analysis    group the chains of calls that could be fused
 *   collapse_iteration merge the iterations of a loop into one node
 *   show_virtual_alias draw the edges between the members of a virtual data set
 * Returns { graph, fusionHints }.
 */
export function buildDisplayGraph(events, options = {}) {
    const { collapse_iteration = false, fusion_analysis = false, show_virtual_alias = false } = options;

    preprocessFilePaths(events);
    const graph = DirectedGraph.fromEvents(events, graphSettings(options));

    graph.computeDepths();
    graph.findCriticalPath();
    graph.computeKeyPaths();

    const fusionHints = fusion_analysis ? graph.findFusions() : [];

    // Compute region extents before coalescing so each iteration gets its own span.
    if (graph.settings.regions) {
        for (const node of graph.getAllNodes()) {
            if (node instanceof RegionNode && !node.region)
                node.computeExtent();
        }
    }

    if (collapse_iteration) {
        graph.coalesceIterations();
    } else {
        graph.numberRegionLabels();
    }

    if (graph.settings.regions) graph.pruneEmptyRegions();

    graph.createVirtualAliasEdges(show_virtual_alias);

    return { graph, fusionHints };
}
