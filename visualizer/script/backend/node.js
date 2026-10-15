// node.js — Graph node types.  Browser/Worker compatible ES module.
// Ported from backend/node.py.
'use strict';

import {
    ElwiseEdge, ProxyEdge, ScalarEdge, AntiDepEdge, PrngEdge, SetupEdge, VirtualAliasEdge,
} from './edge.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function _stddev(values, mean) {
    if (values.length < 2) return 0;
    const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
    return Math.sqrt(variance);
}

function _durationCV(durations) {
    if (durations.length < 2) return 0.0;
    const mean = durations.reduce((s, d) => s + d, 0) / durations.length;
    if (mean <= 0) return 0.0;
    return _stddev(durations, mean) / mean;
}

// ── Base node ─────────────────────────────────────────────────────────────────

/** What the ranks did in the iterations of a call (mpi.js): the mean imbalance, the slowest rank of the worst iteration, the bytes that a rank moved. Null if there is nothing. */
function rankStats(summaries) {
    const present = summaries.filter(s => s && s.work);
    if (!present.length) return null;
    const imbalances = present.map(s => s.imbalance).filter(v => v != null);
    const worst = present.reduce((a, s) => ((s.imbalance ?? 0) > (a.imbalance ?? 0) ? s : a));
    const bytes = present.map(s => s.comm_bytes?.max ?? 0);
    return {
        imbalance:    imbalances.length ? imbalances.reduce((a, b) => a + b, 0) / imbalances.length : null,
        slowest_rank: worst.slowest_rank,
        comm_bytes:   bytes.reduce((a, b) => a + b, 0) / bytes.length,
    };
}

/** The source locations of the iterations of a node as [file, line] pairs, if they are not all its own. */
function badgeLocations(node) {
    const locs = node.locations;
    if (!locs || locs.every(l => l.file === node.file && l.line === node.line)) return null;
    return locs.map(l => [l.file, l.line]);
}

export class Node {
    constructor(graph, label, totalOrder, region = null) {
        this.id          = crypto.randomUUID();
        this.key         = graph.newNodeKey();   // stable, unlike id: see DirectedGraph.newNodeKey()
        this.label       = label;
        this.graph       = graph;
        this.totalOrder  = totalOrder;
        this._incoming   = [];
        this._outgoing   = [];
        this.depth       = 0;
        this.fused       = null;
        this.traceIndex  = graph.currentTraceIndex();

        // keypath: set by computeKeyPaths(); coalesceIterations() reads it.
        this.keypath     = '';
        this.keypathDiscriminator = '';

        // elements: ComputationNode sets this explicitly before using it.
        // Guard here so non-computation nodes have a defined value.
        if (this.elements === undefined) this.elements = null;

        this.isCriticalPath  = false;
        this.hasSnapshot     = false;
        this.iterationCount  = 1;
        this.totalOrders     = [totalOrder];
        this.traceIndices    = [this.traceIndex];
        this.durations       = [];
        this.intervals       = [];

        this.treeParents     = [];

        // Region nesting
        this.region       = null;
        this.nestingLevel = 0;
        if (graph.settings.regions) {
            this.region = graph.getRootRegion();
            if (region instanceof RegionNode) {
                this.region = region;
            } else if (region) {
                this.region = graph.getNodeByLabel(region);
            }
            if (this.region) {
                this.region.addNestedNode(this);
                this.nestingLevel = this.region.nestingLevel + 1;
            }
        }

        // program-order spanning-tree building
        // assumes graph settings has root region synthethis enabled
        if (graph.settings.regions && graph.settings.synthesizeRootRegion)
        {
          if (this.region) {
            var previousNode = graph.getLastNode();
            if (previousNode.region && this.region.id == previousNode.region.id)
              this.treeParents.push(previousNode);
            else
              this.treeParents.push(this.region);
          }
        }

        graph.addNode(this);
    }

    addIncomingEdge(edge) { this._incoming.push(edge); }
    addOutgoingEdge(edge) { this._outgoing.push(edge); }
    getIncomingEdges()    { return this._incoming; }
    getOutgoingEdges()    { return this._outgoing; }
    getParentNodes()      { return this._incoming.map(e => e.source); }
    getChildNodes()       { return this._outgoing.map(e => e.target); }

    toCytoscape() {
        const d = {
            id:               this.id,
            label:            this.label,
            total_order:      this.totalOrder,
            type:             this.type,
            is_critical_path: this.isCriticalPath,
            dag_depth:        this.depth,
            iteration_count:  this.iterationCount,
            trace_index:      this.traceIndex,
            nesting_level:    this.nestingLevel,
            fan_in:           this._incoming.length,
            fan_out:          this._outgoing.length,
        };
        if (this.totalOrders.length > 1) {
            d.total_orders   = this.totalOrders;
            d.trace_indices  = this.traceIndices;
        }
        if (this.fused) {
            d.parent = this.fused.id;
        } else if (this.region) {
            d.parent       = this.region.id;
            d.region_depth = this.nestingLevel;
        }
        return d;
    }

    /** Return a plain object for the Gantt timeline, or null if not applicable. */
    toTimeline()    { return null; }

    /** Return a plain object for source-gutter badge placement, or null to exclude. */
    toSourceBadge() { return null; }

    infoData() {
        const info = {
            internal:    {},
            type:        this.type,
            trace_index: this.traceIndex,
        };
        if (this.iterationCount > 1) info['Repeat count'] = this.iterationCount;
        if (this.region) {
            info['Region']        = this.graph.labelForID(this.region.id);
            info['Nesting Level'] = this.nestingLevel;
        }
        return info;
    }
}

// ── ComputationNode ───────────────────────────────────────────────────────────

export class ComputationNode extends Node {
    // elements and pattern are passed explicitly so CallNode can provide them
    // before the output loop runs.  The output loop calls hasScalarOutput()
    // (which reads this.pattern) and creates UpdateNodes (which read this.elements),
    // both of which must be set before then.
    constructor(graph, jsonData, totalOrder, elements = null, pattern = null) {
        super(graph, jsonData.label, totalOrder, jsonData.region ?? null);

        // Must be set BEFORE the output loop (UpdateNode reads this.elements,
        // hasScalarOutput() reads this.pattern).
        this.elements = elements;
        this.pattern  = pattern;

        this.start    = jsonData.start;
        this.end      = jsonData.end;
        this.duration = this.end - this.start;
        this.durations = [this.duration];
        this.intervals = [[this.start, this.end]];
        // What every rank did in each iteration, for the trace of an MPI program (mpi.js); null for a trace of a single node
        this.rankDatas = [jsonData.rank_data ?? null];
        this.rankSummaries = [jsonData.rank_summary ?? null];

        this.backend = jsonData.backend ?? 'CPU';
        this.file    = jsonData.file ?? null;
        this.line    = jsonData.line ?? null;

        const elwiseInputs = jsonData.elwise_inputs ?? [];
        const proxyInputs  = jsonData.proxy_inputs  ?? [];
        const scalarInputs = jsonData.scalar_inputs  ?? [];
        const outputs      = jsonData.outputs        ?? [];

        // Of an MPI program: host code (an skepu::external call) reads the gathered copy of a container, and so does a random access
        // argument; the elements of a skeleton are in the tiles of the processes (the memory space of its backend). A trace of a single
        // node has no gathered copy, so this reads the same as before.
        const host = jsonData.host === true;
        const replicaProxies = jsonData.replica_inputs ?? true;
        for (const id of elwiseInputs) {
            new ElwiseEdge(graph, graph.labelForID(id), id, this, host);
            graph.readFromID(id, this, this.backend);
        }
        for (const id of proxyInputs) {
            new ProxyEdge(graph, graph.labelForID(id), id, this, replicaProxies);
            graph.readFromID(id, this, this.backend);
        }
        for (const id of scalarInputs) {
            new ScalarEdge(graph, graph.labelForID(id), id, this);
            graph.readFromID(id, this, 'ALL');
        }

        for (const id of outputs) {
            if (graph.settings.antideps && !elwiseInputs.includes(id)) {
                new AntiDepEdge(graph, graph.labelForID(id), id, this);
            }
            if (graph.settings.updates) {
                if (!this.hasScalarOutput()) {
                    new ElwiseEdge(graph, graph.labelForID(id), this,
                        new UpdateNode(graph, id, this.totalOrder + 1,
                                       this.region, this.backend, this.elements));
                } else {
                    new ScalarEdge(graph, graph.labelForID(id), this,
                        ScalarNode.fromID(graph, id, this.totalOrder + 1, this.region, this, outputs.indexOf(id)));
                }
            } else {
                graph.writeToID(id, this, this.hasScalarOutput() ? 'ALL' : this.backend);
            }
            // (what host code writes is in the gathered copy of the first process, until it is given to the owners)
            if (host) graph.writeReplica(id, graph.producerForID(id));
        }

        if (jsonData.prng != null) {
            const id = jsonData.prng;
            new PrngEdge(graph, graph.labelForID(id), id, this);
            graph.writeToID(id, this, this.backend);
            const stream = graph._allocationNodes.get(id);
            if (stream) stream.isPrng = true;              // the allocation of a random stream: it is drawn in its own colour
        }
    }

    hasScalarOutput() { return false; }

    toCytoscape() {
        return {
            ...super.toCytoscape(),
            start:       this.start,
            end:         this.end,
            duration:    this.duration,
            durations:   this.durations,
            intervals:   this.intervals,
            file:        this.file,
            line:        this.line,
            backend:     this.backend,
            duration_cv: _durationCV(this.durations),
            ...(this._rankStats() ?? {}),        // (imbalance, slowest_rank, comm_bytes: a call of an MPI program)
        };
    }

    _rankStats() { return rankStats(this.rankSummaries ?? []); }

    toTimeline() {
        return {
            id:              this.id,
            label:           this.label,
            type:            this.type,
            start:           this.start,
            end:             this.end,
            duration:        this.duration,
            durations:       this.durations,
            duration_cv:     _durationCV(this.durations),
            intervals:       this.intervals,
            ...(this.rankDatas.some(Boolean) ? { rank_datas: this.rankDatas } : {}),
            backend:         this.backend,
            dag_depth:       this.depth,
            total_order:     this.totalOrder,
            fan_in:          this._incoming.length,
            fan_out:         this._outgoing.length,
            nesting_level:   this.nestingLevel,
            iteration_count: this.iterationCount,
        };
    }

    toSourceBadge() {
        const data = {
            id:          this.id,
            type:        this.type,
            trace_index: this.traceIndex,
            total_order: this.totalOrder,
        };
        if (this.totalOrders.length > 1) {
            data.total_orders   = this.totalOrders;
            data.trace_indices  = this.traceIndices;
        }
        if (this.file) {
            data.file = this.file;
            data.line = this.line;
        }
        const locations = badgeLocations(this);
        if (locations) data.locations = locations;
        return data;
    }

    infoData() {
        return {
            ...super.infoData(),
            Duration:  this.duration,
            File:      this.file,
            Line:      this.line,
            Backend:   this.backend,
            durations: this.durations.map((d, i) => ({ x: i, y: d })),
            ...this._rankInfo(),
        };
    }

    // What every rank did in a call of an MPI program: the imbalance, and the work of each rank (in the first iteration)
    _rankInfo() {
        const s = this._rankStats();
        const data = this.rankDatas?.find(Boolean);
        if (!s || !data) return {};
        const fmt = us => us >= 1e3 ? (us / 1e3).toFixed(1) + ' ms' : us.toFixed(0) + ' us';
        const first = this.rankDatas.filter(Boolean).length > 1 ? ' (first iteration)' : '';
        return {
            'Imbalance (longest work / mean over ranks)': +s.imbalance?.toFixed(2),
            'Slowest rank': s.slowest_rank,
            ['Work per rank' + first]: data.map(e => e ? `${e.rank}: ${e.work_start != null ? fmt(e.work_end - e.work_start) : 'none'}` : null).filter(Boolean).join('; '),
            ['Elements per rank' + first]: data.map(e => e ? `${e.rank}: ${e.local_elements}` : null).filter(Boolean).join('; '),
            'Communication per rank (bytes, mean of the busiest)': Math.round(s.comm_bytes),
        };
    }
}

// ── CallNode ──────────────────────────────────────────────────────────────────

export class CallNode extends ComputationNode {
    constructor(graph, jsonData, totalOrder) {
        // Pass elements (4th) and pattern (5th) so ComputationNode can set them
        // before the output loop runs.  hasScalarOutput() reads this.pattern, and
        // UpdateNode creation reads this.elements — both happen inside super().
        super(graph, jsonData, totalOrder,
              jsonData.elements ?? null,
              jsonData.pattern  ?? null);
        this.type = 'skeleton_call';
        this.instanceId = jsonData.object_id ?? null;   // the skeleton instance (what the table of the instances in the header is by)
        // this.elements and this.pattern already set by ComputationNode
        this.parameters = jsonData.parameters ?? null;   // what is not the default: overlap, strides, modes (see TRACE_FORMAT.md)
    }

    hasScalarOutput() {
        // (a reduction of rows or columns, 'Reduce 1D-Partial', has a vector)
        return ['Reduce', 'Reduce 1D-Full', 'Reduce 2D', 'MapReduce'].includes(this.pattern);
    }

    toCytoscape() {
        return { ...super.toCytoscape(), pattern: this.pattern, elements: this.elements };
    }

    toTimeline() {
        return { ...super.toTimeline(), pattern: this.pattern, elements: this.elements };
    }

    // (the badge nodes also say where the instance and the user functions of the call are, for the marks in the source panes)
    toSourceBadge() {
        const badge = super.toSourceBadge();
        const d = badge && this.instanceId != null ? this.graph.instances?.get(this.instanceId) : null;
        if (d) badge.definitions = { instance: d.instance, functions: d.user_functions };
        return badge;
    }

    infoData() {
        const info = { ...super.infoData(), Pattern: this.pattern, Elements: this.elements };
        // where the instance and its user functions are defined (the info pane has links to them)
        const definitions = this.instanceId != null ? this.graph.instances?.get(this.instanceId) : null;
        if (definitions) info.definitions = definitions;
        if (this.parameters)
            info['Parameters'] = Object.entries(this.parameters).map(([k, v]) => `${k.replace(/_/g, ' ')}: ${Array.isArray(v) ? v.join(' x ') : v}`).join('; ');
        // Set by the fusion analysis: the pairs with a consumer of this call's output that could not be fused.
        if (this.fusionNotes?.length) {
            const lines = [...new Set(this.fusionNotes.map(n => `with "${n.consumer}": ${n.text}`))];
            info['Not fused'] = lines.slice(0, 5).join('; ') + (lines.length > 5 ? `; and ${lines.length - 5} more` : '');
        }
        return info;
    }
}

// ── ExternalNode ──────────────────────────────────────────────────────────────

export class ExternalNode extends ComputationNode {
    constructor(graph, jsonData, totalOrder) {
        super(graph, { ...jsonData, host: true }, totalOrder);
        this.type = 'external';
        this.keypathDiscriminator = this.line;
    }
    // toCytoscape delegates to super intentionally.
}

// ── CommunicationNode ─────────────────────────────────────────────────────────
// A collective operation of the MPI backend that moves a container (or its rows) between processes: the gather
// of a container that the host or a random access argument needs, the exchange of the rows that a call needs
// (a MatRow, a halo, the layout of a call), what a skepu::external gives back to the owners. A container of an MPI
// program is in two memory spaces: the tiles of the processes (the memory space of the backend) and the gathered
// copy (REPLICA in graph.js). A gather or a broadcast makes the copy and leaves the tiles as they were, so what
// reads the tiles (the elements of a skeleton) does not depend on it, and what reads the copy (the host, a random
// access argument) does. A scatter, and the exchange after a call, make the tiles. A write makes the copy out of date.
// (A trace of a single node has none.)

export class CommunicationNode extends ComputationNode {
    constructor(graph, jsonData, totalOrder) {
        const ids = jsonData.object_ids ?? [];
        const what = ids.map(id => graph.labelForID(id)).join(', ');
        super(graph, {
            label:        what ? `${jsonData.operation}: ${what}` : jsonData.operation,
            region:       jsonData.region,
            start:        jsonData.start,
            end:          jsonData.end,
            backend:      'ANY',                      // (reads what is there, in whichever memory space)
            proxy_inputs: ids,
            // (a gather reads the tiles; a broadcast and what gives the data of the first process to the owners read the gathered copy)
            replica_inputs: jsonData.operation === 'bcast' || jsonData.operation === 'scatter',
            rank_data:    jsonData.rank_data,         // (what each rank did in it, for the timeline)
        }, totalOrder);

        this.type        = 'communication';
        this.operation   = jsonData.operation;
        this.reason      = jsonData.reason ?? null;
        this.commSeq     = jsonData.comm_seq ?? null;
        this.callSeq     = jsonData.seq || null;     // (the call it is made in; 0 or none for host code)
        this.phase       = jsonData.phase ?? null;   // before or after the work of that call
        this.bytesSent   = jsonData.bytes_sent ?? 0;
        this.bytesReceived = jsonData.bytes_received ?? 0;
        this.ranks       = jsonData.ranks ?? jsonData.rank_data?.length ?? null;
        // What every process did, if the trace files of all of them were merged (mpi.js)
        this.rankSummary = jsonData.rank_summary ?? null;
        this.rankSummaries = [this.rankSummary];
        this.keypathDiscriminator = `${this.operation}:${this.reason}`;
        this.file = null;
        this.line = null;

        // (it is on the side of the host that made it, for the edges)
        this.backend = this._incoming[0]?.source.backend ?? 'CPU';
        // What the container is after it: a gather or a broadcast makes a gathered copy, and the tiles are as they were. What gives the data to
        // the owners (the scatter, and the exchange after a call that puts its results in their places) makes the tiles. The exchange before
        // a call (a halo, the rows of a MatRow) only reads. Anything else is taken to have made the container, as before.
        const op = this.operation;
        for (const id of ids) {
            if (op === 'allgather' || op === 'gather' || op === 'bcast') graph.writeReplica(id, this);
            else if (op === 'alltoallv' && this.phase !== 'after') continue;
            else graph.writeToID(id, this, 'ALL');
        }
    }

    toCytoscape() {
        const d = {
            ...super.toCytoscape(),
            operation:      this.operation,
            reason:         this.reason,
            bytes_sent:     this.bytesSent,
            bytes_received: this.bytesReceived,
        };
        if (this.ranks != null) d.ranks = this.ranks;
        if (this.rankSummary) {
            d.max_wait_us = this.rankSummary.max_wait_us;
            d.bytes_total = this.rankSummary.bytes_sent;
        }
        return d;
    }

    toTimeline() {
        // (the collective that the processes make: when they arrived and when they left, in rank_summary, for each iteration)
        return { ...super.toTimeline(), operation: this.operation, ...(this.rankSummary ? { rank_summaries: this.rankSummaries } : {}) };
    }

    infoData() {
        const info = super.infoData();
        delete info.File; delete info.Line; delete info.Backend;
        const s = this.rankSummary;
        return {
            ...info,
            Operation: this.operation,
            Reason:    this.reason,
            Ranks:     this.ranks,
            // what this process (rank 0) sent and received; the total of all processes when they are merged
            'Bytes sent (rank 0)':     this.bytesSent,
            'Bytes received (rank 0)': this.bytesReceived,
            ...(s ? {
                'Bytes moved (all ranks)':     s.bytes_sent,
                'Longest wait (us)':           Math.round(s.max_wait_us),
                'Last to arrive (rank)':       s.last_arrival_rank,
                'Ranks leave within (us)':     Math.round(s.end_spread_us),
            } : {}),
        };
    }
}

// ── RegionNode ────────────────────────────────────────────────────────────────

export class RegionNode extends Node {
    constructor(graph, jsonData, totalOrder) {
        super(graph, jsonData.label, totalOrder, null);   // always null region for now

        if (!graph.settings.regions) {
            // Regions disabled: undo node registration and exit.
            graph._removeNode(this);
            return;
        }

        this.treeParents = (this.region) ? [this.region] : [];

        // Parent region nesting (the super() call is given null).
        const parentLabel = jsonData.region ?? null;
        if (parentLabel) {
            const parentRegion = graph.getNodeByLabel(parentLabel);
            if (parentRegion instanceof RegionNode) {
                this.region = parentRegion;
                parentRegion.addNestedNode(this);
                this.nestingLevel = parentRegion.nestingLevel + 1;
            }
        }

        graph.setLabelForID(jsonData.label, this.id);
        this.type        = 'region';
        this.regionDepth = jsonData.region_depth;
        this.nestedNodes = [];
        this.regionStart     = null;
        this.regionEnd       = null;
        this.regionIntervals = [];
        this.file = jsonData.file ?? null;
        this.line = jsonData.line ?? null;

        this.persistentRegionId = 'region_' + graph.nextRegionNumber();
    }

    addNestedNode(node)   { this.nestedNodes.push(node); }
    getDirectChildren()   { return this.nestedNodes.filter(n => !(n instanceof RegionNode)); }

    computeExtent() {
        const starts = [], ends = [];
        for (const node of this.nestedNodes) {
            if (node instanceof RegionNode) {
                node.computeExtent();
                if (node.regionStart != null) starts.push(node.regionStart);
                if (node.regionEnd   != null) ends.push(node.regionEnd);
            } else if (node instanceof ComputationNode) {
                starts.push(node.start);
                ends.push(node.end);
                // (of an MPI program: what the other ranks did in the call, which can be later than rank 0 did)
                for (const summary of node.rankSummaries ?? []) {
                    if (!summary?.interval) continue;
                    starts.push(summary.interval.start);
                    ends.push(summary.interval.end);
                }
            } else if (node.timestamp != null) {
                starts.push(node.timestamp);
                ends.push(node.timestamp);
            }
        }
        this.regionStart = starts.length ? Math.min(...starts) : null;
        this.regionEnd   = ends.length   ? Math.max(...ends)   : null;
        if (this.regionStart != null) {
            this.regionIntervals.push([this.regionStart, this.regionEnd]);
        }
    }

    toCytoscape() {
        return {
            ...super.toCytoscape(),
            region_depth:        this.regionDepth,
            nesting_level:       this.nestingLevel,
            region_start:        this.regionStart,
            region_end:          this.regionEnd,
            region_intervals:    this.regionIntervals,
            persistent_region_id: this.persistentRegionId,
            file:                this.file,
            line:                this.line,
        };
    }

    toTimeline() {
        return {
            id:               this.id,
            label:            this.label,
            type:             this.type,
            region_start:     this.regionStart,
            region_end:       this.regionEnd,
            region_intervals: this.regionIntervals,
            region_depth:     this.regionDepth,
            nesting_level:    this.nestingLevel,
            iteration_count:  this.iterationCount,
        };
    }

    toSourceBadge() {
        const data = {
            id:          this.id,
            type:        this.type,
            trace_index: this.traceIndex,
        };
        if (this.totalOrders.length > 1) {
            data.total_orders  = this.totalOrders;
            data.trace_indices = this.traceIndices;
        }
        if (this.file) {
            data.file = this.file;
            data.line = this.line;
        }
        const locations = badgeLocations(this);
        if (locations) data.locations = locations;
        return data;
    }

    infoData() {
        return { ...super.infoData(), File: this.file, Line: this.line };
    }
}

// ── FusedNode ─────────────────────────────────────────────────────────────────

export class FusedNode extends Node {
    constructor(graph, label, totalOrder, region) {
        super(graph, label, totalOrder, region);
        this.type          = 'fusion';
        this.nestedNodes   = [];
        this.elementSizes  = [];    // one entry per participating Map
        this.fusionPattern = null;  // skeleton pattern of the sink stage
        graph.setLabelForID(label, this.id);
    }

    addNestedNode(node) { this.nestedNodes.push(node); }

    toCytoscape() {
        const d = { ...super.toCytoscape() };
        if (this.persistentFusionId) d.persistent_fusion_id = this.persistentFusionId;
        if (this.fusionPattern) {
            d.fusion_pattern = this.fusionPattern;
        }
        if (this.elementSizes.length) {
            d.element_sizes    = this.elementSizes;
            d.uniform_elements = new Set(this.elementSizes.map(s => JSON.stringify(s))).size === 1;
        }
        return d;
    }

    infoData() {
        const info = { ...super.infoData() };
        if (this.fusionPattern) info['Fusion pattern'] = this.fusionPattern;
        // Set by the fusion analysis: see fusion.js
        const f = this.fusionInfo;
        if (f) {
            info['Fusion kind'] = f.kind + (f.placement ? ` (the fused call goes to the place of the ${f.placement === 'sink' ? 'last' : 'first'} call)` : '');
            info['Calls'] = f.calls;
            if (f.intermediates.length) info['Not written or read'] = f.intermediates.join(', ');
            info['Launches saved'] = f.launchesSaved;
            if (f.recompute?.length) info['Recomputation'] = f.recompute.join('; ');
        }
        return info;
    }
}

// ── DataNode (abstract base for non-computation data nodes) ───────────────────

export class DataNode extends Node {
    constructor(graph, label, totalOrder, region) {
        super(graph, label, totalOrder, region);
        this.backend = 'ANY';
    }
    // toCytoscape delegates to super intentionally.
}

// ── AllocationNode ────────────────────────────────────────────────────────────

export class AllocationNode extends DataNode {
    constructor(graph, jsonData, totalOrder) {
        // setLabelForID must happen even when allocations=false (Python does this).
        const isInternal = jsonData.label.toLowerCase().includes('internal');
        if (!isInternal) {
            graph.setLabelForID(jsonData.label, jsonData.object_id);
        }

        // In JS, super() must be called before returning.  Call it, then undo
        // registration if this node should be skipped.
        super(graph, jsonData.label, totalOrder, null);

        if (isInternal || !graph.settings.allocations) {
            graph._removeNode(this);
            return;
        }

        this.type      = 'allocation';
        this.timestamp = jsonData.time ?? null;
        this.isPrng    = false;
        graph.writeToID(jsonData.object_id, this, 'ALL');
        graph._allocationNodes.set(jsonData.object_id, this);
    }

    toCytoscape() {
        return { ...super.toCytoscape(), timestamp: this.timestamp ?? null, ...(this.isPrng ? { is_prng: true } : {}) };
    }

    toTimeline() {
        return {
            id:            this.id,
            label:         this.label,
            type:          this.type,
            timestamp:     this.timestamp ?? null,
            nesting_level: this.nestingLevel,
        };
    }

    toSourceBadge() {
        const data = { id: this.id, type: this.type, trace_index: this.traceIndex };
        if (this.totalOrders.length > 1) {
            data.total_orders  = this.totalOrders;
            data.trace_indices = this.traceIndices;
        }
        return data;
    }

    infoData() { return super.infoData(); }
}

// ── DeallocationNode ──────────────────────────────────────────────────────────

export class DeallocationNode extends DataNode {
    constructor(graph, jsonData, totalOrder) {
        const isInternal = jsonData.label.toLowerCase().includes('internal');

        super(graph, jsonData.label, totalOrder, null);

        if (isInternal || !graph.settings.deallocations) {
            graph._removeNode(this);
            return;
        }

        this.type      = 'deallocation';
        this.timestamp = jsonData.time ?? null;
        this.backend   = 'ANY';
        new AntiDepEdge(graph, jsonData.label,
                        graph.accessorForID(jsonData.object_id), this);
    }

    toCytoscape() {
        return { ...super.toCytoscape(), timestamp: this.timestamp ?? null };
    }

    toTimeline() {
        return {
            id:            this.id,
            label:         this.label,
            type:          this.type,
            timestamp:     this.timestamp ?? null,
            nesting_level: this.nestingLevel,
        };
    }

    toSourceBadge() {
        const data = { id: this.id, type: this.type, trace_index: this.traceIndex };
        if (this.totalOrders.length > 1) {
            data.total_orders  = this.totalOrders;
            data.trace_indices = this.traceIndices;
        }
        return data;
    }
}

// ── UpdateNode ────────────────────────────────────────────────────────────────

/**
 * What a data node that a skeleton call made shares with the call: its source line and the definitions of its instance and user functions, so
 * that selecting the node marks the same lines in the source as selecting the call. Null if no call made it (a host write, or an MPI collective).
 */
function producerInfo(node) {
    for (const e of node.getIncomingEdges()) {
        const call = e.source;
        if (!(call instanceof ComputationNode) || !call.file || !(e instanceof ElwiseEdge || e instanceof ScalarEdge)) continue;
        const info = call.infoData();
        const shared = { File: info.File, Line: info.Line };
        if (info.definitions) shared.definitions = info.definitions;
        return shared;
    }
    return null;
}

export class UpdateNode extends DataNode {
    constructor(graph, id, totalOrder, region, backend, elements = null) {
        const label = graph.labelForID(id);
        super(graph, label, totalOrder, region);
        this.type              = 'container_update';
        this.version           = graph.newVersionNumberForID(id);
        this._versionsOfGraph  = graph.update_versions;    // (how many versions the container has, when the graph is complete)
        this.backend           = backend;
        this.elements          = elements;
        this.isLive            = [];
        this.objectId          = id;
        this.virtualId         = null;
        this.snapshotDataFiles = [];
        graph.writeToID(id, this, backend);
    }

    toCytoscape() {
        const d = { ...super.toCytoscape(), version: this.version, version_count: this._versionsOfGraph[this.objectId] };
        if (this.elements != null) d.elements = this.elements;
        return d;
    }

    infoData() {
        const info = super.infoData();
        info['Version'] = this.version;
        info.internal.is_live = this.isLive;
        return Object.assign(info, producerInfo(this));
    }
}

// ── ScalarNode ────────────────────────────────────────────────────────────────

export class ScalarNode extends DataNode {
    constructor(graph, label, totalOrder, region) {
        super(graph, label, totalOrder, region);
        this.type = 'scalar';
        this.scalarValues = [];   // [total order, value] of each of its iterations (a scalar of one call that is repeated has a node)
    }

    /** The value of the scalar that the trace has for the id, as [total order, value] (nothing if it has none). */
    _takeValue(objectId, totalOrder) {
        const value = this.graph.scalarValues?.get(objectId);
        if (value !== undefined) this.scalarValues.push([totalOrder, value]);
    }

    /** Construct from a raw element_access trace event. */
    static fromEventData(graph, jsonData, totalOrder, region = null) {
        const label = jsonData.label + '[' + jsonData.index + ']';
        if (graph.settings.scalars) {
            const node = new ScalarNode(graph, label, totalOrder, region);
            node.keypathDiscriminator = `${jsonData.file ?? ''}:${jsonData.line ?? ''}`;   // (where the program read the element)
            node._takeValue(jsonData.object_id, totalOrder);
            new ScalarEdge(graph, label, jsonData.source, node, true);   // (the host reads the gathered copy)
            graph.readFromID(jsonData.source, node, 'ALL');
            graph.writeToID(jsonData.object_id, node, 'ALL');
        } else {
            graph.setLabelForID(label, jsonData.object_id);
            graph.writeToID(jsonData.object_id,
                            graph.replicaProducerForID(jsonData.source) ?? graph.producerForID(jsonData.source), 'ALL');
        }
    }

    /**
     * Construct a synthetic scalar output node for a Reduce/MapReduce. The scalars of a region are told apart (by the iteration coalescing,
     * which joins the nodes of one key path) by the call that made them and which of its results they are.
     */
    static fromID(graph, objectId, totalOrder, region, producer = null, slot = 0) {
        const label = 'scalar';
        const node  = new ScalarNode(graph, label, totalOrder, region);
        if (producer) node.keypathDiscriminator = `${producer.label}@${producer.line ?? ''}#${slot}`;
        node._takeValue(objectId, totalOrder);
        graph.setLabelForID(label, objectId);
        graph.writeToID(objectId, node, 'ALL');
        return node;
    }

    // toCytoscape delegates to super intentionally.

    /** The values of its iterations in order (numbers, or the strings 'nan', 'inf', '-inf'), and what made the scalar. */
    infoData() {
        const info = super.infoData();
        if (this.scalarValues.length > 0) {
            const values = [...this.scalarValues].sort((a, b) => a[0] - b[0]).map(v => v[1]);
            info.scalar_values = values;
            info['Value'] = values.length === 1 ? values[0] : `${values.length} values (last ${values[values.length - 1]})`;
        }
        const source = this._incoming.find(e => e.source && e.source.label);
        if (source) info['Result of'] = source.source.label;
        return Object.assign(info, producerInfo(this));
    }
}

// ── TransferNode ──────────────────────────────────────────────────────────────

export class TransferNode extends DataNode {
    constructor(graph, jsonData, totalOrder) {
        const isInternal    = jsonData.label.toLowerCase().includes('internal');
        const direction     = jsonData.direction;
        const objectId      = jsonData.object_id;
        const traceBackend  = jsonData.backend;
        const targetBackend = direction === 'device-to-host' ? 'CPU' : traceBackend;
        const sourceBackend = direction === 'device-to-host' ? traceBackend : 'CPU';

        super(graph, jsonData.label, totalOrder, jsonData.region);

        if (isInternal) {
            graph._removeNode(this);
            return;
        }

        if (!graph.settings.transfers) {
            graph._removeNode(this);
            graph.transferIDMemspace(objectId, sourceBackend, targetBackend);
            return;
        }

        // Apply region nesting (super used null; do it manually).
     /*    if (jsonData.region) {
            const parentRegion = graph.getNodeByLabel(jsonData.region);
            if (parentRegion instanceof RegionNode) {
                this.region = parentRegion;
                parentRegion.addNestedNode(this);
                this.nestingLevel = parentRegion.nestingLevel + 1;
            }
            }*/

        this.type      = 'transfer';
        this.direction = direction;
        this.start     = jsonData.start;
        this.end       = jsonData.end;
        this.duration  = this.end - this.start;
        this.intervals = [[this.start, this.end]];
        this.timestamp = jsonData.time ?? null;
        this.depth     = graph.depthByLabel(jsonData.label) + 1;
        this.backend   = sourceBackend;
        this.keypathDiscriminator = this.direction;

        new ProxyEdge(graph, jsonData.label, objectId, this);
        // Inherit element extents from the upstream producer.
        const producer = this._incoming[0]?.source ?? null;
        this.elements  = producer?.elements ?? null;
        graph.writeToID(objectId, this, targetBackend);
    }

    toCytoscape() {
        const d = {
            ...super.toCytoscape(),
            direction:  this.direction,
            timestamp:  this.timestamp,
            start:      this.start,
            end:        this.end,
            duration:   this.duration,
            intervals:  this.intervals,
        };
        if (this.elements != null) d.elements = this.elements;
        return d;
    }

    toTimeline() {
        return {
            id:              this.id,
            label:           this.label,
            type:            this.type,
            start:           this.start,
            end:             this.end,
            intervals:       this.intervals,
            direction:       this.direction,
            nesting_level:   this.nestingLevel,
            iteration_count: this.iterationCount,
        };
    }

    toSourceBadge() {
        const data = { id: this.id, type: this.type, trace_index: this.traceIndex };
        if (this.totalOrders.length > 1) {
            data.total_orders  = this.totalOrders;
            data.trace_indices = this.traceIndices;
        }
        return data;
    }

    infoData() {
        return {
            ...super.infoData(),
            Direction: this.direction,
            Duration:  this.duration,
            Backend:   this.backend,
        };
    }
}

// ── CompileNode ───────────────────────────────────────────────────────────────

/** The text of a build: of the instance ("dotprod"), and which of its kernels if it has several. */
export function compileLabel(ev) {
    return 'build: ' + ev.label + (ev.qualifier ? ' ' + ev.qualifier : '');
}

/**
 * The build of an OpenCL kernel for a device (a kernel_compile event). It has the id of the skeleton instance whose kernel it is, and is the
 * source of a setup edge to the first call of the instance (the kernels of the library have no instance, and no edge).
 */
export class CompileNode extends DataNode {
    constructor(graph, jsonData, totalOrder) {
        super(graph, compileLabel(jsonData), totalOrder, jsonData.region);

        if (!graph.settings.compiles) {
            graph._removeNode(this);
            return;
        }

        this.type       = 'compile';
        this.instanceId = jsonData.object_id ?? null;
        this.device     = jsonData.device ?? null;
        this.ok         = jsonData.ok !== false;
        this.error      = jsonData.error ?? null;
        this.options    = jsonData.options ?? null;
        this.sourceBytes = jsonData.source_bytes ?? null;
        this.start      = jsonData.start;
        this.end        = jsonData.end;
        this.duration   = this.end - this.start;
        this.intervals  = [[this.start, this.end]];
        this.timestamp  = jsonData.time ?? null;
        this.backend    = jsonData.backend ?? 'OpenCL';
        this.keypathDiscriminator = this.device ?? '';
    }

    toCytoscape() {
        return {
            ...super.toCytoscape(),
            device:    this.device,
            ok:        this.ok,
            timestamp: this.timestamp,
            start:     this.start,
            end:       this.end,
            duration:  this.duration,
            intervals: this.intervals,
        };
    }

    toTimeline() {
        return {
            id:              this.id,
            label:           this.label,
            type:            this.type,
            start:           this.start,
            end:             this.end,
            intervals:       this.intervals,
            device:          this.device,
            ok:              this.ok,
            nesting_level:   this.nestingLevel,
            iteration_count: this.iterationCount,
        };
    }

    infoData() {
        const info = { ...super.infoData(), Backend: this.backend, Device: this.device, Duration: this.duration };
        if (!this.ok) info['Build failed'] = this.error || 'yes';
        if (this.sourceBytes != null) info['Size (bytes)'] = this.sourceBytes;
        if (this.options) info['Build options'] = this.options;
        return info;
    }
}
