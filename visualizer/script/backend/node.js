// node.js — Graph node types.  Browser/Worker compatible ES module.
// Ported from backend/node.py.
'use strict';

import {
    ElwiseEdge, ProxyEdge, ScalarEdge, AntiDepEdge, PrngEdge, VirtualAliasEdge,
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

export class Node {
    constructor(graph, label, totalOrder, region = null) {
        this.id          = crypto.randomUUID();
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

        this.backend = jsonData.backend ?? 'CPU';
        this.file    = jsonData.file ?? null;
        this.line    = jsonData.line ?? null;

        const elwiseInputs = jsonData.elwise_inputs ?? [];
        const proxyInputs  = jsonData.proxy_inputs  ?? [];
        const scalarInputs = jsonData.scalar_inputs  ?? [];
        const outputs      = jsonData.outputs        ?? [];

        for (const id of elwiseInputs) {
            new ElwiseEdge(graph, graph.labelForID(id), id, this);
            graph.readFromID(id, this, this.backend);
        }
        for (const id of proxyInputs) {
            new ProxyEdge(graph, graph.labelForID(id), id, this);
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
                        ScalarNode.fromID(graph, id, this.totalOrder + 1, this.region));
                }
            } else {
                graph.writeToID(id, this, this.hasScalarOutput() ? 'ALL' : this.backend);
            }
        }

        if (jsonData.prng != null) {
            const id = jsonData.prng;
            new PrngEdge(graph, graph.labelForID(id), id, this);
            graph.writeToID(id, this, this.backend);
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
        };
    }

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
        // this.elements and this.pattern already set by ComputationNode
    }

    hasScalarOutput() {
        // Todo: Reduce2D special case
        return this.pattern === 'Reduce' || this.pattern === 'MapReduce';
    }

    toCytoscape() {
        return { ...super.toCytoscape(), pattern: this.pattern, elements: this.elements };
    }

    toTimeline() {
        return { ...super.toTimeline(), pattern: this.pattern, elements: this.elements };
    }

    infoData() {
        return { ...super.infoData(), Pattern: this.pattern, Elements: this.elements };
    }
}

// ── ExternalNode ──────────────────────────────────────────────────────────────

export class ExternalNode extends ComputationNode {
    constructor(graph, jsonData, totalOrder) {
        super(graph, jsonData, totalOrder);
        this.type = 'external';
        this.keypathDiscriminator = this.line;
    }
    // toCytoscape delegates to super intentionally.
}

// ── RegionNode ────────────────────────────────────────────────────────────────

export class RegionNode extends Node {
    // Class-level counter for stable persistent IDs — reset by DirectedGraph.fromEvents.
    static regionCounter = 0;

    constructor(graph, jsonData, totalOrder) {
        super(graph, jsonData.label, totalOrder, null);   // always null region for now

        if (!graph.settings.regions) {
            // Regions disabled: undo node registration and exit.
            graph._removeNode(this);
            return;
        }

        this.treeParents = (this.region) ? [this.region] : [];

        // Apply parent region nesting (super() used null; do it manually now).
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

        RegionNode.regionCounter += 1;
        this.persistentRegionId = 'region_' + RegionNode.regionCounter;
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
        graph.writeToID(jsonData.object_id, this, 'ALL');
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

export class UpdateNode extends DataNode {
    constructor(graph, id, totalOrder, region, backend, elements = null) {
        const label = graph.labelForID(id);
        super(graph, label, totalOrder, region);
        this.type              = 'container_update';
        this.version           = graph.newVersionNumberForID(id);
        this.backend           = backend;
        this.elements          = elements;
        this.isLive            = [];
        this.objectId          = id;
        this.virtualId         = null;
        this.snapshotDataFiles = [];
        graph.writeToID(id, this, backend);
    }

    toCytoscape() {
        const d = { ...super.toCytoscape(), version: this.version };
        if (this.elements != null) d.elements = this.elements;
        return d;
    }

    infoData() {
        const info = super.infoData();
        info['Version'] = this.version;
        info.internal.is_live = this.isLive;
        return info;
    }
}

// ── ScalarNode ────────────────────────────────────────────────────────────────

export class ScalarNode extends DataNode {
    constructor(graph, label, totalOrder, region) {
        super(graph, label, totalOrder, region);
        this.type = 'scalar';
    }

    /** Construct from a raw element_access trace event. */
    static fromEventData(graph, jsonData, totalOrder, region = null) {
        const label = jsonData.label + '[' + jsonData.index + ']';
        if (graph.settings.scalars) {
            const node = new ScalarNode(graph, label, totalOrder, region);
            new ScalarEdge(graph, label, jsonData.source, node);
            graph.readFromID(jsonData.source, node, 'ALL');
            graph.writeToID(jsonData.object_id, node, 'ALL');
        } else {
            graph.setLabelForID(label, jsonData.object_id);
            graph.writeToID(jsonData.object_id,
                            graph.producerForID(jsonData.source), 'ALL');
        }
    }

    /** Construct a synthetic scalar output node for a Reduce/MapReduce. */
    static fromID(graph, objectId, totalOrder, region) {
        const label = 'scalar';
        const node  = new ScalarNode(graph, label, totalOrder, region);
        graph.setLabelForID(label, objectId);
        graph.writeToID(objectId, node, 'ALL');
        return node;
    }

    // toCytoscape delegates to super intentionally.
    infoData() { return super.infoData(); }
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
