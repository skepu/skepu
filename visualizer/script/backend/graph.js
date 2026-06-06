// graph.js — Directed graph: construction, analysis, serialization.
// Browser/Worker compatible ES module.  Ported from backend/graph.py.
'use strict';

import {
    Node, ComputationNode, CallNode, ExternalNode, RegionNode, FusedNode,
    AllocationNode, DeallocationNode, UpdateNode, ScalarNode, TransferNode,
} from './node.js';
import { ElwiseEdge, VirtualAliasEdge } from './edge.js';

// ── Backend → memory-space map ────────────────────────────────────────────────

const BACKEND_TO_MEMSPACE = {
    'CPU':    'CPU',
    'OpenMP': 'CPU',
    'OpenCL': 'GPU (OpenCL)',
    'CUDA':   'GPU (CUDA)',
};

const ALL_MEMSPACES = [...new Set(Object.values(BACKEND_TO_MEMSPACE))];

function makeMemSpaceDict() {
    const d = { ANY: null };
    for (const ms of ALL_MEMSPACES) d[ms] = null;
    return d;
}

// ── GraphSettings ─────────────────────────────────────────────────────────────

export class GraphSettings {
    constructor() {
        this.updates       = true;
        this.transfers     = true;
        this.antideps      = true;
        this.allocations   = true;
        this.deallocations = true;
        this.regions       = true;
        this.scalars       = true;
        this.synthesizeRootRegion = true;
    }
}

// ── DirectedGraph ─────────────────────────────────────────────────────────────

export class DirectedGraph {

    constructor() {
        this._nodes = [];
        this._dependenceEdges = [];
        this._treeEdges = [];

        this._labelByID      = {};
        this._nodeLabelIndex = {};
        this._nodeIdIndex    = {};
        this._edgeIdIndex    = {};

        // Keyed by object_id; auto-vivified to makeMemSpaceDict() on first access.
        this._latestProducerOfID = {};
        this._latestConsumerOfID = {};

        // Version counter per object_id (defaultdict(int) equivalent).
        this.update_versions = {};

        // Virtual container registry.
        this._virtual_containers    = {};
        this._virtual_id_for_object = {};

        // Decorative edges never affect layout or DAG analysis.
        this._decorative_edges = [];

        // comesBefore memo: Map<Node, Map<Node, boolean>> using object identity.
        this.comesBeforeCache = new Map();

        // Set by computeEquivalenceClasses():
        //   regions mode  → Map<RegionNode, Array<Array<Node>>>
        //   flat mode     → Array<Array<Node>>
        this.equivalenceClasses = null;

        this.settings    = new GraphSettings();
        this._trace_index = 0;

        this._rootRegionNode = null;
    }

    // ── Auto-vivification helpers ─────────────────────────────────────────────

    _producerEntry(id) {
        if (!this._latestProducerOfID[id]) this._latestProducerOfID[id] = makeMemSpaceDict();
        return this._latestProducerOfID[id];
    }

    _consumerEntry(id) {
        if (!this._latestConsumerOfID[id]) this._latestConsumerOfID[id] = makeMemSpaceDict();
        return this._latestConsumerOfID[id];
    }

    // ── Node / edge registry ──────────────────────────────────────────────────

    addNode(node) {
        this._nodes.push(node);
        this._nodeIdIndex[node.id]       = node;
        this._nodeLabelIndex[node.label] = node;
    }

    getLastNode() {
      return this._nodes.at(-1); 
    }

    /** Undo addNode — called by conditional constructors (AllocationNode etc.)
     *  when a node should be silently skipped rather than registered. */
    _removeNode(node) {
        const idx = this._nodes.indexOf(node);
        if (idx !== -1) this._nodes.splice(idx, 1);
        delete this._nodeIdIndex[node.id];
        if (this._nodeLabelIndex[node.label] === node)
            delete this._nodeLabelIndex[node.label];
        // Also detach from parent region if it was nested during super().
        if (node.region) {
            const i = node.region.nestedNodes ? node.region.nestedNodes.indexOf(node) : -1;
            if (i !== -1) node.region.nestedNodes.splice(i, 1);
        }
    }

    addDependenceEdge(edge) {
        const src = this._nodeIdIndex[edge.source.id];
        const tgt = this._nodeIdIndex[edge.target.id];
        src.addOutgoingEdge(edge);
        tgt.addIncomingEdge(edge);
        this._dependenceEdges.push(edge);
        this._edgeIdIndex[edge.id] = edge;
    }

    addTreeEdge(edge) {
        this._treeEdges.push(edge);
    }

    addDecorativeEdge(edge) {
        this._decorative_edges.push(edge);
    }

    getNodeById(id)          { return this._nodeIdIndex[id]; }
    getEdgeById(id)          { return this._edgeIdIndex[id]; }
    getNodeByLabel(label)    { return this._nodeLabelIndex[label]; }

    setLabelForID(label, id) { this._labelByID[id] = label; }
    labelForID(id)           { return this._labelByID[id] ?? ''; }
    depthByLabel(label)      { return this._nodeLabelIndex[label]?.depth ?? 0; }

    newVersionNumberForID(id) {
        if (!this.update_versions[id]) this.update_versions[id] = 0;
        return ++this.update_versions[id];
    }

    setCurrentTraceIndex(index) { this._trace_index = index; }
    currentTraceIndex()         { return this._trace_index; }
    
    getAllNodes()          { return this._nodes; }
    getRootNodes()         { return this._nodes.filter(n => n.getIncomingEdges().length === 0); }
    getLeafNodes()         { return this._nodes.filter(n => n.getOutgoingEdges().length === 0); }
    getGlobalRegionNodes() { return this._nodes.filter(n => !n.region); }
    getGlobalFreeNodes()   { return this._nodes.filter(n => !n.region && !(n instanceof RegionNode)); }
    getAllRegions()        { return this._nodes.filter(n => n instanceof RegionNode); }
    setRootRegion(node)    { this._rootRegionNode = node; }
    getRootRegion()        { return this._rootRegionNode; }

    // ── Memory-space tracking ─────────────────────────────────────────────────

    writeToID(id, node, backend) {
        this.readFromID(id, node, backend);
        const e = this._producerEntry(id);
        e['ANY'] = node;
        if (backend === 'ALL') {
            for (const ms of ALL_MEMSPACES) e[ms] = node;
        } else {
            const ms = BACKEND_TO_MEMSPACE[backend];
            if (ms !== undefined) e[ms] = node;
        }
    }

    readFromID(id, node, backend) {
        const e = this._consumerEntry(id);
        e['ANY'] = node;
        if (backend === 'ALL') {
            for (const ms of ALL_MEMSPACES) e[ms] = node;
        } else {
            const ms = BACKEND_TO_MEMSPACE[backend];
            if (ms !== undefined) e[ms] = node;
        }
    }

    transferIDMemspace(objectId, sourceBackend, targetBackend) {
        const e   = this._producerEntry(objectId);
        const src = BACKEND_TO_MEMSPACE[sourceBackend];
        const tgt = BACKEND_TO_MEMSPACE[targetBackend];
        e[tgt] = e[src] ?? null;
    }

    producerForID(id, backend = 'ANY') {
        const e = this._latestProducerOfID[id];
        if (!e) return null;
        const ms = (backend !== 'ANY') ? (BACKEND_TO_MEMSPACE[backend] ?? 'ANY') : 'ANY';
        return e[ms] ?? null;
    }

    accessorForID(id, backend = 'ANY') {
        const e = this._latestConsumerOfID[id];
        if (!e) return null;
        const ms = (backend !== 'ANY') ? (BACKEND_TO_MEMSPACE[backend] ?? 'ANY') : 'ANY';
        return e[ms] ?? null;
    }

    // ── Snapshot & virtual containers ─────────────────────────────────────────

    attachSnapshot(objectId, dataFile, totalOrder = null) {
        const e    = this._latestProducerOfID[objectId];
        const node = e ? e['ANY'] : null;
        if (node instanceof UpdateNode) {
            node.hasSnapshot = true;
            node.snapshotDataFiles.push([totalOrder, dataFile]);
        }
    }

    registerVirtualContainer(virtualId, label, objectIds) {
        this._virtual_containers[virtualId] = { label, object_ids: objectIds };
        for (const oid of objectIds) this._virtual_id_for_object[oid] = virtualId;
    }

    createVirtualAliasEdges(createEdges = true) {
        for (const [virtualId, info] of Object.entries(this._virtual_containers)) {
            const { label, object_ids } = info;
            const objIdSet = new Set(object_ids);
            const members = this._nodes
                .filter(n => n instanceof UpdateNode && objIdSet.has(n.objectId))
                .sort((a, b) => a.totalOrder - b.totalOrder);
            if (members.length < 2) continue;
            for (const n of members) n.virtualId = virtualId;
            if (createEdges) {
                for (let i = 0; i < members.length - 1; i++)
                    new VirtualAliasEdge(this, members[i], members[i + 1], label);
            }
        }
    }

    pruneEmptyRegions() {
        let changed = true;
        while (changed) {
            changed = false;
            const toRemove = new Set(
                this._nodes.filter(n => n instanceof RegionNode && n.nestedNodes.length === 0)
            );
            if (toRemove.size === 0) break;
            changed = true;
            this._nodes = this._nodes.filter(n => !toRemove.has(n));
            for (const node of toRemove) {
                delete this._nodeIdIndex[node.id];
                if (this._nodeLabelIndex[node.label] === node)
                    delete this._nodeLabelIndex[node.label];
                if (node.region?.nestedNodes) {
                    const i = node.region.nestedNodes.indexOf(node);
                    if (i !== -1) node.region.nestedNodes.splice(i, 1);
                }
            }
        }
    }

    // ── Graph analysis ────────────────────────────────────────────────────────

    nodePrecedes(n1, n2) {
        for (const e of n2.getIncomingEdges())
            if (e.source === n1) return true;
        return false;
    }

    comesBefore(n1, n2) {
        const inner = this.comesBeforeCache.get(n1);
        if (inner?.has(n2)) return inner.get(n2);

        let result;
        if (n1 === n2) {
            result = false;
        } else if (this.nodePrecedes(n1, n2)) {
            result = true;
        } else if (this.nodePrecedes(n2, n1)) {
            result = false;
        } else {
            result = false;
            for (const edge of n2.getIncomingEdges()) {
                if (this.comesBefore(n1, edge.source)) { result = true; break; }
            }
        }

        if (!this.comesBeforeCache.has(n1)) this.comesBeforeCache.set(n1, new Map());
        this.comesBeforeCache.get(n1).set(n2, result);
        return result;
    }

    computeEquivalenceClasses() {
        const eqClassesHelper = (nodes) => {
            const classes = [];
            for (const node of nodes) {
                let placed = false;
                for (const cls of classes) {
                    let compatible = true;
                    for (const other of cls) {
                        if (this.comesBefore(node, other) || this.comesBefore(other, node)) {
                            compatible = false; break;
                        }
                    }
                    if (compatible) { cls.push(node); placed = true; break; }
                }
                if (!placed) classes.push([node]);
            }
            return classes;
        };

        if (!this.settings.regions) {
            this.equivalenceClasses = eqClassesHelper(this.getGlobalFreeNodes());
        } else {
            this.equivalenceClasses = new Map();
            for (const region of this.getAllRegions())
                this.equivalenceClasses.set(region, eqClassesHelper(region.getDirectChildren()));
        }
    }

    computeKeyPaths() {
        const helper = (node, keypath) => {
            node.keypath = keypath + '->' + node.type + '_' + node.label + '_' +  node.keypathDiscriminator;
            if (node instanceof RegionNode) {
                for (const nested of node.nestedNodes)
                    helper(nested, node.keypath);
            }
        };
        for (const node of this.getGlobalRegionNodes()) helper(node, '[global]');
    }

    computeDepths() {
        for (const node of this._nodes) node.depth = 0;
        const inDeg = new Map(this._nodes.map(n => [n, n.getIncomingEdges().length]));
        const queue = this._nodes.filter(n => inDeg.get(n) === 0);
        for (let i = 0; i < queue.length; i++) {
            const node = queue[i];
            for (const child of node.getChildNodes()) {
                child.depth = Math.max(child.depth, node.depth + 1);
                inDeg.set(child, inDeg.get(child) - 1);
                if (inDeg.get(child) === 0) queue.push(child);
            }
        }
    }

    findCriticalPath() {
        let deepest = null;
        for (const node of this._nodes) {
            if (node.type === 'region') continue;
            if (!deepest || node.depth > deepest.depth) deepest = node;
        }
        let cur = deepest;
        while (cur) {
            cur.isCriticalPath = true;
            let maxDepth = -1, maxEdge = null, next = null;
            for (const edge of cur.getIncomingEdges()) {
                const p = edge.source;
                if (p.depth > maxDepth) { maxDepth = p.depth; maxEdge = edge; next = p; }
            }
            if (maxEdge) { maxEdge.isCriticalPath = true; cur = next; }
            else break;
        }
    }

    findFusions() {
        this.findParallelFusions();
        return this.findSerialFusions();
    }

    findParallelFusions() {
        const findInClass = (eqClass, region) => {
            let fused = null, root = null;
            for (const node of eqClass) {
                if (!(node instanceof CallNode) || node.pattern !== 'Map') continue;
                if (!root) { root = node; continue; }
                if (!fused) {
                    fused = new FusedNode(this, 'Parallel Fusion', node.totalOrder, region);
                    fused.fusionPattern = 'Map';
                    if (root.region?.nestedNodes) {
                        const i = root.region.nestedNodes.indexOf(root);
                        if (i !== -1) root.region.nestedNodes.splice(i, 1);
                    }
                    root.fused = fused; root.region = fused;
                    fused.elementSizes.push(root.elements);
                }
                if (node.region?.nestedNodes) {
                    const i = node.region.nestedNodes.indexOf(node);
                    if (i !== -1) node.region.nestedNodes.splice(i, 1);
                }
                node.fused = fused; node.region = fused;
                fused.elementSizes.push(node.elements);
            }
        };

        if (this.settings.regions && this.equivalenceClasses instanceof Map) {
            for (const [region, classes] of this.equivalenceClasses)
                for (const cls of classes) findInClass(cls, region);
        } else if (Array.isArray(this.equivalenceClasses)) {
            for (const cls of this.equivalenceClasses) findInClass(cls, null);
        }
        return [];
    }

    findSerialFusions() {
        const seen    = new Set();
        const fusions = [];

        const helper = (node, fused) => {
            let candidate;

            if (!(node instanceof CallNode)) {
                candidate = (node instanceof UpdateNode) ? node.fused !== null : false;
            } else {
                candidate = !(fused !== null && node.pattern !== 'Map');
            }

            if (candidate) {
                for (const edge of node.getIncomingEdges()) {
                    if (!(edge instanceof ElwiseEdge)) continue;
                    const parent = edge.source;
                    if (!(parent instanceof CallNode) && !(parent instanceof UpdateNode)) continue;

                    // Fan-out check
                    if (parent instanceof UpdateNode) {
                        if (parent.getOutgoingEdges().length > 1) continue;
                    } else {
                        const sameLabel = parent.getOutgoingEdges().filter(
                            e => e instanceof ElwiseEdge && e.label === edge.label
                        ).length;
                        if (sameLabel > 1) continue;
                    }

                    if (!(parent instanceof UpdateNode)) continue;

                    const incoming = parent.getIncomingEdges();
                    if (!incoming.length) continue;
                    const upstream = incoming[0].source;

                    const pairKey = node.id + '|' + upstream.id;
                    if (!seen.has(pairKey)) {
                        seen.add(pairKey);
                        fusions.push([node.id, upstream.id]);
                    }

                    fused = node.fused || fused || parent.fused;
                    if (!fused) {
                        fused = new FusedNode(this, 'Serial Fusion', node.totalOrder, node.region);
                        fused.fusionPattern = node.pattern;
                    }

                    node.fused   = fused;
                    parent.fused = fused;

                    if (node.region instanceof RegionNode) {
                        const i = node.region.nestedNodes.indexOf(node);
                        if (i !== -1) node.region.nestedNodes.splice(i, 1);
                    }
                    node.region = fused;

                    if (parent.region instanceof RegionNode) {
                        const i = parent.region.nestedNodes.indexOf(parent);
                        if (i !== -1) parent.region.nestedNodes.splice(i, 1);
                    }
                    parent.region = fused;

                    helper(upstream, fused);
                }
            } else {
                for (const edge of node.getIncomingEdges()) {
                    if (!(edge instanceof ElwiseEdge)) continue;
                    if (!edge.source.fused) helper(edge.source, null);
                }
            }
        };

        for (const node of this.getLeafNodes()) helper(node, null);

        // Second pass: reduction sinks unreachable from leaves via ElwiseEdges.
        for (const node of this.getAllNodes()) {
            if (node instanceof CallNode && !node.fused) {
                if (!node.getOutgoingEdges().some(e => e instanceof ElwiseEdge))
                    helper(node, null);
            }
        }

        return fusions;
    }

    coalesceIterations() {
        this.computeKeyPaths();

        // Group nodes by keypath.
        const groups = new Map();
        for (const node of this._nodes) {
            if (!groups.has(node.keypath)) groups.set(node.keypath, []);
            groups.get(node.keypath).push(node);
        }

        const canonical = new Map();
        for (const [kp, grp] of groups) canonical.set(kp, grp[0]);

        const canonicalByID = new Map();

        const restIds = new Set();
        for (const [kp, nodes] of groups) {
            const can = canonical.get(kp);
            for (let i = 1; i < nodes.length; i++) {
                const node = nodes[i];
                restIds.add(node.id);
                canonicalByID.set(node.id, can);
                can.iterationCount += 1;
                can.totalOrders.push(node.totalOrder);
                can.traceIndices.push(node.traceIndex);
                if (node.treeParents.length > 0)
                  can.treeParents.push(canonical.get(node.treeParents[0].keypath));
                if (node.durations) can.durations.push(...node.durations);
                if (node.intervals) can.intervals.push(...node.intervals);
                if (node.snapshotDataFiles) can.snapshotDataFiles.push(...node.snapshotDataFiles);
                if (node instanceof RegionNode) {
                    can.regionIntervals.push(...node.regionIntervals);
                    if (node.regionStart != null) {
                        if (can.regionStart == null) {
                            can.regionStart = node.regionStart;
                            can.regionEnd   = node.regionEnd;
                        } else {
                            can.regionStart = Math.min(can.regionStart, node.regionStart);
                            can.regionEnd   = Math.max(can.regionEnd,   node.regionEnd);
                        }
                    }
                }
                for (const edge of node.getIncomingEdges())
                    edge.target = canonical.get(edge.target.keypath);
                for (const edge of node.getOutgoingEdges())
                    edge.source = canonical.get(edge.source.keypath);
            }
        }

        this._nodes = this._nodes.filter(n => !restIds.has(n.id));

        // Fix stale region/fused pointers.
        const survivingIds = new Set(this._nodes.map(n => n.id));
        for (const node of this._nodes) {
            if (node.region && !survivingIds.has(node.region.id))
                node.region = canonical.get(node.region.keypath) ?? node.region;
            if (node.fused && !survivingIds.has(node.fused.id))
                node.fused = canonical.get(node.fused.keypath) ?? node.fused;
        }

        // Deduplicate edges with identical (source, target).
        const seen = new Map();
        const deduped = [];
        for (const edge of this._dependenceEdges) {
            const key = edge.source.id + '|' + edge.target.id;
            if (!seen.has(key)) { seen.set(key, edge); deduped.push(edge); }
            else seen.get(key).iterationCount += 1;
        }
        this._dependenceEdges = deduped;
    }

    toSubscript(val) {
      var strval = '' + val;
      strval = strval.replace("0", "₀");
      strval = strval.replace("1", "₁");
      strval = strval.replace("2", "₂");
      strval = strval.replace("3", "₃");
      strval = strval.replace("4", "₄");
      strval = strval.replace("5", "₅");
      strval = strval.replace("6", "₆");
      strval = strval.replace("7", "₇");
      strval = strval.replace("8", "₈");
      strval = strval.replace("9", "₉");
      return strval;
    }

    numberRegionLabels() {
        const counts = {};
        for (const node of this._nodes) {
            if (!(node instanceof RegionNode)) continue;
            if (!counts[node.keypath]) counts[node.keypath] = 1;
            node.label = node.label + '' + this.toSubscript(counts[node.keypath]++);
        }
    }

    // ── Factory ───────────────────────────────────────────────────────────────

    static fromEvents(events, settings = null) {
        const g = new DirectedGraph();
        if (settings !== null) g.settings = settings;

        RegionNode.regionCounter = 0;
        if (g.settings.synthesizeRootRegion)
          g.setRootRegion(new RegionNode(g, { label: "[global]" }, 0));

        for (let i = 0; i < events.length; i++) {
            const ev = events[i];
            g.setCurrentTraceIndex(i);
            const order = (i + 1) * 100;
            const t = ev.type;
            if      (t === 'skeleton_call')     new CallNode(g, ev, order);
            else if (t === 'external')          new ExternalNode(g, ev, order);
            else if (t === 'allocation')        new AllocationNode(g, ev, order);
            else if (t === 'deallocation')      new DeallocationNode(g, ev, order);
            else if (t === 'element_access')    ScalarNode.fromEventData(g, ev, order);
            else if (t === 'transfer')          new TransferNode(g, ev, order);
            else if (t === 'region')            new RegionNode(g, ev, order);
            else if (t === 'snapshot')          g.attachSnapshot(ev.object_id, ev.data_file, order);
            else if (t === 'virtual_container') g.registerVirtualContainer(ev.virtual_id, ev.label, ev.object_ids);
        }

        return g;
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    toTimelineAndBadges(expandedPids = null, expandAll = false) {
        const pidSet = new Set(expandedPids ?? []);
        const allPids = this._nodes.filter(n => n instanceof RegionNode)
                                   .map(n => n.persistentRegionId);
        const collapsedPids = expandAll
            ? new Set()
            : new Set(allPids.filter(p => !pidSet.has(p)));

        const ancestorCollapsed = (node) => {
            let r = node.region;
            while (r) {
                if (r instanceof RegionNode && collapsedPids.has(r.persistentRegionId)) return true;
                r = r.region;
            }
            return false;
        };

        const visibleIds = new Set(this._nodes.filter(n => !ancestorCollapsed(n)).map(n => n.id));

        const tl = [], sb = [];
        for (const node of this._nodes) {
            const hidden = !visibleIds.has(node.id);
            const t = node.toTimeline();
            if (t !== null) { if (hidden) t.hidden = true; tl.push({ data: t }); }
            const b = node.toSourceBadge();
            if (b !== null) { if (hidden) b.hidden = true; sb.push({ data: b }); }
        }
        return [tl, sb];
    }

    toCytoscape(expandedPids = null, expandAll = false) {
        const pidSet = new Set(expandedPids ?? []);
        const allPids = this._nodes.filter(n => n instanceof RegionNode)
                                   .map(n => n.persistentRegionId);
        const collapsedPids = expandAll
            ? new Set()
            : new Set(allPids.filter(p => !pidSet.has(p)));

        const ancestorCollapsed = (node) => {
            let r = node.region;
            while (r) {
                if (r instanceof RegionNode && collapsedPids.has(r.persistentRegionId)) return true;
                r = r.region;
            }
            return false;
        };

        const visibleIds = new Set(this._nodes.filter(n => !ancestorCollapsed(n)).map(n => n.id));

        const effectiveEndpoint = (node) => {
            if (visibleIds.has(node.id)) return [node.id, false];
            let r = node.region;
            while (r) {
                if (visibleIds.has(r.id)) return [r.id, true];
                r = r.region;
            }
            return [null, true];
        };

        const emitEdges = (edgeList, cyEdges, seenPromoted) => {
            for (const edge of edgeList) {
                const [srcId, srcP] = effectiveEndpoint(edge.source);
                const [tgtId, tgtP] = effectiveEndpoint(edge.target);
                if (!srcId || !tgtId) continue;
                if (srcId === tgtId && (srcP || tgtP)) continue;
                if (!srcP && !tgtP) {
                    cyEdges.push({ data: edge.toCytoscape() });
                } else {
                    const d = edge.toCytoscape();
                    d.source = srcId; d.target = tgtId;
                    const key = `${srcId}|${tgtId}|${d.type ?? ''}|${d.access_mode ?? ''}`;
                    if (seenPromoted.has(key)) continue;
                    seenPromoted.add(key);
                    d.id = crypto.randomUUID();
                    cyEdges.push({ data: d });
                }
            }
        };

        const cyNodes = [], cyEdges = [];
        for (const node of this._nodes) {
            if (!visibleIds.has(node.id)) continue;
            const d = node.toCytoscape();
            if (node instanceof RegionNode && collapsedPids.has(node.persistentRegionId))
                d.collapsed = true;
            cyNodes.push({ data: d });
        }

        const seenP = new Set();

        if (this.settings.mode == 'dependence-dag')
        {
          emitEdges(this._dependenceEdges,  cyEdges, seenP);
          emitEdges(this._decorative_edges, cyEdges, seenP);
        }
        else if (this.settings.mode == 'program-tree')
        {
        //  emitEdges(this._treeEdges,        cyEdges, seenP);
  
          for (const item of cyNodes) {
            item.data.parent = undefined;
            if (item.data.type == "region")
              item.data.type = "tree-region";
          }
  
          // total order tree mode 
          if (true)
          {
            for (const node of this._nodes) {
              for (var parent of node.treeParents)
              {
                var d = {};
                d.target = node.id;
                d.id = crypto.randomUUID();
                d.source = parent.id;
                cyEdges.push({ data: d });
              }
            }
          }
        }

        return [cyNodes, cyEdges, allPids];
    }

    summary() {
        console.log(`Nodes: ${this._nodes.length}, Edges: ${this._dependenceEdges.length}`);
    }
}
