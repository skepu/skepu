/**
 * dagre-worker.js — runs the dagre graph-layout algorithm off the main thread.
 *
 * The worker replicates the exact graph-building logic from cytoscape-dagre so
 * that positions are identical to what the synchronous layout would produce:
 *   • compound parent relationships are preserved
 *   • edges whose source or target is a compound (parent) node are skipped
 *   • all graph-level options (rankDir, nodeSep, …) are forwarded verbatim
 *
 * Incoming message
 * ────────────────
 * {
 *   type    : 'layout',
 *   nodes   : [ { id, width, height, parent? } … ],
 *   edges   : [ { id, source, target, minlen?, weight? } … ],
 *   options : { rankDir?, nodeSep?, edgeSep?, rankSep?, align?,
 *               acyclicer?, ranker? }
 * }
 *
 * Outgoing message — success
 * ──────────────────────────
 * { type: 'done', positions: { <nodeId>: { x, y } … } }
 *
 * Outgoing message — failure
 * ──────────────────────────
 * { type: 'error', message: <string> }
 */

importScripts('../static/dagre/dagre.min.js');

// ── Compound-node layout tuning ──────────────────────────────────────────────
// This flag is intentionally duplicated from main.js so the worker is
// self-contained.  Keep the two values in sync when changing behaviour.
//
// When true, compound (parent) nodes are given width=0, height=0 so dagre
// computes their bounding box bottom-up from their children rather than
// starting from the pre-inflated Cytoscape bounding box.  That pre-inflation
// is the primary source of the excessive spacing seen with nested region nodes.
// Set to false to restore the old behaviour (pass actual bounding-box dims).
var COMPOUND_ZERO_DIMENSIONS = true;
// ─────────────────────────────────────────────────────────────────────────────

self.onmessage = function (ev) {
  var msg = ev.data;
  if (!msg || msg.type !== 'layout') return;

  try {
    var opts  = msg.options || {};
    var nodes = msg.nodes   || [];
    var edges = msg.edges   || [];

    /* ── Build dagre graphlib graph ─────────────────────────────────────── */
    var g = new dagre.graphlib.Graph({ multigraph: true, compound: true });

    /* Graph-level options — only set non-null values (dagre uses its own
       defaults for anything left undefined).                               */
    var gObj = {};
    if (opts.nodeSep   != null) gObj.nodesep   = opts.nodeSep;
    if (opts.edgeSep   != null) gObj.edgesep   = opts.edgeSep;
    if (opts.rankSep   != null) gObj.ranksep   = opts.rankSep;
    if (opts.rankDir   != null) gObj.rankdir   = opts.rankDir;
    if (opts.align     != null) gObj.align     = opts.align;
    if (opts.acyclicer != null) gObj.acyclicer = opts.acyclicer;
    if (opts.ranker    != null) gObj.ranker    = opts.ranker;
    g.setGraph(gObj);
    g.setDefaultEdgeLabel(function () { return {}; });
    g.setDefaultNodeLabel(function () { return {}; });

    /* Add nodes */
    nodes.forEach(function (n) {
      var w = (COMPOUND_ZERO_DIMENSIONS && n.isParent) ? 0 : n.width;
      var h = (COMPOUND_ZERO_DIMENSIONS && n.isParent) ? 0 : n.height;
      g.setNode(n.id, { width: w, height: h, name: n.id });
    });

    /* Set compound parent relationships (mirrors cytoscape-dagre exactly) */
    nodes.forEach(function (n) {
      if (n.parent != null) g.setParent(n.id, n.parent);
    });

    /* Build a set of parent-node IDs so we can apply the same edge filter
       that cytoscape-dagre uses: skip edges where source or target is a
       compound (parent) node, because dagre can't handle them.            */
    var parentIds = {};
    nodes.forEach(function (n) {
      if (n.isParent) parentIds[n.id] = true;
    });

    /* Add edges */
    edges.forEach(function (e) {
      if (parentIds[e.source] || parentIds[e.target]) return;
      g.setEdge(
        e.source, e.target,
        { minlen: e.minlen || 1, weight: e.weight || 1, name: e.id },
        e.id
      );
    });

    /* ── Run dagre ──────────────────────────────────────────────────────── */
    dagre.layout(g);

    /* ── Collect positions ──────────────────────────────────────────────── */
    var positions = {};
    g.nodes().forEach(function (v) {
      var n = g.node(v);
      if (n) positions[v] = { x: n.x, y: n.y };
    });

    self.postMessage({ type: 'done', positions: positions });

  } catch (err) {
    self.postMessage({ type: 'error', message: String(err) });
  }
};
