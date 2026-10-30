// layout.js — the layout of a graph (a Cytoscape instance), shared by SkeVU and the designer: dagre in a worker, or ELK for a graph with regions
// (ELK lays out the regions with what is in them, so a region is where its nodes are and nothing else is in it).
//
// A layout is started with `startLayout`, which gives the positions of the nodes that are not regions, by id (the regions have the size of what is
// in them). `opts`: { rankDir: 'TB' | 'LR', nodeSep, rankSep, nodeDimensionsIncludeLabels, ranker, acyclicer, algorithm, elkOptions }; the distances
// are scaled for regions here (layout_options.js).

import { globalUtil } from './util.js';
import { COMPOUND_SEP_SCALE, LAYOUT_NODE_SEP, LAYOUT_RANK_SEP } from './layout_options.js';

// One ELK instance per page: the worker is expensive to spawn.
var _sharedElk    = null;
var _sharedElkUrl = null;

function _getElk(workerUrl) {
  if (!_sharedElk || workerUrl !== _sharedElkUrl) {
    _sharedElkUrl = workerUrl;
    _sharedElk    = new ELK(globalUtil.elkOptions(workerUrl));  // ELK loaded as <script>
  }
  return _sharedElk;
}

function _collectElkPositions(nodes, positions, parentAbsX, parentAbsY) {
  if (!nodes) return;
  nodes.forEach(function(n) {
    var absX = parentAbsX + (n.x || 0);
    var absY = parentAbsY + (n.y || 0);
    positions[n.id] = { x: absX + (n.width || 0) / 2, y: absY + (n.height || 0) / 2 };
    if (n.children && n.children.length) _collectElkPositions(n.children, positions, absX, absY);
  });
}

/** The default URLs of the layout workers (relative to the page's script folder). */
export function layoutWorkerUrls() {
  return {
    dagre: window.skevu?.dagreWorker || new URL('./workers/dagre-worker.js', import.meta.url).href,
    elk:   window.skevu?.elkWorker   || new URL('../static/cytoscape.js-elk-2.3.0/dist/elk.bundled.js', import.meta.url).href,
  };
}

/**
 * Starts the layout of the graph: ELK if `elk`, else dagre. Gives `{ promise, stop }`; the promise is the positions, and is rejected if the
 * algorithm failed (a very long chain of dependences can overflow the stack of its recursion, and a worker has little of it).
 */
export function startLayout(cy, opts, urls, elk) {
  return elk ? _elkLayout(cy, opts, urls.elk) : _dagreLayout(cy, opts, urls.dagre);
}

function _dagreLayout(cy, layoutOpts, workerUrl) {
  var worker = null, stopped = false;
  var promise = (async function() {
    var workerNodes = [];
    cy.nodes().forEach(function(n) {
      var dims  = n.layoutDimensions({ nodeDimensionsIncludeLabels: layoutOpts.nodeDimensionsIncludeLabels || false });
      var entry = { id: n.id(), width: dims.w, height: dims.h };
      if (n.isChild())  entry.parent   = n.parent().id();
      if (n.isParent()) entry.isParent = true;
      workerNodes.push(entry);
    });
    var workerEdges = [];
    cy.edges().forEach(function(e) {
      if (!e.source().isParent() && !e.target().isParent() && !e.hasClass('grouped-hidden'))
        workerEdges.push({ id: e.id(), source: e.source().id(), target: e.target().id() });
    });
    var hasCompound = workerNodes.some(function(n) { return n.isParent; });
    var sepScale    = hasCompound ? COMPOUND_SEP_SCALE : 1.0;

    worker = await globalUtil.fetchWorker(workerUrl);
    return new Promise(function(resolve, reject) {
      if (stopped) { worker.terminate(); return; }
      worker.onmessage = function(ev) {
        worker.terminate();
        if (ev.data.type === 'done') resolve(ev.data.positions);
        else reject(new Error('dagre-worker error: ' + ev.data.message));
      };
      worker.onerror = function(err) {
        worker.terminate();
        reject(new Error('dagre-worker uncaught error: ' + (err.message || err)));
      };
      worker.postMessage({
        type: 'layout', nodes: workerNodes, edges: workerEdges,
        options: {
          rankDir:   layoutOpts.rankDir,
          nodeSep:   layoutOpts.nodeSep != null ? layoutOpts.nodeSep * sepScale : undefined,
          rankSep:   layoutOpts.rankSep != null ? layoutOpts.rankSep * sepScale : undefined,
          acyclicer: layoutOpts.acyclicer,
          ranker:    layoutOpts.ranker,
        },
      });
    });
  })();
  return { promise: promise, stop: function() { stopped = true; if (worker) worker.terminate(); } };
}

function _elkLayout(cy, layoutOpts, workerUrl) {
  var dir = layoutOpts.rankDir === 'LR' ? 'RIGHT' : 'DOWN';

  var elkNodeById = {};
  cy.nodes().forEach(function(n) {
    var dims = n.layoutDimensions({ nodeDimensionsIncludeLabels: layoutOpts.nodeDimensionsIncludeLabels || false });
    var en = { id: n.id() };
    if (!n.isParent()) { en.width = dims.w; en.height = dims.h; }
    else en.layoutOptions = { 'elk.direction': dir };
    elkNodeById[n.id()] = en;
  });

  var rootChildren = [];
  cy.nodes().forEach(function(n) {
    var en = elkNodeById[n.id()];
    if (n.isChild()) { var p = elkNodeById[n.parent().id()]; (p.children = p.children || []).push(en); }
    else rootChildren.push(en);
  });

  var elkEdges = [];
  cy.edges().forEach(function(e) {
    if (!e.hasClass('grouped-hidden'))
      elkEdges.push({ id: e.id(), sources: [e.source().id()], targets: [e.target().id()] });
  });

  var elkOptions = Object.assign({
    'algorithm':                                 layoutOpts.algorithm || 'layered',
    'elk.direction':                             dir,
    'elk.hierarchyHandling':                     'INCLUDE_CHILDREN',
    'elk.spacing.nodeNode':                      layoutOpts.nodeSep != null ? layoutOpts.nodeSep : LAYOUT_NODE_SEP,
    'elk.layered.spacing.nodeNodeBetweenLayers': layoutOpts.rankSep != null ? layoutOpts.rankSep : LAYOUT_RANK_SEP,
    'elk.layered.cycleBreaking.strategy':        'GREEDY',
    'elk.layered.nodePlacement.strategy':        'NETWORK_SIMPLEX',
    'elk.edgeRouting':                           'SPLINES',
    // Nodes and components that the dependences do not order (the independent trials of a loop, a forest) are placed in the order of the trace
    // (the order of the nodes that are given), not in the order that the algorithm happens to get to them
    'elk.layered.considerModelOrder.strategy':   'NODES_AND_EDGES',
    'elk.layered.considerModelOrder.components': 'MODEL_ORDER',
  }, layoutOpts.elkOptions || {});

  var promise = _getElk(workerUrl).layout({ id: 'root', layoutOptions: elkOptions, children: rootChildren, edges: elkEdges })
    .then(function(result) {
      var positions = {};
      _collectElkPositions(result.children, positions, 0, 0);
      return positions;
    })
    .catch(function(err) {
      _sharedElk = null;  // invalidate on error; next layout gets a fresh instance
      throw new Error('ELK layout error: ' + err);
    });
  return { promise: promise, stop: function() {} };
}
