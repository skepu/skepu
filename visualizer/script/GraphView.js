// GraphView.js — Cytoscape-based SkePU DAG renderer (provider-mode only).
//
// Renders a SkePU execution-trace graph inside an arbitrary container element.
// All state is instance-local; multiple GraphView instances can coexist on the
// same page independently.  A TraceDataProvider is always required — it owns
// the fetch pipeline, colour spectrum, region collapse state, and the data
// caches shared with TimelineView and SourceCodeView.
//
// ── Quick start ───────────────────────────────────────────────────────────────
//
//   import { GraphView }       from './GraphView.js';
//   import { TraceDataProvider } from './TraceDataProvider.js';
//
//   const provider = new TraceDataProvider({ fetchGraph: myFetch, settings: {} });
//   const view = new GraphView(document.getElementById('my-graph'), {
//       provider,
//       onRenderEnd: stats => console.log(stats.nodeCount, 'nodes'),
//   });
//
//   view.resetFirstLoad();   // call once after loading a new trace
//   view.render();           // → provider.fetch() → 'data' event → render
//
// ── Constructor options ───────────────────────────────────────────────────────
//
//   provider              Required. TraceDataProvider instance.
//   dagreWorkerUrl        URL for dagre-worker.js.
//   elkWorkerUrl          URL for elk.bundled.js (hierarchical traces).
//   settings              Initial settings object (merged with defaults).
//   darkMode              Boolean; auto-detected from <html class="dark-mode"> if omitted.
//   onEdgeClick(data)     Called when a graph edge is clicked.
//   onRenderStart()       Called at the start of each render.
//   onRenderEnd(stats)    Called after layout completes.
//                         stats: { nodeCount, edgeCount, eventCount, snapshotCount,
//                                  modelingMs, renderingMs }
//   onError(err)          Called on non-abort render failures.

'use strict';

// ── Shared page resources ─────────────────────────────────────────────────────
// One tooltip and one ELK instance per page is correct: only one element can be
// hovered at a time, and the ELK worker is expensive to spawn.

var _sharedTooltip     = null;
var _sharedTooltipTimer = null;

function _ensureTooltip() {
  if (!_sharedTooltip) {
    _sharedTooltip = document.createElement('div');
    _sharedTooltip.className = 'gv-tooltip';
    _sharedTooltip.style.cssText =
      'position:fixed;display:none;background:rgba(30,30,30,0.92);color:#eee;' +
      'font:12px/1.4 monospace;padding:6px 9px;border-radius:4px;' +
      'white-space:pre;pointer-events:none;z-index:9999;max-width:320px;' +
      'box-shadow:0 2px 8px rgba(0,0,0,0.4);';
    document.body.appendChild(_sharedTooltip);
  }
  return _sharedTooltip;
}

function _showTooltip(content, x, y) {
  var tip = _ensureTooltip();
  clearTimeout(_sharedTooltipTimer);
  _sharedTooltipTimer = setTimeout(function() {
    tip.textContent = content;
    tip.style.display = 'block';
    _positionTooltip(x, y);
  }, 500);
}

function _hideTooltip() {
  clearTimeout(_sharedTooltipTimer);
  if (_sharedTooltip) _sharedTooltip.style.display = 'none';
}

function _positionTooltip(x, y) {
  var tip = _sharedTooltip;
  if (!tip) return;
  var pad = 14;
  tip.style.left = '-9999px';
  tip.style.top  = '-9999px';
  var tw = tip.offsetWidth, th = tip.offsetHeight;
  var left = x + pad, top = y + pad;
  if (left + tw > window.innerWidth  - 8) left = x - tw - pad;
  if (top  + th > window.innerHeight - 8) top  = y - th - pad;
  tip.style.left = left + 'px';
  tip.style.top  = top  + 'px';
}

var _sharedElk       = null;
var _sharedElkUrl    = null;

function _getElk(workerUrl) {
  if (!_sharedElk || workerUrl !== _sharedElkUrl) {
    _sharedElkUrl = workerUrl;
    _sharedElk    = new ELK({ workerUrl: workerUrl });  // ELK loaded as <script>
  }
  return _sharedElk;
}



function toSubscript(val) {
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

// ── Constants ─────────────────────────────────────────────────────────────────

const NODE_SIZE_SCALE    = 100;
const NODE_SIZE_MIN      = 15;
const COMPOUND_SEP_SCALE = 0.15;
const LAYOUT_NODE_SEP    = 15;
const LAYOUT_RANK_SEP    = 20;
const LAYOUT_FIT_PAD     = 30;
const ANIM_ZOOM_MS       = 400;
const ZOOM_FIT_PAD       = 40;
const ZOOM_NODE_DIVISOR  = 2.3;
const HIGHLIGHT_LIGHT    = '#0055cc';
const HIGHLIGHT_DARK     = '#e8720c';

const DEFAULT_SETTINGS = {
  'container-updates':       false,
  'container-allocations':   false,
  'container-deallocations': false,
  'container-transfers':     false,
  'element-accesses':        false,
  'anti-deps':               false,
  'virtual-alias-edges':     false,
  'edge-labels':             false,
  'node-labels':             true,
  'show_regions':            false,
  'collapse_iteration':      false,
  'fusion_analysis':         false,
  'graph-mode':      'dependence-dag',
  'node-color-call': 'duration-total',
  'node-size-call':  'fixed',
  'direction':       'vertical',
  'edge-width':      'fixed',
  'edge-opacity':    'fixed',
};

// ── GraphView class ───────────────────────────────────────────────────────────

export class GraphView {

  // ── Constructor ─────────────────────────────────────────────────────────────

  constructor(container, options) {
    options = options || {};

    if (!options.provider)
      throw new Error('GraphView: options.provider (TraceDataProvider) is required');

    this._provider = options.provider;

    // Worker URLs for layout engines.
    this._dagreWorkerUrl = options.dagreWorkerUrl
      || new URL('./workers/dagre-worker.js', import.meta.url).href;
    this._elkWorkerUrl   = options.elkWorkerUrl
      || new URL('./static/cytoscape.js-elk-2.3.0/dist/elk.bundled.js', import.meta.url).href;

    // Callbacks.
    this._onEdgeClick     = options.onEdgeClick     || null;
    this._onRenderStart   = options.onRenderStart   || null;
    this._onRenderEnd     = options.onRenderEnd     || null;
    this._onError         = options.onError         || null;
    this._onNodeHover     = options.onNodeHover     || null;
    this._onNodeHoverEnd  = options.onNodeHoverEnd  || null;
    this._onEdgeHover     = options.onEdgeHover     || null;
    this._onEdgeHoverEnd  = options.onEdgeHoverEnd  || null;

    // Settings: always start from defaults so callers that pass a partial object
    // (or {}) still get all keys.  graph.js overwrites this with State.settings
    // after construction to establish the shared reference.
    this.settings = Object.assign({}, DEFAULT_SETTINGS, options.settings || {});

    // Appearance.
    this._darkMode = (options.darkMode !== undefined)
      ? options.darkMode
      : document.documentElement.classList.contains('dark-mode');

    // Companion LegendView — set by the caller after both objects are created
    // so mouseover events can highlight the matching legend entry.
    this._legendView = null;

    // Per-instance render state.
    this._cy            = null;
    this._renderSeq     = 0;
    this._currentLayout = null;

    this._opaqueNodes    = [];  // node IDs at full opacity; others dimmed
    this._lastGraphNodes = [];  // local copy for _resolveVisible parent-map lookup

    // Edge-grouping bookkeeping (reset before each layout pass).
    this._groupedEdgeReps   = new Set();
    this._groupedEdgeHidden = new Set();
    this._bidirEdgeReps     = new Set();

    // Build inner DOM.  The container just needs to be sized; we manage the rest.
    var s = container.style;
    if (!s.position || s.position === 'static') s.position = 'relative';

    this._cyEl = document.createElement('div');
    this._cyEl.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;';

    this._overlayEl = document.createElement('div');
    this._overlayEl.style.cssText =
      'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;' +
      'z-index:10;background:inherit;opacity:0;';

    container.appendChild(this._cyEl);
    container.appendChild(this._overlayEl);

    // ── Wire provider events ─────────────────────────────────────────────────
    // Use a closure variable to carry the render-sequence value from 'fetchstart'
    // through to 'data', so _renderFromData knows which render it belongs to.
    var self = this;
    var _pendingSeq = 0;

    this._provider.on('fetchstart', function() {
      self._cancelCurrentRender();
      _pendingSeq = self._renderSeq;
      if (self._onRenderStart) self._onRenderStart();
      self._showOverlay();
    });

    this._provider.on('data', function(rawData, minmax) {
      self._renderFromData(rawData, minmax, _pendingSeq);
    });

    this._provider.on('error', function(err) {
      self._hideOverlay();
      if (self._onError) self._onError(err);
      else console.error('[GraphView] render error:', err);
    });

    // Incoming selection from another view — update graph highlight and zoom.
    // Skip when source === 'graph' to avoid echo-back for our own node clicks.
    this._provider.on('select', function(ev) {
      if (ev.source === 'graph') return;
      var ids = ev.allIds && ev.allIds.length > 1 ? ev.allIds : null;
      if (ids) {
        self.selectNodes(ids, { focusId: ev.focusId || ev.nodeId });
      } else {
        self.selectNode(ev.focusId || ev.nodeId, { zoom: true });
      }
    });

    // Clear event — emitted by this GraphView's own background click.  Handled
    // here so the visual state is reset via the same code path as any other
    // initiator that might emit 'clear' in future.
    this._provider.on('clear', function() {
      self.clearSelection();
    });
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  /**
   * Reset the first-load flag so the next render treats this trace as freshly
   * loaded — enabling auto-expand on small traces, etc.  Call once after each
   * new trace is loaded into the backend.
   */
  resetFirstLoad() {
    this._provider.resetFirstLoad();
  }

  /**
   * Fetch graph data from the backend and render it.
   * Delegates to provider.fetch() which emits 'data' and drives _renderFromData.
   */
  render() {
    return this._provider.fetch();
  }

  /** The live Cytoscape instance, or null before first render. */
  get cy() { return this._cy; }

  /**
   * Public ancestor resolver — maps a node ID to the Cytoscape element that
   * should be highlighted when that node might be inside a collapsed region.
   * Returns the node's own element when visible, the nearest visible enclosing
   * compound otherwise, or an empty collection when nothing is visible.
   * Mirrors the private _resolveVisible used internally.
   */
  resolveVisible(nodeId) { return this._resolveVisible(nodeId); }

  /**
   * Highlight multiple nodes simultaneously and animate-fit to `opts.focusId`
   * (or to the bounding box of all nodes when no focusId is given).
   * Nodes inside collapsed regions are resolved to their visible ancestor.
   * opts: { zoom: bool, focusId: string, source: string }
   */
  selectNodes(nodeIds, opts) {
    if (!this._cy) return;
    opts = opts || {};
    var zoom     = (opts.zoom !== false);
    var self     = this;
    this._cy.nodes().removeClass('selected_gutter');
    this._cy.edges().removeClass('edge-highlighted');
    var eles = this._cy.collection();
    (nodeIds || []).forEach(function(id) {
      var ele = self._resolveVisible(id);
      if (ele && ele.length > 0) {
        ele.addClass('selected_gutter');
        ele.connectedEdges().addClass('edge-highlighted');
        eles = eles.union(ele);
      }
    });
    if (zoom && eles.length > 0) {
      var focusEle = opts.focusId ? this._resolveVisible(opts.focusId) : null;
      if (!focusEle || !focusEle.length) focusEle = eles;
      var pad = this._cy.height() / ZOOM_NODE_DIVISOR;
      this._cy.animate({ fit: { eles: focusEle, padding: pad } },
                       { duration: ANIM_ZOOM_MS });
    }
  }

  /**
   * Return the data objects of all currently-selected (selected_gutter) nodes.
   * Useful for reading type, persistent_region_id etc. without touching cy directly.
   */
  selectedNodes() {
    if (!this._cy) return [];
    return this._cy.nodes('.selected_gutter').map(function(n) { return n.data(); });
  }

  /**
   * Test whether the live graph contains at least one element matching selector.
   * Returns false when no graph is loaded yet.
   */
  hasElements(selector) {
    return !!(this._cy && this._cy.$(selector).length > 0);
  }

  /** Animate-fit the viewport to a single node, highlighting it. */
  zoomToFit(nodeId) {
    if (!this._cy) return;
    this._cy.nodes().removeClass('selected_gutter');
    var item = this._cy.$id(nodeId);
    if (!item.length) return;
    item.addClass('selected_gutter');
    var pad = (item.data('type') === 'region' && !item.data('collapsed'))
      ? ZOOM_FIT_PAD : this._cy.height() / ZOOM_NODE_DIVISOR;
    this._cy.animate({ fit: { eles: item, padding: pad } }, { duration: ANIM_ZOOM_MS });
  }

  /** Animate-fit the viewport to a group of nodes. */
  zoomToFitGroup(nodeIds) {
    if (!this._cy) return;
    this._cy.nodes().removeClass('selected_gutter');
    var sel = nodeIds.map(function(id) { return '#' + id; }).join(',');
    var items = this._cy.$(sel);
    items.addClass('selected_gutter');
    this._cy.animate({ fit: { eles: items, padding: ZOOM_FIT_PAD } }, { duration: ANIM_ZOOM_MS });
  }

  /** Download the current graph as a high-resolution PNG. */
  exportImage(filename) {
    if (!this._cy) return;
    var png64 = this._cy.png({ scale: 4, bg: 'white' });
    var a = document.createElement('a');
    a.href     = png64;
    a.download = filename || 'graph.png';
    a.click();
  }

  /**
   * Re-run the layout engine on the existing graph without fetching new data.
   * Useful after the container is resized.
   */
  relayout() {
    if (!this._cy) return;
    var self = this;
    this._renderSeq++;
    var mySeq = this._renderSeq;
    this._showOverlay();
    this._cy.resize();
    var dag_direction = this.settings['direction'] === 'vertical' ? 'TB' : 'LR';
    var daglayout = {
      directed: true, name: 'dagre',
      nodeSep: LAYOUT_NODE_SEP, rankSep: LAYOUT_RANK_SEP,
      rankDir: dag_direction, nodeDimensionsIncludeLabels: true,
      ranker: 'longest-path', acyclicer: 'greedy', elkOptions: {},
      algorithm: (this.settings['graph-mode'] == 'dependence-dag') ? 'layered' : 'mrtree',
    };
    this._startLayoutWorker(mySeq, daglayout, function(positions) {
      if (!self._cy) return;
      self._applyLayoutPositions(positions);
      self._cy.once('render', function() {
        if (mySeq === self._renderSeq) self._hideOverlay();
      });
    });
  }

  /**
   * Update dark-mode state.  Call cy.style().update() after this if a graph
   * is already rendered.
   */
  set darkMode(val) {
    this._darkMode = val;
    if (this._cy) this._cy.style().update();
  }
  get darkMode() { return this._darkMode; }

  /** Attach a LegendView so node/edge hover events highlight the matching entry. */
  set legendView(v) { this._legendView = v; }

  /**
   * Abort any in-progress render (fetch + layout worker) without destroying
   * the view.  The current graph remains visible.  Safe to call at any time.
   */
  stop() {
    this._provider.abort();
    this._cancelCurrentRender();
    this._hideOverlay();
  }

  /** Tear down the Cytoscape instance and clean up. */
  destroy() {
    if (this._currentLayout) { try { this._currentLayout.stop(); } catch (e) {} }
    if (this._cy)            { this._cy.destroy(); this._cy = null; }
  }

  /**
   * Programmatically highlight a node — for driving the graph from an external
   * view such as a Gantt chart, source pane, or info pane.
   *
   * nodeId:      Graph node UUID.
   * opts.zoom:   Animate-zoom to the node (default true).
   * opts.source: Originating view name — if 'graph' the zoom is skipped (the
   *              user already sees it) matching the existing app behaviour.
   */
  selectNode(nodeId, opts) {
    if (!this._cy) return;
    opts = opts || {};
    var zoom = (opts.zoom !== false) && (opts.source !== 'graph');

    this._cy.nodes().removeClass('selected_gutter');
    this._cy.edges().removeClass('edge-highlighted');

    var ele = this._resolveVisible(nodeId);
    if (ele && ele.length > 0) {
      ele.addClass('selected_gutter');
      ele.connectedEdges().addClass('edge-highlighted');
      if (zoom) {
        var pad = (ele.data('type') === 'region' && !ele.data('collapsed'))
          ? ZOOM_FIT_PAD : this._cy.height() / ZOOM_NODE_DIVISOR;
        this._cy.animate({ fit: { eles: ele, padding: pad } }, { duration: ANIM_ZOOM_MS });
      }
    }
  }

  /**
   * Clear all selection highlights and restore full opacity to all nodes.
   * Call this when an external view deselects or the user navigates away.
   */
  clearSelection() {
    if (!this._cy) return;
    this._cy.nodes().removeClass('selected_gutter');
    this._cy.edges().removeClass('edge-highlighted');
    this._opaqueNodes = [];
    this._cy.style().update();
  }

  /**
   * Dim all nodes except those in `ids`.  Pass an empty array (or call
   * clearSelection) to restore full opacity.  Used by the info pane when
   * showing a container_update node whose live-set is known.
   */
  setOpaqueNodes(ids) {
    this._opaqueNodes = ids || [];
    if (this._cy) this._cy.style().update();
  }

  /**
   * Switch the colour spectrum used for skeleton_call / external node colours.
   * The spectrum value is owned by the provider; this method updates it there
   * and triggers an immediate Cytoscape style update.
   */
  setColorSpectrum(name) {
    this._provider.colorSpectrum = name;
    if (this._cy) this._cy.style().update();
  }

  // ── Read-only data accessors (all delegate to the provider) ──────────────────

  /** Raw node array from the last /graph response. */
  get lastGraphNodes() { return this._provider.lastGraphNodes; }
  /** {mode → {min, max}} normalisation ranges from the last render. */
  get lastMinmax()     { return this._provider.lastMinmax;     }

  // ── Color / normalisation helpers (all delegate to the provider) ─────────────

  _spectrumColor(t) {
    return this._provider.spectrumColor(t);
  }

  _spectrumColorDark(t, factor) {
    return this._provider.spectrumColorDark(t, factor);
  }

  _highlightColor() { return this._darkMode ? HIGHLIGHT_DARK : HIGHLIGHT_LIGHT; }

  /**
   * Normalise a Cytoscape element property to [0, 1] for colour/size style functions.
   * Delegates to provider.normFromEle() which owns the minmax table.
   *
   * key:    settings key (e.g. 'node-color-call', 'node-size-call')
   * ele:    Cytoscape element
   * minmax: {mode → {min, max}} from the current render
   */
  _normProp(key, ele, minmax) {
    return this._provider.normFromEle(key, ele, minmax);
  }

  // ── Render pipeline ──────────────────────────────────────────────────────────

  /**
   * Cancel the current in-progress layout/render without touching the fetch layer.
   * Bumps _renderSeq so any pending rAF callbacks self-discard.
   * Safe to call at any time.
   */
  _cancelCurrentRender() {
    if (this._currentLayout) {
      try { this._currentLayout.stop(); } catch (e) {}
      this._currentLayout = null;
    }
    this._renderSeq++;
  }

  /**
   * Render graph data that was delivered by the provider's 'data' event.
   * mySeq is the render-sequence value captured in the 'fetchstart' handler;
   * if it no longer matches _renderSeq a newer render superseded this one.
   *
   * @param {object} rawData   raw server response (nodes, edges, …)
   * @param {object} minmax    normalisation table built by the provider
   * @param {number} mySeq     sequence guard
   */
  _renderFromData(rawData, minmax, mySeq) {
    if (mySeq !== this._renderSeq) { this._hideOverlay(); return; }

    var nodes = rawData.nodes;
    var edges = rawData.edges;

    // Keep a local copy of nodes for _resolveVisible parent-map lookup.
    this._lastGraphNodes = nodes;

    this._startCytoscapeRender(rawData, nodes, edges, minmax, mySeq);
  }

  /**
   * Cytoscape initialisation + layout path.  Called from _renderFromData after
   * the provider delivers data and mySeq is still current.
   */
  _startCytoscapeRender(rawData, nodes, edges, minmax, mySeq) {
    var self = this;

    var dag_direction = this.settings['direction'] === 'vertical' ? 'TB' : 'LR';
    var daglayout = {
      directed: true, name: 'dagre',
      nodeSep: LAYOUT_NODE_SEP, rankSep: LAYOUT_RANK_SEP,
      rankDir: dag_direction,
      nodeDimensionsIncludeLabels: true,
      ranker: 'longest-path', acyclicer: 'greedy',
      elkOptions: {},
      algorithm: (this.settings['graph-mode'] == 'dependence-dag') ? 'layered' : 'mrtree',
    };

    var renderStart = Date.now();
    var stats = {
      nodeCount:     nodes.length,
      edgeCount:     edges.length,
      eventCount:    rawData.event_count    || 0,
      snapshotCount: rawData.snapshot_count || 0,
      modelingMs: (rawData.request_time != null && rawData.response_time != null)
        ? Math.round((rawData.response_time - rawData.request_time) * 1000) : null,
      renderingMs:   null,
      timelineNodes: rawData.timeline_nodes || [],
      graphNodes:    nodes,
      badgeNodes:    rawData.badge_nodes || nodes,
      lastMinmax:    minmax,
    };

    // Yield one animation frame so the overlay is painted before the
    // synchronous Cytoscape init blocks the thread.
    requestAnimationFrame(function() {
      if (mySeq !== self._renderSeq) { self._hideOverlay(); return; }

      // Destroy the previous instance to release memory and event listeners.
      if (self._cy) self._cy.destroy();
      self._groupedEdgeReps.clear();
      self._groupedEdgeHidden.clear();
      self._bidirEdgeReps.clear();
      self._cyEl.innerHTML = '';

      self._cy = cytoscape({
        container: self._cyEl,
        elements:  edges.concat(nodes),
        wheelSensitivity: 0.0,
        minZoom: 0.01, maxZoom: 5,
        userZoomingEnabled: true,
        style:  self._buildStyle(minmax, rawData),
        layout: { name: 'null' },
      });

      // Mark region nodes as phantoms when region display is off — they stay
      // in the graph for ELK's hierarchical layout but are invisible.
      if (!self.settings['show_regions']) {
        self._cy.nodes('[type="region"]').addClass('layout-phantom');
      }

      self._bindCyEvents(rawData, daglayout, mySeq);

      // Second animation frame: Cytoscape has rendered once, style functions
      // have produced valid node sizes — safe to start layout.
      requestAnimationFrame(function() {
        if (mySeq !== self._renderSeq || !self._cy) {
          if (mySeq === self._renderSeq) self._hideOverlay();
          return;
        }
        self._cy.resize();
        self._applyEdgeGrouping();
        self._startLayoutWorker(mySeq, daglayout, function(positions) {
          if (!self._cy) return;
          self._applyLayoutPositions(positions);
          self._cy.once('render', function() {
            if (mySeq !== self._renderSeq) return;
            // Remove the snapshot canvas before fading the overlay so the
            // new Cytoscape canvas is already visible when the fade begins.
            var snap = self._overlayEl.querySelector('.gv-snap');
            if (snap) snap.remove();
            self._hideOverlay();
            stats.renderingMs = Date.now() - renderStart;
            if (self._onRenderEnd) self._onRenderEnd(stats);
            // Notify listeners (e.g. SourceCodeView) that the layout and first
            // Cytoscape frame are complete.
            self._provider._emit('renderend', stats);
          });
        });
      });
    });
  }

  // ── Cytoscape style ──────────────────────────────────────────────────────────
  // Returns the style array for cytoscape({style: ...}).  Captures self, minmax,
  // and per-render settings snapshots into closures so style functions always
  // refer to the correct instance state even after settings change.

  _buildStyle(minmax, data) {
    var self         = this;
    var node_labels  = this.settings['node-labels'];
    var edge_labels  = this.settings['edge-labels'];
    var transfer_map = { 'host-to-device': '↑', 'device-to-host': '↓' };

    function dark()     { return self._darkMode; }
    function hcol()     { return self._highlightColor(); }
    function sc(t)      { return self._spectrumColor(t); }
    function scd(t)     { return self._spectrumColorDark(t); }
    function norm(k, e) { return self._normProp(k, e, minmax); }

    // Region colours — match the regionColorLight/Dark values used in LegendView.
    function regionColorLight(d) {
      return 'rgb(' + Math.max(20, 225 - d*34) + ',' + Math.max(185, 240 - d*9) + ',255)';
    }
    function regionColorDark(d) {
      return 'rgb(' + Math.min(100, 42+d*15) + ',' + Math.min(74, 31+d*10) + ',' + Math.min(42, 16+d*6) + ')';
    }

    return [
      {
        selector: 'node',
        style: {
          'color':              function() { return dark() ? 'white' : 'black'; },
          'text-outline-color': function() { return dark() ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.5)'; },
          'text-outline-width': 0.8,
          'font-size': '0.6em',
          'min-zoomed-font-size': '0.4em',
          'opacity': function(ele) {
            if (self._opaqueNodes.length === 0) return 1;
            return self._opaqueNodes.includes(ele.data('id')) ? 1 : 0.2;
          },
        }
      },
      {
        selector: 'node[type="allocation"]',
        style: {
          'content':          function(ele) { return node_labels ? ele.data('label') : ''; },
          'text-valign':      'center',
          'shape':            'star',
          'border-width':     1,
          'background-color': 'white',
          'border-color':     '#333',
        }
      },
      {
        selector: 'node[type="deallocation"]',
        style: {
          'content':          function(ele) { return node_labels ? ele.data('label') : ''; },
          'text-valign':      'center',
          'shape':            'star',
          'border-width':     1,
          'background-color': 'black',
          'border-color':     '#ddd',
        }
      },
      {
        selector: 'node[type="external"]',
        style: {
          'content':          function(ele) { return node_labels ? ele.data('label') : ''; },
          'text-valign':      'center',
          'shape':            'octagon',
          'background-color': function(ele) { return sc(norm('node-color-call', ele)); },
          'border-width':     3,
          'border-color':     function(ele) { return scd(norm('node-color-call', ele)); },
          'width':  function(ele) { return NODE_SIZE_SCALE * Math.sqrt(norm('node-size-call', ele)) + NODE_SIZE_MIN; },
          'height': function(ele) { return NODE_SIZE_SCALE * Math.sqrt(norm('node-size-call', ele)) + NODE_SIZE_MIN; },
        }
      },
      {
        selector: 'node[type="transfer"]',
        style: {
          'content':          function(ele) { return node_labels ? ele.data('label') + (transfer_map[ele.data('direction')] || '') : ''; },
          'text-valign':      'center',
          'shape':            'diamond',
          'background-color': 'purple',
          'border-width':     3,
          'border-color':     '#6a006a',
        }
      },
      {
        selector: 'node[type="container_update"], node[type="scalar"]',
        style: {
          'content': function(ele) {
            if (!node_labels) return '';
            return ele.data('iteration_count') > 1
              ? ele.data('label') + ' x' + ele.data('iteration_count')
              : ele.data('label') + '' + toSubscript(ele.data('version'));
          },
          'text-valign':      'center',
          'shape':            'rectangle',
          'height':           20, 'width': 45,
          'background-color': 'lightgray',
          'border-width':     3,
          'border-color':     '#979797',
        }
      },
      {
        selector: 'node[type="container_update"][has_snapshot]',
        style: {
          'background-color': function() { return dark() ? '#3a2e00' : '#fff0a0'; },
          'border-color':     function() { return dark() ? '#c8a000' : '#b8860b'; },
        }
      },
      {
        selector: 'node[type="scalar"]',
        style: {
          'content':       function(ele) { return node_labels ? ele.data('label') : ''; },
          'text-margin-y': 2,
          'shape':         'triangle',
          'height':        20, 'width': 25,
        }
      },
      {
        selector: 'node[type="skeleton_call"]',
        style: {
          'content': function(ele) {
            if (!node_labels) return '';
            return ele.data('iteration_count') > 1
              ? ele.data('label') + ' x' + ele.data('iteration_count')
              : ele.data('label');
          },
          'shape':      'ellipse',
          'text-valign': 'center',
          'background-color': function(ele) {
            var mode = self.settings['node-color-call'];
            if (mode === 'backend') {
              return ({ CPU:'yellow', OpenMP:'red', OpenCL:'blue', CUDA:'green' })[ele.data('backend')] || sc(norm('node-color-call', ele));
            }
            if (mode === 'pattern') {
              return ({ Map:'yellow', Reduce:'red', MapReduce:'orange', MapOverlap:'green' })[ele.data('pattern')] || sc(norm('node-color-call', ele));
            }
            return sc(norm('node-color-call', ele));
          },
          'border-width': 3,
          'border-color': function(ele) {
            var mode = self.settings['node-color-call'];
            if (mode === 'backend') {
              return ({ CPU:'#b3b300', OpenMP:'#b30000', OpenCL:'#0000b3', CUDA:'#007a00' })[ele.data('backend')] || '#555';
            }
            if (mode === 'pattern') {
              return ({ Map:'#b3b300', Reduce:'#b30000', MapReduce:'#b37a00', MapOverlap:'#007a00' })[ele.data('pattern')] || '#555';
            }
            return scd(norm('node-color-call', ele));
          },
          'width':  function(ele) { return NODE_SIZE_SCALE * Math.sqrt(norm('node-size-call', ele)) + NODE_SIZE_MIN; },
          'height': function(ele) { return NODE_SIZE_SCALE * Math.sqrt(norm('node-size-call', ele)) + NODE_SIZE_MIN; },
        }
      },
      {
        selector: 'node[type="region"]',
        style: {
          'content': function(ele) {
            if (!node_labels) return '';
            return ele.data('iteration_count') > 1
              ? ele.data('label') + ' x' + ele.data('iteration_count')
              : ele.data('label');
          },
          'border-color':     function() { return dark() ? '#170D00' : '#006770'; },
          'border-width':     0,
          'background-color': function(ele) {
            var d = ele.data('region_depth') || 0;
            return dark() ? regionColorDark(d) : regionColorLight(d);
          },
        }
      },
      {
        selector: 'node[type="fusion"]',
        style: {
          'content':          function(ele) { return node_labels ? ele.data('label') : ''; },
          'background-color': '#88f',
        }
      },
      {
        selector: 'node[type="fusion"][collapsed]',
        style: {
          'content':          function(ele) { return node_labels ? ele.data('label') : ''; },
          'text-halign':      'center', 'text-valign': 'center',
          'background-color': '#88f',
        }
      },
      {
        selector: 'node[type="region"][collapsed], node[type="tree-region"]',
        style: {
          'content': function(ele) {
            if (!node_labels) return '';
            return ele.data('iteration_count') > 1
              ? ele.data('label') + ' x' + ele.data('iteration_count')
              : ele.data('label');
          },
          'shape':            'round-rectangle',
          'text-valign':      'center',
          'border-width':     3,
          'height':           40, 'width': 64,
          'border-color':     function() { return dark() ? '#733103' : '#006770'; },
          'background-color': function() { return dark() ? '#3B1A02' : 'skyblue'; },
        }
      },
      {
        selector: 'node.layout-phantom',
        style: {
          'background-opacity': 0, 'border-width': 0,
          'label': '', 'events': 'no', 'padding': 0,
        }
      },
      {
        selector: '.selected_gutter',
        style: { 'border-width': 5, 'border-color': function() { return hcol(); } }
      },
      {
        selector: 'edge.grouped-hidden',
        style: { 'display': 'none' }
      },
      {
        selector: 'edge',
        style: {
          'content': function(ele) {
            if (self._groupedEdgeReps.has(ele.id())) return ele.data('label') || '';
            return edge_labels ? (ele.data('label') || '') : '';
          },
          'text-outline-width': 0.8,
          'text-outline-color': function() { return dark() ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.5)'; },
          'color':  function() { return dark() ? 'white' : 'black'; },
          'font-size': '0.5em',
          'width': function(ele) {
            var m = self.settings['edge-width'];
            if (m === 'critical-path' || m === 'cross-backend') {
              return (m === 'critical-path' ? ele.data('is_critical_path') : ele.data('cross_backend')) ? 5 : 1;
            }
            if (m === 'iterations') {
              var mm2 = minmax['iterations'];
              if (!mm2 || mm2.max === mm2.min) return 1.5;
              return 1 + Math.max(0, Math.min(1, (ele.data('iteration_count') - mm2.min) / (mm2.max - mm2.min))) * 5;
            }
            if (m === 'order-span') {
              var mm3 = minmax['order-span'];
              if (!mm3 || mm3.max === mm3.min) return 1.5;
              return 1 + Math.max(0, Math.min(1, (ele.data('order_span') - mm3.min) / (mm3.max - mm3.min))) * 5;
            }
            if (m === 'elements') {
              var wels = ele.source().data('elements');
              if (!wels) return 1.5;
              var mm4 = minmax['elements'];
              if (!mm4 || mm4.max === mm4.min) return 1.5;
              return 1 + Math.max(0, Math.min(1, (wels.reduce(function(a,b){return a*b;},1) - mm4.min) / (mm4.max - mm4.min))) * 5;
            }
            return 1.5;
          },
          'target-arrow-shape': 'triangle',
          'line-color':         function() { return dark() ? '#ccc' : '#333'; },
          'target-arrow-color': function() { return dark() ? '#ccc' : '#333'; },
          'curve-style':  'bezier',
          'taxi-direction': 'downward',
          'opacity': function(ele) {
            // opaqueNodes dimming takes priority over all settings-based modes.
            if (self._opaqueNodes.length > 0 &&
                !self._opaqueNodes.includes(ele.data('id'))) return 0.2;
            var m = self.settings['edge-opacity'];
            if (m === 'critical-path' || m === 'cross-backend') {
              return (m === 'critical-path' ? ele.data('is_critical_path') : ele.data('cross_backend')) ? 1 : 0.15;
            }
            if (m === 'iterations') {
              var mm2 = minmax['iterations'];
              if (!mm2 || mm2.max === mm2.min) return 1;
              return 0.15 + Math.max(0, Math.min(1, (ele.data('iteration_count') - mm2.min) / (mm2.max - mm2.min))) * 0.85;
            }
            if (m === 'order-span') {
              var mm3 = minmax['order-span'];
              if (!mm3 || mm3.max === mm3.min) return 1;
              return 0.15 + Math.max(0, Math.min(1, (ele.data('order_span') - mm3.min) / (mm3.max - mm3.min))) * 0.85;
            }
            if (m === 'elements') {
              var oels = ele.source().data('elements');
              if (!oels) return 1;
              var mm4 = minmax['elements'];
              if (!mm4 || mm4.max === mm4.min) return 1;
              return 0.15 + Math.max(0, Math.min(1, (oels.reduce(function(a,b){return a*b;},1) - mm4.min) / (mm4.max - mm4.min))) * 0.85;
            }
            return 1;
          },
        }
      },
      {
        selector: 'edge[type="anti-dep"]',
        style: { 'line-color': 'red', 'target-arrow-color': 'red' }
      },
      {
        selector: 'edge[access_mode="elwise"]',
        style: { 'line-style': 'dashed' }
      },
      {
        selector: 'edge[access_mode="scalar"]',
        style: { 'line-style': 'dotted' }
      },
      {
        selector: 'edge[type="virtual-alias"]',
        style: {
          'line-color':         function() { return dark() ? '#bb8fce' : '#8e44ad'; },
          'line-style':         'dashed',
          'line-dash-pattern':  [8, 4],
          'target-arrow-shape': 'none',
          'source-arrow-shape': 'none',
          'width':              1.5,
          'opacity':            0.65,
        }
      },
      {
        selector: 'edge[type="prng-dep"]',
        style: {
          'curve-style':       'segments',
          'segment-weights':   '0.2 0.4 0.6 0.8',
          'segment-distances': '3 -3 3 -3',
          'opacity':           0.65,
        }
      },
      {
        selector: ':selected, :active',
        style: { 'overlay-opacity': 0 }
      },
      {
        selector: 'edge.edge-highlighted',
        style: {
          'line-color':         function() { return hcol(); },
          'target-arrow-color': function() { return hcol(); },
          'width': 3,
        }
      },
      {
        selector: 'edge.edge-bidir',
        style: {
          'source-arrow-shape': 'triangle',
          'source-arrow-color': function() { return dark() ? '#ccc' : '#333'; },
        }
      },
      {
        selector: 'edge.edge-bidir[type="anti-dep"]',
        style: { 'source-arrow-color': 'red' }
      },
      {
        selector: 'edge.edge-highlighted[type="anti-dep"]',
        style: { 'line-color': 'red', 'target-arrow-color': 'red', 'width': 3 }
      },
      {
        selector: 'node.cy-hovered',
        style: { 'border-width': 6 }
      },
      {
        selector: 'edge.cy-hovered',
        style: { 'width': 4 }
      },
    ];
  }

  // ── Legend key helpers ────────────────────────────────────────────────────────

  /** Map a node type string to its data-legend-key value, or null if unmapped. */
  _typeToLegendKey(type) {
    return ({
      'skeleton_call':    'skeleton',
      'external':         'external',
      'scalar':           'scalar-node',
      'transfer':         'transfer',
      'container_update': 'container_update',
      'allocation':       'allocation',
      'deallocation':     'deallocation',
      'region':           'region',
      'fusion':           'fusion',
    })[type] || null;
  }

  /** Map an edge element to its data-legend-key value, or null if unmapped. */
  _edgeToLegendKey(ele) {
    var d = ele.data();
    if (d.type === 'anti-dep')      return 'anti-dep';
    if (d.type === 'virtual-alias') return 'virtual-alias';
    if (d.type === 'prng-dep')      return 'prng-dep';
    if (d.access_mode === 'elwise') return 'elwise';
    if (d.access_mode === 'scalar') return 'scalar';
    if (d.type === 'forward-dep')   return 'proxy';
    return null;
  }

  // ── Cytoscape event binding ──────────────────────────────────────────────────

  _bindCyEvents(data, daglayout, mySeq) {
    var self = this;
    var cy   = this._cy;

    // ── Background click: broadcast 'clear' to all attached views ───────────
    cy.on('click', function() {
      self._provider._emit('clear');
    });

    // ── Node click ───────────────────────────────────────────────────────────
    cy.on('click', 'node', function(event) {
      if (event.target.hasClass('layout-phantom')) return;
      var ele    = event.target;
      var nodeId = ele.data('id');

      cy.nodes().removeClass('selected_gutter');
      cy.edges().removeClass('edge-highlighted');
      ele.addClass('selected_gutter');
      ele.connectedEdges().addClass('edge-highlighted');

      // Broadcast via the shared event bus so every attached view
      // (TimelineView, SourceCodeView, …) can update itself.  The 'graph'
      // source tag prevents this GraphView from receiving its own event back.
      self._provider._emit('select', {
        nodeId:  nodeId,
        source:  'graph',
        focusId: nodeId,
        allIds:  [nodeId],
      });
    });

    // ── Region double-click: toggle collapsed/expanded and re-render ─────────
    cy.on('dblclick', 'node[type="region"]', function(event) {
      event.stopPropagation();
      if (event.target.hasClass('layout-phantom')) return;
      var pid = event.target.data('persistent_region_id');
      if (!pid) return;
      var collapsed = self._provider.collapsedNodes;
      collapsed[pid] = (collapsed[pid] === false);
      self.render();
    });

    // ── Edge click ───────────────────────────────────────────────────────────
    cy.on('click', 'edge', function(event) {
      event.stopPropagation();
      var ele = event.target;
      var d   = ele.data();

      cy.edges().removeClass('edge-highlighted');
      cy.nodes().removeClass('selected_gutter');
      ele.addClass('edge-highlighted');
      cy.$id(d.source).addClass('selected_gutter');
      cy.$id(d.target).addClass('selected_gutter');

      // If this is a grouped representative, also highlight all group members.
      if (self._groupedEdgeReps.has(ele.id())) {
        cy.edges().each(function(e) {
          var ed = e.data();
          if (ed.source === d.source && ed.target === d.target &&
              (ed.type || '') === (d.type || '') &&
              (ed.access_mode || '') === (d.access_mode || '')) {
            e.addClass('edge-highlighted');
          }
        });
      }

      if (self._onEdgeClick) self._onEdgeClick(d, cy);
    });

    // ── Scroll / pinch-zoom (trackpad / touch) ───────────────────────────────
    this._cyEl.addEventListener('wheel', function(event) {
      event.preventDefault();
      if (!self._isZooming)
        cy.panBy({ x: event.deltaX * -1.5, y: event.deltaY * -1.5 });
    }, { passive: false });

    this._cyEl.addEventListener('gesturestart', function(e) {
      self._baseZoom = cy.zoom();
      self._isZooming = true;
    });
    this._cyEl.addEventListener('gestureend', function() {
      self._isZooming = false;
    });
    this._cyEl.addEventListener('gesturechange', function(e) {
      var rect = e.target.getBoundingClientRect();
      cy.zoom({
        level: Math.pow(e.scale, 1.2) * self._baseZoom,
        renderedPosition: { x: e.clientX - rect.left, y: e.clientY - rect.top },
      });
    });

    // ── Hover tooltip + border highlight + legend highlight ──────────────────
    cy.on('mouseover', 'node, edge', function(event) {
      var target = event.target;
      if (target.hasClass('layout-phantom')) return;
      target.addClass('cy-hovered');
      var content = target.isNode()
        ? self._nodeTooltipContent(target)
        : self._edgeTooltipContent(target);
      if (content) _showTooltip(content, event.originalEvent.clientX, event.originalEvent.clientY);
      if (self._legendView) {
        var key = target.isNode()
          ? self._typeToLegendKey(target.data('type'))
          : self._edgeToLegendKey(target);
        self._legendView.highlightKey(key);
      }
      if (target.isNode()) {
        if (self._onNodeHover) self._onNodeHover(target.data('id'));
      } else {
        if (self._onEdgeHover) self._onEdgeHover(target.data('source'), target.data('target'));
      }
    });
    cy.on('mousemove', 'node, edge', function(event) {
      _positionTooltip(event.originalEvent.clientX, event.originalEvent.clientY);
    });
    cy.on('mouseout mousedown', 'node, edge', function(event) {
      event.target.removeClass('cy-hovered');
      _hideTooltip();
      if (self._legendView) self._legendView.clearHighlight();
      if (event.target.isNode()) {
        if (self._onNodeHoverEnd) self._onNodeHoverEnd(event.target.data('id'));
      } else {
        if (self._onEdgeHoverEnd) self._onEdgeHoverEnd(event.target.data('source'), event.target.data('target'));
      }
    });
    cy.on('mousedown viewport', _hideTooltip);
    this._cyEl.addEventListener('mouseleave', function() {
      cy.elements().removeClass('cy-hovered');
      _hideTooltip();
      if (self._legendView) self._legendView.clearHighlight();
      if (self._onNodeHoverEnd)  self._onNodeHoverEnd(null);
      if (self._onEdgeHoverEnd)  self._onEdgeHoverEnd(null, null);
    });
  }

  // ── Tooltip content helpers ──────────────────────────────────────────────────

  _nodeTooltipContent(ele) {
    var d = ele.data(), lines = [];
    if (d.label) lines.push(d.label);
    var dur = _fmtDur(d.duration);
    if (dur) lines.push('time     ' + dur);
    if (d.elements && d.elements.length) lines.push('size     ' + d.elements.join(' \xd7 '));
    if (d.backend) lines.push('backend  ' + d.backend);
    if (d.pattern) lines.push('pattern  ' + d.pattern);
    if (d.file && d.line != null && d.line !== -1)
      lines.push(d.file.split('/').pop().split('\\').pop() + ':' + d.line);
    return lines.join('\n');
  }

  _edgeTooltipContent(ele) {
    var d = ele.data();
    var typeLabel = d.type === 'forward-dep'   ? 'Data dependence'
                  : d.type === 'anti-dep'      ? 'Anti-dependence'
                  : d.type === 'virtual-alias' ? 'Virtual alias: ' + (d.virtual_label || '')
                  : d.type === 'prng-dep'      ? 'PRNG dep: '      + (d.prng_label   || '')
                  : d.type || 'Edge';
    var lines = [typeLabel];
    if (d.label)       lines.push(d.label);
    if (d.access_mode) lines.push('mode  ' + d.access_mode);
    return lines.join('\n');
  }

  // ── Visible-node resolution ──────────────────────────────────────────────────

  /**
   * Return the Cytoscape collection for nodeId.  If the node is hidden inside
   * a collapsed compound region, walk up the parent chain (using the last raw
   * graph data) until a visible ancestor is found.  Returns an empty collection
   * if no visible target exists.
   */
  _resolveVisible(nodeId) {
    if (!this._cy) return null;
    var el = this._cy.$id(nodeId);
    if (el.length > 0) return el;
    // Build a quick id→parent map from the stored raw node data.
    var parentMap = {};
    this._lastGraphNodes.forEach(function(n) {
      if (n.data && n.data.id && n.data.parent)
        parentMap[n.data.id] = n.data.parent;
    });
    var cur = parentMap[nodeId];
    while (cur) {
      var ancestor = this._cy.$id(cur);
      if (ancestor.length > 0) return ancestor;
      cur = parentMap[cur];
    }
    return this._cy.collection();
  }

  // ── Edge grouping ────────────────────────────────────────────────────────────

  _applyEdgeGrouping() {
    var cy = this._cy;
    if (!cy) return;

    // Reset previous pass.
    cy.edges().each(e => {
      if (this._groupedEdgeHidden.has(e.id())) e.removeClass('grouped-hidden');
      if (this._groupedEdgeReps.has(e.id())) {
        var orig = e.scratch('_origLabel');
        if (orig !== undefined) e.data('label', orig);
      }
      if (this._bidirEdgeReps.has(e.id())) e.removeClass('edge-bidir');
    });
    this._groupedEdgeReps.clear();
    this._groupedEdgeHidden.clear();
    this._bidirEdgeReps.clear();

    // Group parallel edges (same source, target, type, access_mode).
    var groups = {};
    cy.edges().each(function(e) {
      if (!e.source().visible() || !e.target().visible()) return;
      var d   = e.data();
      var key = d.source + '\x00' + d.target + '\x00' + (d.type || '') + '\x00' + (d.access_mode || '');
      (groups[key] = groups[key] || []).push(e);
    });

    var reps   = this._groupedEdgeReps;
    var hidden = this._groupedEdgeHidden;
    Object.keys(groups).forEach(function(key) {
      var grp = groups[key];
      if (grp.length < 2) return;
      var rep = grp.find(function(e) { return e.data('is_critical_path'); }) || grp[0];
      if (rep.scratch('_origLabel') === undefined)
        rep.scratch('_origLabel', rep.data('label') !== undefined ? rep.data('label') : '');
      rep.data('label', 'x' + grp.length);
      reps.add(rep.id());
      grp.forEach(function(e) {
        if (e.id() !== rep.id()) { e.addClass('grouped-hidden'); hidden.add(e.id()); }
      });
    });

    // Bidirectional pass: merge (A→B) and (B→A) into one double-headed edge.
    var bidirs = this._bidirEdgeReps;
    Object.keys(groups).forEach(function(key) {
      var parts  = key.split('\x00');
      if (parts[2] === 'virtual-alias' || parts[2] === 'prng-dep') return;
      var revKey = parts[1] + '\x00' + parts[0] + '\x00' + parts[2] + '\x00' + parts[3];
      if (key >= revKey || !groups[revKey]) return;
      var fwdGrp = groups[key];
      var fwdRep = fwdGrp.find(function(e) { return reps.has(e.id()); }) || fwdGrp[0];
      groups[revKey].forEach(function(e) {
        if (!e.hasClass('grouped-hidden')) { e.addClass('grouped-hidden'); hidden.add(e.id()); }
      });
      fwdRep.addClass('edge-bidir');
      bidirs.add(fwdRep.id());
    });
  }

  // ── Layout workers ────────────────────────────────────────────────────────────
  // Self-contained: spacing constants are defined locally above.

  _startLayoutWorker(layoutSeq, layoutOpts, onPositions) {
    var traceHasRegions = this._provider.allRegionPids.length > 0;
    if (traceHasRegions) {
      this._startElkLayout(layoutSeq, layoutOpts, onPositions);
    } else {
      this._startDagreLayout(layoutSeq, layoutOpts, onPositions);
    }
  }

  _startDagreLayout(layoutSeq, layoutOpts, onPositions) {
    var self = this;
    var cy   = this._cy;

    // Serialise nodes for the worker.
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

    var worker = new Worker(this._dagreWorkerUrl);
    this._currentLayout = { stop: function() { worker.terminate(); self._currentLayout = null; } };

    worker.onmessage = function(ev) {
      worker.terminate();
      self._currentLayout = null;
      if (layoutSeq !== self._renderSeq) return;
      if (ev.data.type === 'done') onPositions(ev.data.positions);
      else { console.error('[GraphView] dagre-worker error:', ev.data.message); self._hideOverlay(); }
    };
    worker.onerror = function(err) {
      worker.terminate();
      self._currentLayout = null;
      if (layoutSeq === self._renderSeq) self._hideOverlay();
      console.error('[GraphView] dagre-worker uncaught error:', err.message || err);
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
  }

  _startElkLayout(layoutSeq, layoutOpts, onPositions) {
    var self = this;
    var cy   = this._cy;
    var dir  = layoutOpts.rankDir === 'LR' ? 'RIGHT' : 'DOWN';

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
      'algorithm':                                 layoutOpts.algorithm,
      'elk.direction':                             dir,
      'elk.hierarchyHandling':                     'INCLUDE_CHILDREN',
      'elk.spacing.nodeNode':                      layoutOpts.nodeSep != null ? layoutOpts.nodeSep : LAYOUT_NODE_SEP,
      'elk.layered.spacing.nodeNodeBetweenLayers': layoutOpts.rankSep != null ? layoutOpts.rankSep : LAYOUT_RANK_SEP,
      'elk.layered.cycleBreaking.strategy':        'GREEDY',
      'elk.layered.nodePlacement.strategy':        'NETWORK_SIMPLEX',
      'elk.edgeRouting':                           'SPLINES',
    }, layoutOpts.elkOptions || {});

    var elk = _getElk(this._elkWorkerUrl);
    this._currentLayout = { stop: function() { self._currentLayout = null; } };

    elk.layout({ id: 'root', layoutOptions: elkOptions, children: rootChildren, edges: elkEdges })
      .then(function(result) {
        self._currentLayout = null;
        if (layoutSeq !== self._renderSeq) return;
        var positions = {};
        _collectElkPositions(result.children, positions, 0, 0);
        onPositions(positions);
      })
      .catch(function(err) {
        self._currentLayout = null;
        _sharedElk = null;  // invalidate on error; next render gets a fresh instance
        if (layoutSeq === self._renderSeq) self._hideOverlay();
        console.error('[GraphView] ELK layout error:', err);
      });
  }

  _applyLayoutPositions(positions) {
    this._cy.nodes().not(':parent').positions(function(n) {
      return positions[n.id()] || n.position();
    });
    this._cy.fit(undefined, LAYOUT_FIT_PAD);
  }

  // ── Overlay (loading transition) ─────────────────────────────────────────────

  _showOverlay() {
    // Capture the current Cytoscape canvas layers into a static snapshot image
    // so the user sees the old graph while the new one is being built and laid out.
    // If there is no previous graph (first render) we just show the blank overlay.
    if (this._cy) {
      var first = this._cyEl.querySelector('canvas');
      if (first && first.width > 0 && first.height > 0) {
        // Remove any leftover snapshot from a previous render cycle.
        var prev = this._overlayEl.querySelector('.gv-snap');
        if (prev) prev.remove();

        var snap = document.createElement('canvas');
        snap.className = 'gv-snap';
        snap.width  = first.width;
        snap.height = first.height;
        snap.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;';
        var ctx = snap.getContext('2d');
        this._cyEl.querySelectorAll('canvas').forEach(function(c) {
          try { ctx.drawImage(c, 0, 0); } catch (e) {}
        });
        this._overlayEl.appendChild(snap);
      }
    }
    this._overlayEl.style.opacity      = '1';
    this._overlayEl.style.pointerEvents = 'all';
  }

  _hideOverlay() {
    // The snapshot canvas is removed by the caller (inside cy.once('render'))
    // before this is called, so the new Cytoscape canvas is already visible
    // underneath when the overlay fades out — no flash.
    this._overlayEl.style.opacity      = '0';
    this._overlayEl.style.pointerEvents = 'none';
  }
}

// ── Module-level pure utilities ───────────────────────────────────────────────

function _fmtDur(us) {
  if (us == null || us < 0) return null;
  if (us >= 1e6)  return (us / 1e6).toFixed(1)  + ' s';
  if (us >= 1000) return (us / 1000).toFixed(1) + ' ms';
  return us.toFixed(0) + ' μs';
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

// ── LegendView ────────────────────────────────────────────────────────────────
//
// Renders the legend panel for a GraphView.  Defined in GraphView.js because
// it is tightly coupled: it calls graphView.hasElements() to filter out entry
// types that are not present in the current graph, and graphView._spectrumColor()
// / _spectrumColorDark() for the spectrum gradient.
//
// Usage (see graph.js):
//
//   import { GraphView, LegendView } from './GraphView.js';
//
//   const legend = new LegendView(document.getElementById('legend'), graphView);
//   legend.settings = sharedSettingsObject;   // same reference as graphView.settings
//   legend.render(darkMode);                  // call after each render / dark-mode change
//
export class LegendView {
  // legendEl  — the #legend container element (safe to pass null; render() is a no-op)
  // graphView — the associated GraphView instance (used for hasElements + spectrum colors)
  constructor(legendEl, graphView) {
    this._legendEl  = legendEl;
    this._gv        = graphView;
    this.settings   = {};   // set by graph.js to the shared settings reference
  }

  // Build or rebuild the full legend panel.
  render(darkMode) {
    if (!this._legendEl) return;

    var dark    = !!darkMode;
    var midClr  = this._gv._spectrumColor(0.15);
    var midDark = this._gv._spectrumColorDark(0.15);

    // ── Theme-aware colours ────────────────────────────────────────────────
    var textClr   = dark ? '#c8c0d8' : '#333333';
    var edgeClr   = dark ? '#aaaaaa' : '#444444';
    var ctFill    = '#cccccc';   // container_update / scalar
    var ctStroke  = '#979797';   // darkened version of fill
    var allocFill = 'white';     // allocation — matches Cytoscape background-color
    var allocStr  = '#333';      // allocation border
    var dealFill  = 'black';     // deallocation — matches Cytoscape background-color
    var dealStr   = '#ddd';      // deallocation border — light against dark fill
    // region depth-0 colours, matching regionColorLight/Dark in GraphView
    var regionF   = dark ? 'rgb(42,31,16)'        : 'rgb(225,240,255)';
    var regionS   = dark ? 'rgba(200,160,80,0.5)' : 'rgba(70,155,220,0.6)';
    var fusionF   = '#aaaaff';
    var fusionS   = '#6655cc';

    // ── SVG shape helpers ──────────────────────────────────────────────────
    var STAR = '10,2 11.8,7.6 17.6,7.5 12.9,10.9 14.7,16.5 10,13 5.3,16.5 7.2,10.9 2.4,7.5 8.2,7.6';
    // Octagon rotated 22.5° so edges (not vertices) face top and bottom.
    var OCTA = '17.4,6.9 17.4,13.1 13.1,17.4 6.9,17.4 2.6,13.1 2.6,6.9 6.9,2.6 13.1,2.6';

    function nodeIcon(inner) {
      return '<svg viewBox="0 0 20 20" width="18" height="18">' + inner + '</svg>';
    }

    function edgeIcon(dashArray, color) {
      color = color || edgeClr;
      var da = dashArray ? ' stroke-dasharray="' + dashArray + '"' : '';
      return '<svg viewBox="0 0 44 12" width="40" height="12">' +
        '<line x1="2" y1="6" x2="34" y2="6" stroke="' + color + '" stroke-width="1.5"' + da + '/>' +
        '<polygon points="44,6 32,2 32,10" fill="' + color + '"/>' +
        '</svg>';
    }

    // Returns true if the current graph contains at least one element matching selector.
    var gv = this._gv;
    function inGraph(selector) {
      return gv.hasElements(selector);
    }

    // ── Node entries ───────────────────────────────────────────────────────
    var nodeEntries = [];
    if (inGraph('node[type="skeleton_call"]'))
      nodeEntries.push([nodeIcon('<circle cx="10" cy="10" r="9" fill="' + midClr + '" stroke="' + midDark + '" stroke-width="1.5"/>'),
       'Skeleton', 'skeleton']);
    if (inGraph('node[type="external"]'))
      nodeEntries.push([nodeIcon('<polygon points="' + OCTA + '" fill="' + midClr + '" stroke="' + midDark + '" stroke-width="1.5"/>'),
       'External', 'external']);
    if (inGraph('node[type="scalar"]'))
      nodeEntries.push([nodeIcon('<polygon points="10,2 18,17 2,17" fill="' + ctFill + '" stroke="' + ctStroke + '" stroke-width="2"/>'),
       'Scalar', 'scalar-node']);
    if (inGraph('node[type="transfer"]'))
      nodeEntries.push([nodeIcon('<polygon points="10,1 19,10 10,19 1,10" fill="purple" stroke="#6a006a" stroke-width="1.5"/>'),
       'Transfer', 'transfer']);
    if (inGraph('node[type="container_update"]'))
      nodeEntries.push([nodeIcon('<rect x="3" y="3" width="14" height="14" fill="' + ctFill + '" stroke="' + ctStroke + '" stroke-width="2"/>'),
       'Data Update', 'container_update']);
    if (inGraph('node[type="allocation"]'))
      nodeEntries.push([nodeIcon('<polygon points="' + STAR + '" fill="' + allocFill + '" stroke="' + allocStr + '" stroke-width="1"/>'),
       'Allocation', 'allocation']);
    if (inGraph('node[type="deallocation"]'))
      nodeEntries.push([nodeIcon('<polygon points="' + STAR + '" fill="' + dealFill + '" stroke="' + dealStr + '" stroke-width="1"/>'),
       'Deallocation', 'deallocation']);
    if (this.settings['show_regions'] && inGraph('node[type="region"]'))
      nodeEntries.push([nodeIcon('<rect x="1" y="3" width="18" height="14" rx="2" fill="' + regionF + '" stroke="' + regionS + '" stroke-width="1.5"/>'),
       'Instrumented Region', 'region']);
    if (inGraph('node[type="fusion"]'))
      nodeEntries.push([nodeIcon('<rect x="1" y="3" width="18" height="14" rx="2" fill="' + fusionF + '" stroke="' + fusionS + '" stroke-width="1.5"/>'),
       'Fusion Suggestion', 'fusion']);

    // ── Edge entries ───────────────────────────────────────────────────────
    var aliasClr = dark ? '#bb8fce' : '#8e44ad';
    var aliasIcon =
      '<svg viewBox="0 0 44 12" width="40" height="12">' +
      '<line x1="2" y1="6" x2="42" y2="6" stroke="' + aliasClr + '" stroke-width="1.5" stroke-dasharray="8 4"/>' +
      '</svg>';

    var edgeEntries = [];
    if (inGraph('edge[type="forward-dep"][access_mode="proxy"]'))
      edgeEntries.push([edgeIcon(''),      'Full',            'proxy']);
    if (inGraph('edge[access_mode="elwise"]'))
      edgeEntries.push([edgeIcon('5 3'),   'Elwise',          'elwise']);
    if (inGraph('edge[access_mode="scalar"]'))
      edgeEntries.push([edgeIcon('1.5 3'), 'Scalar',          'scalar']);
    if (inGraph('edge[type="anti-dep"]'))
      edgeEntries.push([edgeIcon('', 'red'), 'Anti',          'anti-dep']);
    if (inGraph('edge[type="virtual-alias"]'))
      edgeEntries.push([aliasIcon,           'Virtual Dataset','virtual-alias']);
    if (inGraph('edge[type="prng-dep"]')) {
      var prngIcon =
        '<svg viewBox="0 0 44 12" width="40" height="12">' +
        '<polyline points="2,6 9,3.6 16,8.4 23,3.6 30,8.4 37,6" stroke="' + edgeClr + '" stroke-width="1.5" fill="none"/>' +
        '<polygon points="44,6 32,2 32,10" fill="' + edgeClr + '"/>' +
        '</svg>';
      edgeEntries.push([prngIcon, 'PRNG', 'prng-dep']);
    }

    function item(icon, label, key) {
      var keyAttr = key ? ' data-legend-key="' + key + '"' : '';
      return '<div class="legend-item"' + keyAttr + '>' + icon +
             '<span style="color:' + textClr + '">' + label + '</span></div>';
    }

    function section(title, entries) {
      return '<div class="legend-section">' +
        '<div class="legend-section-title">' + title + '</div>' +
        '<div class="legend-items">' +
        entries.map(function(e) { return item(e[0], e[1], e[2]); }).join('') +
        '</div></div>';
    }

    var specSection =
      '<div class="legend-section legend-spectrum-section">' +
      '<div class="legend-section-title">Node color</div>' +
      '<div class="legend-spectrum-row">' +
      '<span class="legend-spec-end" style="color:' + textClr + '">min</span>' +
      '<span id="color-bar"></span>' +
      '<span class="legend-spec-end" style="color:' + textClr + '">max</span>' +
      '</div></div>';

    this._legendEl.innerHTML = specSection + section('Nodes', nodeEntries) + section('Dependences', edgeEntries);

    // Populate the freshly-inserted #color-bar gradient.
    this._updateColorBar();
  }

  /**
   * Highlight the legend entry whose data-legend-key equals `key`.
   * Clears any previously-highlighted entry first.  Pass null to just clear.
   */
  highlightKey(key) {
    if (!this._legendEl) return;
    this._legendEl.querySelectorAll('.legend-item.legend-item-hovered')
      .forEach(function(el) { el.classList.remove('legend-item-hovered'); });
    if (!key) return;
    var entry = this._legendEl.querySelector('[data-legend-key="' + key + '"]');
    if (entry) entry.classList.add('legend-item-hovered');
  }

  /** Remove the legend hover highlight from any currently-highlighted entry. */
  clearHighlight() {
    if (!this._legendEl) return;
    this._legendEl.querySelectorAll('.legend-item.legend-item-hovered')
      .forEach(function(el) { el.classList.remove('legend-item-hovered'); });
  }

  // Repaint only the #color-bar gradient swatch (no full legend rebuild).
  _updateColorBar() {
    var bar = this._legendEl && this._legendEl.querySelector('#color-bar');
    if (!bar) return;
    // Sample the active spectrum at 20 evenly-spaced points to produce a smooth
    // linear-gradient.  This avoids importing COLOR_SPECTRUMS directly and keeps
    // the gradient in sync with whatever spectrum the provider currently uses.
    var N    = 20;
    var pts  = [];
    for (var i = 0; i <= N; i++) {
      var t = i / N;
      pts.push(this._gv._spectrumColor(t) + ' ' + Math.round(t * 100) + '%');
    }
    bar.style.background = 'linear-gradient(to right,' + pts.join(',') + ')';
  }
}
