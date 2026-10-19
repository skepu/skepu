// TraceDataProvider.js — Data coordinator for the SkePU visualizer.
//
// Owns the fetch pipeline, min/max normalisation, colour spectrum, and the
// cached response data that all five views (graph, Gantt, source code, info
// pane, stats popup) need to display.  Acts as the single communication hub:
// each view registers listeners once, and the provider drives them on every
// new trace render.
//
// ── Typical use ───────────────────────────────────────────────────────────────
//
//   import { TraceDataProvider } from './TraceDataProvider.js';
//
//   const provider = new TraceDataProvider({
//       fetchGraph:    params => transport.fetchGraph(params),
//       settings:      sharedSettingsObject,
//       colorSpectrum: 'viridis',
//   });
//
//   provider.on('fetchstart', ()              => showSpinner());
//   provider.on('data',       (raw, minmax)   => updateViews(raw, minmax));
//   provider.on('error',      err             => showAlert(err.message));
//
//   provider.fetch();   // kick off the first load
//
// ── Events ─────────────────────────────────────────────────────────────────────
//
//   'fetchstart'            Emitted before the HTTP/worker request is sent.
//                           No arguments.
//
//   'data'  (raw, minmax)   Emitted when a fetch completes successfully.
//                           raw    — the raw server response (nodes, edges, …)
//                           minmax — {mode → {min, max}} normalisation ranges
//                                    computed from this response
//
//   'renderend' (stats)     Emitted by a GraphView after its Cytoscape layout
//                           finishes and the first frame is painted.  stats is
//                           the same object passed to GraphView's onRenderEnd
//                           callback: { nodeCount, edgeCount, badgeNodes, … }.
//                           Only fired when a GraphView is attached in provider
//                           mode (provider option).
//
//   'select' ({nodeId, source, focusId, allIds, iterIdx, traceIdx})
//                           Emitted by any view when the user clicks a node,
//                           a Gantt bar, or a source-code badge.
//                           nodeId   — the primary node being selected
//                           source   — 'graph' | 'gantt' | 'code-listing' | 'simulation' | …
//                           focusId  — node to zoom/scroll to (defaults to nodeId)
//                           allIds   — all node IDs to highlight (defaults to [nodeId])
//                           iterIdx  — 0-based iteration index (optional; drives
//                                      TimelineView.scrollToInterval when > 0)
//                           traceIdx — raw trace-file index for InfoView highlight
//                                      (optional; falls back to iterIdx when absent)
//                           Each view ignores events where source matches its
//                           own identity, preventing echo-back to the originator.
//
//   'rendersettled' (stats) Emitted by a GraphView when the layout and the
//                           size of the graph have settled after a render.
//
//   'clear' ({background})  Emitted by GraphView on a click that selects
//                           nothing; `background` is true when the canvas
//                           itself was clicked (that also stops a simulation).
//                           Each attached view clears its selection.
//
//   'error' (err)           Emitted on non-abort failures.  AbortErrors from
//                           cancelled requests are silently dropped.

'use strict';

// ── Color spectrums ────────────────────────────────────────────────────────────
// Canonical definition.  LegendView (LegendView.js) uses the provider's
// spectrumColor / spectrumColorDark methods rather than importing these directly.

// A rainbow, the colours of visible light from violet (low) to red (high), at eight hues that are the same distance apart (40°): what a
// backend or a pattern is coloured by has colours that differ clearly (see categoryColor).
function hsl(h, s, l) {
  var k = function(n) { return (n + h / 30) % 12; };
  var a = s * Math.min(l, 1 - l);
  var f = function(n) { return Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1)))); };
  return [f(0), f(8), f(4)];
}
var RAINBOW = [280, 240, 200, 160, 120, 80, 40, 0].map(function(h) { return hsl(h, 0.72, 0.56); });

export const COLOR_SPECTRUMS = {
  'default':   { stops: [[0,255,0],[255,0,0]] },
  'viridis':   { stops: [[68,1,84],[72,40,120],[62,83,160],[42,120,142],
                          [34,168,132],[122,209,81],[189,223,38],[253,231,37]] },
  'plasma':    { stops: [[13,8,135],[83,2,163],[139,10,165],[184,50,137],
                          [219,92,104],[244,136,73],[254,188,42],[240,249,33]] },
  'inferno':   { stops: [[0,0,4],[40,11,84],[101,21,110],[159,42,99],
                          [212,72,66],[245,125,21],[250,193,39],[252,255,164]] },
  'cividis':   { stops: [[0,34,78],[30,68,122],[66,97,141],[98,124,143],
                          [137,152,133],[177,177,94],[215,202,48],[253,231,37]] },
  'cool-warm': { stops: [[59,76,192],[221,221,221],[180,4,38]] },
  'grayscale': { stops: [[240,240,240],[20,20,20]] },
  'rainbow':   { stops: RAINBOW, categorical: true },
};

// The categories that a node can be coloured by (View > Nodes > Color: Backend, Pattern). Each has a fixed place on
// the colour spectrum that is chosen in the Colors menu, so that a backend or a pattern keeps its colour whatever
// else is in the trace.
export const BACKEND_ORDER = ['CPU', 'OpenMP', 'OpenCL', 'CUDA', 'Hybrid'];
export const PATTERN_ORDER = ['Map', 'MapReduce', 'Reduce', 'Scan', 'MapOverlap', 'MapPool', 'MapPairs', 'MapPairsReduce'];

/** The skeleton of a pattern name: 'MapOverlap 2D' -> 'MapOverlap', 'Reduce 1D-Full' -> 'Reduce'. */
export function basePattern(pattern) { return (pattern || '').split(' ')[0]; }

/** Linear interpolation between RGB colour stops at normalised position t ∈ [0,1]. */
export function lerpColor(stops, t) {
  var n = stops.length - 1;
  var i = Math.min(Math.floor(t * n), n - 1);
  var f = t * n - i;
  var a = stops[i], b = stops[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

// ── Minmax helper ─────────────────────────────────────────────────────────────

function _mm(minmax, key, value) {
  if (value == null || isNaN(value)) return;
  var e = minmax[key];
  if (!e) { minmax[key] = { min: value, max: value }; return; }
  if (value < e.min) e.min = value;
  if (value > e.max) e.max = value;
}

/**
 * Build the {mode → {min, max}} normalisation table from a raw /graph response.
 * Exported so GraphView can reuse it in standalone (no-provider) mode.
 *
 * @param {Array} nodes  raw node objects from server response
 * @param {Array} edges  raw edge objects from server response
 * @returns {object}     minmax table
 */
export function buildMinmax(nodes, edges) {
  var minmax = {};
  nodes.forEach(function(node) {
    if (node.data.type !== 'skeleton_call' && node.data.type !== 'external') return;
    var durs = (node.data.durations && node.data.durations.length)
      ? node.data.durations : [node.data.duration];
    durs.forEach(function(d) { _mm(minmax, 'duration-total', d); });
    _mm(minmax, 'dag-depth',     node.data.dag_depth);
    _mm(minmax, 'total-order',   node.data.total_order);
    _mm(minmax, 'iterations',    node.data.iteration_count || 1);
    _mm(minmax, 'nesting-level', node.data.nesting_level   || 0);
    _mm(minmax, 'fan-in',        node.data.fan_in          || 0);
    _mm(minmax, 'fan-out',       node.data.fan_out         || 0);
    _mm(minmax, 'duration-cv',   node.data.duration_cv     || 0);
    if (node.data.imbalance != null) _mm(minmax, 'imbalance', node.data.imbalance);
    if (node.data.comm_bytes != null) _mm(minmax, 'comm-bytes', node.data.comm_bytes);
    if (node.data.type === 'skeleton_call' && node.data.elements) {
      var el = node.data.elements.reduce(function(a, b) { return a * b; });
      _mm(minmax, 'elements', el);
      if (node.data.duration) _mm(minmax, 'duration-per-element', node.data.duration / el);
    }
  });
  edges.forEach(function(edge) {
    if (edge.data.order_span != null) _mm(minmax, 'order-span', edge.data.order_span);
  });
  return minmax;
}

// ── Normalisation sentinels ────────────────────────────────────────────────────

/** Returned in 'fixed' colour/size mode — near the low end of the spectrum. */
export const NORM_FIXED      = 0.05;
/** Returned when min === max (uniform data → mid-spectrum). */
export const NORM_DEGENERATE = 0.5;

// ── TraceDataProvider ─────────────────────────────────────────────────────────

export class TraceDataProvider {

  // ── Constructor ─────────────────────────────────────────────────────────────

  /**
   * @param {object}   options
   * @param {Function} options.fetchGraph      Required.  Async fn(params, signal) → raw data.
   * @param {object}   [options.settings]      Shared settings object (reference, not copy).
   * @param {string}   [options.colorSpectrum] Initial spectrum name (default 'default').
   */
  constructor(options) {
    options = options || {};
    if (!options.fetchGraph)
      throw new Error('TraceDataProvider: options.fetchGraph is required');

    this._fetchGraph    = options.fetchGraph;
    // Optional — required only when InfoView (or another node-detail consumer) is used.
    this._fetchNodeData = options.fetchNodeData || null;
    // Shared reference — both provider and views read the same dict.
    this.settings = options.settings || {};

    // Colour spectrum.
    this._colorSpectrum = options.colorSpectrum || 'default';

    // Fetch lifecycle.
    this._fetchAbort = null;
    this._fetchSeq   = 0;

    // First-load flag: enables auto-expand and other one-time behaviours.
    this._firstLoad = true;

    // Region collapse state — shared with GraphView via object identity.
    this._collapsedNodes = {};
    this._allRegionPids  = [];
    this._allFusionPids  = [];     // fusion nodes share the dict: true = collapsed; they start expanded
    this._regionParents  = {};     // id of a region or fusion node -> id of the one around it (or null)

    // Cached last response — available immediately after 'data' fires.
    this._lastMinmax        = null;
    this._lastGraphNodes    = [];
    this._lastTimelineNodes = [];
    this._lastNodeCount     = 0;

    // Event listener registry.
    this._listeners = { fetchstart: [], data: [], renderend: [], rendersettled: [], select: [], clear: [], error: [] };
  }

  // ── Event emitter ────────────────────────────────────────────────────────────

  /**
   * Register a listener for an event.
   * @param {'fetchstart'|'data'|'renderend'|'rendersettled'|'select'|'clear'|'error'} event   (an event that is not one of these is ignored)
   * @param {Function} handler
   * @returns {TraceDataProvider}  this (chainable)
   */
  on(event, handler) {
    if (this._listeners[event]) this._listeners[event].push(handler);
    return this;
  }

  /**
   * Remove a previously registered listener.
   * @returns {TraceDataProvider}  this (chainable)
   */
  off(event, handler) {
    if (this._listeners[event])
      this._listeners[event] = this._listeners[event].filter(function(h) { return h !== handler; });
    return this;
  }

  _emit(event, a, b) {
    var list = this._listeners[event];
    if (!list) return;
    for (var i = 0; i < list.length; i++) {
      try { list[i](a, b); } catch (e) {
        console.error('[TraceDataProvider] listener error in "' + event + '":', e);
      }
    }
  }

  // ── Fetch pipeline ────────────────────────────────────────────────────────────

  /**
   * Reset the first-load flag so the next fetch treats the current trace as
   * freshly loaded (enables auto-expand of small traces, etc.).
   * Call once after each new trace file is loaded into the backend.
   */
  resetFirstLoad() {
    this._firstLoad = true;
    // The collapse state is that of the previous trace: the regions of the new one are numbered from
    // region_1 again, so they would inherit it. Cleared in place: other modules hold this object.
    for (var pid of Object.keys(this._collapsedNodes)) delete this._collapsedNodes[pid];
    this._allRegionPids = [];
    this._allFusionPids = [];
    this._regionParents = {};
  }

  /**
   * Cancel any in-flight fetch without starting a new one.
   * Bumps the internal sequence so stale 'data' callbacks self-discard.
   */
  abort() {
    if (this._fetchAbort) {
      try { this._fetchAbort.abort(); } catch (e) {}
      this._fetchAbort = null;
    }
    this._fetchSeq++;
  }

  /**
   * Fetch graph data from the backend, compute normalisation ranges, update
   * the cached data, and emit 'data'.  Cancels any previously in-flight fetch.
   *
   * @returns {Promise<void>}  Resolves when listeners have been called, or
   *                           immediately on abort / cancellation.
   */
  fetch() {
    this.abort();                          // cancel any in-flight request
    var mySeq = this._fetchSeq;            // captured after abort() bumped it
    this._fetchAbort = new AbortController();

    this._emit('fetchstart');

    var isFusion = function(k) { return k.indexOf('fusion_') === 0; };
    var expandedPids = Object.keys(this._collapsedNodes)
      .filter(function(k) { return !isFusion(k) && this._collapsedNodes[k] === false; }, this);
    var collapsedFusions = Object.keys(this._collapsedNodes)
      .filter(function(k) { return isFusion(k) && this._collapsedNodes[k] === true; }, this);

    var self = this;
    return this._fetchGraph({
      mode:                    this.settings['graph-mode'],
      container_allocations:   this.settings['container-allocations'],
      container_deallocations: this.settings['container-deallocations'],
      container_snapshots:     this.settings['container-snapshots'],
      container_transfers:     this.settings['container-transfers'],
      kernel_compiles:         this.settings['kernel-compiles'],
      show_communications:     this.settings['communications'],
      show_scalars:            this.settings['element-accesses'],
      anti_deps:               this.settings['anti-deps'],
      data_as_edges:           !this.settings['container-updates'],
      show_regions:            true,
      expand_all_regions:      !this.settings['show_regions'],
      first_load:              this._firstLoad,
      collapse_iteration:      this.settings['collapse_iteration'],
      aggregate_by:            this.settings['aggregate_by'],
      fusion_analysis:         this.settings['fusion_analysis'],
      show_virtual_alias:      this.settings['virtual-alias-edges'],
      expanded_regions:        expandedPids.join(','),
      collapsed_fusions:       collapsedFusions.join(','),
    }, this._fetchAbort.signal)
    .then(function(rawData) {
      if (mySeq !== self._fetchSeq) return;   // superseded by a newer fetch
      self._fetchAbort = null;

      // ── Update region collapse state ────────────────────────────────────────
      var allRegionPids = rawData.all_region_pids || [];
      self._allRegionPids = allRegionPids;
      var defaultCollapsed = !rawData.auto_expand_regions;
      allRegionPids.forEach(function(pid) {
        if (!(pid in self._collapsedNodes)) self._collapsedNodes[pid] = defaultCollapsed;
      });
      self._regionParents = rawData.region_parents || {};
      var allFusionPids = rawData.all_fusion_pids || [];
      self._allFusionPids = allFusionPids;
      allFusionPids.forEach(function(pid) {
        if (!(pid in self._collapsedNodes)) self._collapsedNodes[pid] = false;
      });
      self._firstLoad = false;

      // ── Build normalisation ranges ──────────────────────────────────────────
      var minmax = buildMinmax(rawData.nodes, rawData.edges);

      // ── Cache ───────────────────────────────────────────────────────────────
      self._lastMinmax        = minmax;
      self._lastGraphNodes    = rawData.nodes;
      self._lastTimelineNodes = rawData.timeline_nodes || [];
      self._lastNodeCount     = rawData.nodes.length;

      // ── Notify all registered views ─────────────────────────────────────────
      self._emit('data', rawData, minmax);
    })
    .catch(function(err) {
      if (err && err.name === 'AbortError') return;
      if (mySeq === self._fetchSeq) self._emit('error', err);
    });
  }

  // ── Cached data accessors ─────────────────────────────────────────────────────
  // Views read these after 'data' fires; safe to read at any time (start as empty).

  /** Raw node array from the last successful fetch. */
  get lastGraphNodes()    { return this._lastGraphNodes;    }
  /** Timeline node array for the Gantt chart from the last fetch. */
  get lastTimelineNodes() { return this._lastTimelineNodes; }
  /** {mode → {min, max}} normalisation ranges from the last fetch. */
  get lastMinmax()        { return this._lastMinmax;        }
  /** All persistent region IDs from the last fetch (for collapse-all UI). */
  get allRegionPids()     { return this._allRegionPids;     }
  /** All persistent fusion ids from the last fetch (for collapse-all UI). */
  get allFusionPids()     { return this._allFusionPids;     }
  /** Nesting of the regions and fusion nodes of the last fetch: id -> id around it, or null. */
  get regionParents()     { return this._regionParents;     }
  /**
   * Mutable dict: persistent_region_id → true (collapsed) / false (expanded).
   * Shared by reference with GraphView so double-click expand/collapse writes
   * are seen by the next fetch without any explicit sync step.
   */
  get collapsedNodes()    { return this._collapsedNodes;    }
  /** Total node count from the last fetch (used for auto-render threshold). */
  get lastNodeCount()     { return this._lastNodeCount;     }

  // ── Node-detail fetch ─────────────────────────────────────────────────────────

  /**
   * Fetch detailed data for a single node (duration plot, snapshots, source
   * location, live-set, …).  Requires options.fetchNodeData to be provided.
   *
   * @param {string} nodeId  UUID of the graph node.
   * @returns {Promise<object>}  Resolved with the node-info data object.
   */
  fetchNodeData(nodeId) {
    if (!this._fetchNodeData)
      return Promise.reject(new Error('TraceDataProvider: no fetchNodeData function provided'));
    return this._fetchNodeData(nodeId);
  }

  // ── Colour spectrum ───────────────────────────────────────────────────────────

  get colorSpectrum() { return this._colorSpectrum; }
  set colorSpectrum(name) { this._colorSpectrum = name; }

  /**
   * Map normalised t ∈ [0, 1] to a CSS rgb() string using the active spectrum.
   */
  spectrumColor(t) {
    var stops = (COLOR_SPECTRUMS[this._colorSpectrum] || COLOR_SPECTRUMS['default']).stops;
    var c = lerpColor(stops, Math.max(0, Math.min(1, t)));
    return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
  }

  /**
   * The colour of a category (a backend, a pattern) on the active spectrum, or null if it is not one of `order`.
   * `dark` is the darkened colour that the borders have.
   */
  categoryColor(order, key, dark) {
    var i = order.indexOf(key);
    if (i < 0) return null;
    var t = order.length > 1 ? i / (order.length - 1) : 0;
    // a spectrum of distinct hues (the rainbow): the categories are on the hues themselves, as far apart as there are hues, not between two
    var spectrum = COLOR_SPECTRUMS[this._colorSpectrum] || COLOR_SPECTRUMS['default'];
    if (spectrum.categorical && order.length > 1 && order.length <= spectrum.stops.length) {
      var last = spectrum.stops.length - 1;
      t = Math.round(i * last / (order.length - 1)) / last;
    }
    return dark ? this.spectrumColorDark(t) : this.spectrumColor(t);
  }

  /** The colour of a backend ('CPU', 'OpenCL', ...) on the active spectrum; null for another name. */
  backendColor(backend, dark) { return this.categoryColor(BACKEND_ORDER, backend, dark); }

  /** The colour of a skeleton pattern ('Map', 'MapOverlap 2D', ...) on the active spectrum; null for another name. */
  patternColor(pattern, dark) { return this.categoryColor(PATTERN_ORDER, basePattern(pattern), dark); }

  /**
   * Like spectrumColor but darkened by `factor` (default 0.72) — used for
   * node border colours to add depth.
   */
  spectrumColorDark(t, factor) {
    factor = (factor !== undefined) ? factor : 0.72;
    var stops = (COLOR_SPECTRUMS[this._colorSpectrum] || COLOR_SPECTRUMS['default']).stops;
    var c = lerpColor(stops, Math.max(0, Math.min(1, t)));
    return 'rgb(' + Math.round(c[0]*factor) + ',' + Math.round(c[1]*factor) + ',' + Math.round(c[2]*factor) + ')';
  }

  // ── Normalisation ─────────────────────────────────────────────────────────────
  //
  // Three variants for the three call sites:
  //   normValue     — takes a pre-resolved numeric value (lowest level)
  //   normFromData  — extracts the value from a plain node-data object
  //   normFromEle   — extracts the value from a Cytoscape element via ele.data()
  //
  // All three read this.settings[settingsKey] to determine which property to use.
  // An optional minmax override is accepted; falls back to this.lastMinmax.

  /**
   * Normalise a raw numeric `value` for the given `mode` to [0, 1].
   *
   * @param {string}       mode    e.g. 'duration-total', 'dag-depth', 'fixed', …
   * @param {number|null}  value
   * @param {object}       [minmax]  override; defaults to this.lastMinmax
   * @returns {number}  value in [0, 1]
   */
  normValue(mode, value, minmax) {
    if (mode === 'fixed') return NORM_FIXED;
    if (value == null)    return NORM_DEGENERATE;
    var mm = ((minmax || this._lastMinmax) || {})[mode];
    if (!mm || mm.max === mm.min) return NORM_DEGENERATE;
    return Math.max(0, Math.min(1, (value - mm.min) / (mm.max - mm.min)));
  }

  /**
   * Normalise a property from a plain node-data object `d` using settings key `key`.
   * Used by the Gantt chart renderer (which operates on raw data, not cy elements).
   *
   * @param {string} key     settings key, e.g. 'node-color-call'
   * @param {object} d       plain node data object (from rawData.nodes[i].data)
   * @param {object} [minmax]
   * @returns {number}
   */
  normFromData(key, d, minmax) {
    var mode = this.settings[key];
    if (mode === 'fixed') return NORM_FIXED;
    var value;
    if      (mode === 'duration-total')      value = d.duration || 0;
    else if (mode === 'dag-depth')           value = d.dag_depth || 0;
    else if (mode === 'total-order')         value = d.total_order || 0;
    else if (mode === 'iterations')          value = d.iteration_count || 1;
    else if (mode === 'nesting-level')       value = d.nesting_level || 0;
    else if (mode === 'fan-in')              value = d.fan_in  || 0;
    else if (mode === 'fan-out')             value = d.fan_out || 0;
    else if (mode === 'duration-cv')         value = d.duration_cv || 0;
    else if (mode === 'imbalance')           value = d.imbalance;      // (null for a call that is not of an MPI program: the default colour)
    else if (mode === 'comm-bytes')          value = d.comm_bytes;
    else if (mode === 'elements') {
      value = (d.elements || [1]).reduce(function(a, b) { return a * b; }, 1);
    } else if (mode === 'duration-per-element') {
      var el = (d.elements || [1]).reduce(function(a, b) { return a * b; }, 1);
      value = (d.duration || 0) / (el || 1);
    } else {
      return NORM_DEGENERATE;
    }
    return this.normValue(mode, value, minmax);
  }

  /**
   * Normalise a property from a Cytoscape element `ele` using settings key `key`.
   * Used by GraphView's Cytoscape style functions.
   *
   * @param {string} key     settings key, e.g. 'node-color-call'
   * @param {object} ele     Cytoscape element (supports ele.data(field))
   * @param {object} [minmax]
   * @returns {number}
   */
  normFromEle(key, ele, minmax) {
    var mode = this.settings[key];
    if (mode === 'fixed') return NORM_FIXED;
    var value;
    if      (mode === 'duration-total')      value = ele.data('duration');
    else if (mode === 'dag-depth')           value = ele.data('dag_depth');
    else if (mode === 'total-order')         value = ele.data('total_order');
    else if (mode === 'iterations')          value = ele.data('iteration_count') || 1;
    else if (mode === 'nesting-level')       value = ele.data('nesting_level')   || 0;
    else if (mode === 'fan-in')              value = ele.data('fan_in')          || 0;
    else if (mode === 'fan-out')             value = ele.data('fan_out')         || 0;
    else if (mode === 'duration-cv')         value = ele.data('duration_cv')     || 0;
    else if (mode === 'imbalance')           value = ele.data('imbalance');
    else if (mode === 'comm-bytes')          value = ele.data('comm_bytes');
    else if (mode === 'elements') {
      var els = ele.data('elements');
      if (!els) return NORM_DEGENERATE;
      value = els.reduce(function(a, b) { return a * b; });
    } else if (mode === 'duration-per-element') {
      var els = ele.data('elements');
      if (!els) return NORM_DEGENERATE;
      value = ele.data('duration') / els.reduce(function(a, b) { return a * b; });
    } else {
      return NORM_DEGENERATE;
    }
    return this.normValue(mode, value, minmax);
  }
}
