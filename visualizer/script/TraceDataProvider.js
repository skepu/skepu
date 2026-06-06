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
//   'clear'                 Emitted by GraphView when the user clicks empty
//                           canvas space (background click).  Each attached
//                           view clears its own visual selection state when it
//                           receives this event.
//
//   'error' (err)           Emitted on non-abort failures.  AbortErrors from
//                           cancelled requests are silently dropped.

'use strict';

// ── Color spectrums ────────────────────────────────────────────────────────────
// Canonical definition.  LegendView (GraphView.js) uses the provider's
// spectrumColor / spectrumColorDark methods rather than importing these directly.

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
};

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

    // Cached last response — available immediately after 'data' fires.
    this._lastMinmax        = null;
    this._lastGraphNodes    = [];
    this._lastTimelineNodes = [];
    this._lastNodeCount     = 0;

    // Event listener registry.
    this._listeners = { fetchstart: [], data: [], renderend: [], select: [], clear: [], error: [] };
  }

  // ── Event emitter ────────────────────────────────────────────────────────────

  /**
   * Register a listener for an event.
   * @param {'fetchstart'|'data'|'renderend'|'select'|'clear'|'error'} event
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
  resetFirstLoad() { this._firstLoad = true; }

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

    var expandedPids = Object.keys(this._collapsedNodes)
      .filter(function(k) { return this._collapsedNodes[k] === false; }, this);

    var self = this;
    return this._fetchGraph({
      mode:                    this.settings['graph-mode'],
      container_allocations:   this.settings['container-allocations'],
      container_deallocations: this.settings['container-deallocations'],
      container_transfers:     this.settings['container-transfers'],
      show_scalars:            this.settings['element-accesses'],
      anti_deps:               this.settings['anti-deps'],
      data_as_edges:           !this.settings['container-updates'],
      show_regions:            true,
      expand_all_regions:      !this.settings['show_regions'],
      first_load:              this._firstLoad,
      collapse_iteration:      this.settings['collapse_iteration'],
      fusion_analysis:         this.settings['fusion_analysis'],
      show_virtual_alias:      this.settings['virtual-alias-edges'],
      expanded_regions:        expandedPids.join(','),
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
