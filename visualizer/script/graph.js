// graph.js — Wires one TraceDataProvider + one GraphView into the main-page
//             singleton architecture (State, Gantt, legend).
//
// TraceDataProvider owns: fetch pipeline, colour spectrum, normalisation, and
// the cached data that all five views share.  GraphView owns the Cytoscape
// render.  This file bridges both to the legacy State/window-global surface so
// the rest of the app (ui.js, settings.js, …) continues to work unchanged while
// the view classes are progressively extracted.

'use strict';

import { State, BADGE_PLACEMENT_DELAY_MS, ANIM_SCROLL_MS } from './state.js';
import { fetchGraph, fetchGetData }                       from './transport.js';
import { TraceDataProvider }                            from './TraceDataProvider.js';
import { GraphView, LegendView }                        from './GraphView.js';
import { InfoView }                                     from './InfoView.js';
import { SourceCodeView }                               from './SourceCodeView.js';
import { TimelineView }                                 from './TimelineView.js';
import { SimulationEngine }                             from './simulation.js';

// ── Menubar render indicator ──────────────────────────────────────────────────
// Touches only #render-indicator in the menu bar.  GraphView manages its own
// graph-area overlay (#cy-layout-overlay) with snapshot transitions internally.

var _menubarIndicatorTimer = null;
var MENUBAR_INDICATOR_DELAY_MS = 150;  // short renders finish before this fires

function showMenubarIndicator() {
  clearTimeout(_menubarIndicatorTimer);
  _menubarIndicatorTimer = setTimeout(function() {
    _menubarIndicatorTimer = null;
    var ind = document.getElementById('render-indicator');
    if (ind) ind.classList.add('active');
  }, MENUBAR_INDICATOR_DELAY_MS);
}

function hideMenubarIndicator() {
  clearTimeout(_menubarIndicatorTimer);
  _menubarIndicatorTimer = null;
  var ind = document.getElementById('render-indicator');
  if (ind) ind.classList.remove('active');
}

// ── SourceCodeView timing constants ──────────────────────────────────────────
SourceCodeView.BADGE_PLACEMENT_DELAY_MS = BADGE_PLACEMENT_DELAY_MS;
SourceCodeView.ANIM_SCROLL_MS           = ANIM_SCROLL_MS;

// ── TraceDataProvider singleton ───────────────────────────────────────────────
// Owns the fetch pipeline, colour spectrum, and all cached response data.

var _provider = new TraceDataProvider({
  fetchGraph:    fetchGraph,
  fetchNodeData: fetchGetData,
  settings:      State.settings,      // shared reference — settings are always in sync
  colorSpectrum: State._colorSpectrum,
});

// Sync State data caches immediately when new data arrives so that Gantt,
// source pane, SimulationEngine, and the legend can read them.
// Rendering side-effects (badges, legend) fire from onRenderEnd
// (after Cytoscape is ready) since LegendView.render() calls hasElements().
// TimelineView renders directly from 'data' (no Cytoscape dependency).
_provider.on('data', function(rawData, minmax) {
  State._lastMinmax     = minmax;
  State._lastGraphNodes = rawData.nodes;
  State._lastNodeCount  = rawData.nodes.length;
  // allRegionPids is replaced (not mutated) by the provider each fetch.
  State._allRegionPids     = _provider.allRegionPids;
});

// ── GraphView singleton ───────────────────────────────────────────────────────
// _simEngine is declared here so the onRenderStart closure can reference it
// even though SimulationEngine is instantiated later (after all singletons exist).
var _simEngine = null;

var _view = new GraphView(document.getElementById('cy'), {

  provider: _provider,

  onRenderStart: function() {
    if (_simEngine) _simEngine.stop();
    // Show the menubar spinner only after a trace has been loaded — the initial
    // empty-state render (worker returns zero nodes immediately) should be silent.
    if (State._traceLoaded) showMenubarIndicator();
    // Dismiss any pending alert — settings were just reapplied.
    if (typeof clearAlert === 'function') clearAlert();
  },

  onRenderEnd: function(stats) {
    // State data caches were already updated in the provider 'data' listener above.
    // TimelineView renders from the 'data' event (no Cytoscape dependency).
    // Source-pane highlight clearing and badge placement are driven by the provider's
    // 'fetchstart' and 'renderend' events via _sourceView.attachProvider(_provider).
    _legendView.render(State._darkMode);

    // Hide the menubar spinner.
    hideMenubarIndicator();

    // Update the ⓘ info dropdown with counts and timings from this render.
    function _set(id, val) {
      var el = document.getElementById(id);
      if (el && val != null) el.textContent = val;
    }
    _set('event-count',    stats.eventCount);
    _set('node-count',     stats.nodeCount);
    _set('edge-count',     stats.edgeCount);
    _set('snapshot-count', stats.snapshotCount);
    // gantt-count is owned by TimelineView (countEl option) — it writes the
    // actual rendered bar count which may differ from timeline_nodes.length.
    _set('modeling-time',  stats.modelingMs);
    _set('rendering-time', stats.renderingMs);
    if (stats.modelingMs != null && stats.renderingMs != null)
      _set('response-time', stats.modelingMs + stats.renderingMs);
  },

  onError: function(err) {
    hideMenubarIndicator();
    if (typeof showAlert === 'function')
      showAlert('Graph render failed: ' + (err && err.message ? err.message : err));
  },

  onNodeHover: function(nodeId) {
    if (State.sourceView) State.sourceView.highlightBadgeNode(nodeId);
  },

  onNodeHoverEnd: function(nodeId) {
    if (State.sourceView) State.sourceView.clearBadgeHighlight();
  },

  onEdgeHover: function(fromId, toId) {
    if (State.sourceView) State.sourceView.highlightBadgeArrow(fromId, toId);
  },

  onEdgeHoverEnd: function(fromId, toId) {
    if (State.sourceView) State.sourceView.clearBadgeArrowHighlight();
  },

});

// ── Share mutable singleton state ─────────────────────────────────────────────
// settings: same object reference so ui.js checkbox/select changes are seen
// immediately by both GraphView's style functions and the provider's fetch params.
_view.settings = State.settings;

// Publish references so other modules can reach view/provider methods without
// a circular import through graph.js.
State.graphView = _view;
State.provider  = _provider;

// ── LegendView singleton ──────────────────────────────────────────────────────

var _legendView = new LegendView(document.getElementById('legend'), _view);
_legendView.settings = State.settings;  // same reference as _view.settings

// Wire the legend back into the graph view so hover events can highlight entries.
_view.legendView = _legendView;

// Expose to ui.js (syncSpectrumMenuIcons) and settings.js (window.renderLegend).
State.legendView = _legendView;
window.renderLegend = function() { _legendView.render(State._darkMode); };

// ── SourceCodeView singleton ──────────────────────────────────────────────────

var _sourceView = new SourceCodeView({
  initialTheme: State._hljsTheme,
  codePaths:    State._codePaths,

  onThemeChange: function(theme) {
    State._hljsTheme = theme;
    if (typeof saveSettings === 'function') saveSettings();
  },

});

State.sourceView = _sourceView;

// Attach the source view to the provider:
//   'fetchstart'  → clears line highlight + stale badges
//   'renderend'   → places fresh badges after layout completes
//   'select'      → highlights the source line when another view selects a node
_sourceView.attachProvider(_provider);

// ── Global exposure for inline onclick handlers ───────────────────────────────
// index.html menu buttons call setHljsTheme / _applyHljsTheme as window globals;
// worker-loader.js button onclicks call switchTab.
// Badge pills no longer use a global — clicks are handled via event delegation
// inside each SourceCodeView instance.
window.switchTab       = function(btn, idx)  { _sourceView.switchTab(btn, idx); };
window.setHljsTheme    = function(theme)     { _sourceView.setTheme(theme); };
window._applyHljsTheme = function(theme)     { _sourceView.restoreTheme(theme); };
// window.badgeClick is no longer needed: badge clicks are handled via event
// delegation on the tabPanesEl container inside each SourceCodeView instance.

// ── InfoView singleton ────────────────────────────────────────────────────────

var _infoView = new InfoView({
  provider: _provider,

  onOpaqueNodes: function(ids) {
    State.opaqueNodes = ids;
  },

  onHighlightNode: function(data) {
    if (State.sourceView) State.sourceView.highlightNode(data);
  },

  onSelectIteration: function(nodeId, iterIdx) {
    if (State.graphView) State.graphView.selectNode(nodeId, { zoom: false });
    if (State.timelineView) State.timelineView.scrollToInterval(nodeId, iterIdx);
  },

  onOpen: function() {
    if (typeof openInfoPane === 'function') openInfoPane();
  },

  onError: function(err) {
    if (typeof showAlert === 'function')
      showAlert('Could not load node info: ' + (err && err.message ? err.message : err));
  },
});

State.infoView = _infoView;

// ── TimelineView singleton ────────────────────────────────────────────────────

var _timelineView = new TimelineView(
  document.getElementById('gantt-svg-container'), {
    paneEl:          document.getElementById('gantt-pane'),
    scaleSliderEl:   document.getElementById('gantt-scale'),
    scaleLabelEl:    document.getElementById('gantt-scale-label'),
    countEl:         document.getElementById('gantt-count'),
    initialScale:    State._ganttScale,
    initialOpen:     State._ganttPaneOpen,
    onPaneToggle:    function() { if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons(); },
    onSettingsChange: function() { if (typeof saveSettings === 'function') saveSettings(); },
    onSimBarClick:   function(nodeId, iterIdx) {
      return !!(_simEngine && _simEngine.ganttClick(nodeId, iterIdx));
    },
    getSimDragging:  function() { return !!(_simEngine && _simEngine.isDragging); },
    animScrollMs:    ANIM_SCROLL_MS,
  }
);

State.timelineView = _timelineView;

// Attach to the provider so the chart renders as soon as data arrives
// (before Cytoscape layout — the Gantt has no Cytoscape dependency).
_timelineView.attachProvider(_provider);

// Provider 'select' — any view (GraphView, TimelineView, SourceCodeView) emits
// this when the user clicks a node, bar, or badge.  Each view has already
// registered its own listener to update its visual state (skipping its own
// source to prevent echo-back).  graph.js only handles the concerns that live
// outside the individual view classes: the InfoView and State tracking.
_provider.on('select', function(ev) {
  State._ganttSelectedNodeId = ev.nodeId;
  // traceIdx is the raw trace-file index used by InfoView for line highlighting.
  // It is distinct from iterIdx (0-based iteration) and is only present in
  // events emitted by simulation.js.  Fall back to iterIdx for all other sources.
  var infoIdx = ev.traceIdx != null ? ev.traceIdx : ev.iterIdx;
  if (State.infoView) State.infoView.open(ev.nodeId, ev.source, infoIdx);
});

// Provider 'clear' — emitted by GraphView on background click.  Each view
// clears its own visual selection via its own listener (registered in
// attachProvider).  graph.js handles the State-level cleanup that lives
// outside the view classes.
_provider.on('clear', function() {
  State.opaqueNodes          = [];   // setter also calls view.setOpaqueNodes([])
  State._ganttSelectedNodeId = null;
  if (typeof closePopup === 'function') closePopup();
});

// ── Forward State._ganttPaneOpen writes to TimelineView ──────────────────────
// settings.js loadSettings() writes State._ganttPaneOpen directly (before calling
// openGanttPane / closeGanttPane) so TimelineView._paneOpen must be kept in sync
// to avoid rendering into a visually-closed pane.
var _ganttPaneOpenStore = State._ganttPaneOpen;
Object.defineProperty(State, '_ganttPaneOpen', {
  get: function()    { return _ganttPaneOpenStore; },
  set: function(val) {
    _ganttPaneOpenStore = val;
    _timelineView._paneOpen = val;
    // When the pane is being closed by a direct State write (not via close()),
    // collapse it visually immediately — open/close calls handle this themselves.
    if (!val && _timelineView._paneEl)
      _timelineView._paneEl.style.flex = '0 0 0';
  },
  configurable: true,
  enumerable:   true,
});

// ── Forward State._ganttScale writes to TimelineView ─────────────────────────
// settings.js loadSettings() sets State._ganttScale + updates DOM sliders directly
// without going through ganttScaleChanged.  Sync the internal scale so the next
// render uses the correct value.
var _ganttScaleStore = State._ganttScale;
Object.defineProperty(State, '_ganttScale', {
  get: function()    { return _ganttScaleStore; },
  set: function(val) {
    _ganttScaleStore = parseFloat(val);
    _timelineView._scale = _ganttScaleStore;
  },
  configurable: true,
  enumerable:   true,
});

// ── Forward State._darkMode writes to the view ───────────────────────────────
// ui.js toggleDarkMode() and settings.js applyStoredLayout() write State._darkMode.
var _darkModeStore = State._darkMode;
Object.defineProperty(State, '_darkMode', {
  get: function()    { return _darkModeStore; },
  set: function(val) {
    _darkModeStore = val;
    _view.darkMode = val;
    // TimelineView re-renders so dark-mode colour palette updates immediately.
    if (_timelineView._paneOpen) _timelineView.reRenderPreservingScroll();
  },
  configurable: true,
  enumerable:   true,
});

// ── Forward State._colorSpectrum writes to provider + view ───────────────────
// ui.js colorSpectrumChanged() sets State._colorSpectrum = name then calls settingChanged() which
// debounces into a full re-render.  The provider setter updates the data-layer
// spectrum (used in next spectrumColor() call); the view setter triggers an
// immediate Cytoscape style update so the graph recolours before the re-render.
var _colorSpectrumStore = State._colorSpectrum;
Object.defineProperty(State, '_colorSpectrum', {
  get: function()    { return _colorSpectrumStore; },
  set: function(val) {
    _colorSpectrumStore      = val;
    _provider.colorSpectrum  = val;   // data layer — read by spectrumColor()
    _view.setColorSpectrum(val);      // triggers cy.style().update()
  },
  configurable: true,
  enumerable:   true,
});

// ── Forward State.opaqueNodes writes to the view ─────────────────────────────
// InfoView sets State.opaqueNodes = [...] via the onOpaqueNodes callback.
var _opaqueNodesStore = State.opaqueNodes;
Object.defineProperty(State, 'opaqueNodes', {
  get: function()    { return _opaqueNodesStore; },
  set: function(val) { _opaqueNodesStore = val; _view.setOpaqueNodes(val); },
  configurable: true,
  enumerable:   true,
});

// ── SimulationEngine singleton ────────────────────────────────────────────────

_simEngine = new SimulationEngine(_provider, {
  progressEl:  document.getElementById('sim-progress'),
  fillEl:      document.getElementById('sim-progress-fill'),
  handleEl:    document.getElementById('sim-progress-handle'),
  ticksEl:     document.getElementById('sim-ticks'),
  playPauseEl: document.getElementById('sim-btn-playpause'),
});
State.simEngine = _simEngine;

// ── Global exposure ───────────────────────────────────────────────────────────
// HTML inline handlers and worker-loader.js reach these as window globals.
// State.graphView / State.provider / State.simEngine are the canonical references.
window.fetchViewDataAndRender = function() { closePopup(); _provider.fetch(); };
window.renderImage            = function() { _view.exportImage(); };
window.zoomToFit              = function(id)  { _view.zoomToFit(id); };
window.zoomToFitGroup         = function(ids) { _view.zoomToFitGroup(ids); };

// stopRendering: abort the fetch, cancel the layout, hide the indicator.
window.stopRendering = function() { hideMenubarIndicator(); _view.stop(); };

// Simulation controls for HTML inline onclick handlers.
window.simStart  = function() { _simEngine.start(); };
window.simPause  = function() { _simEngine.pause(); };
window.simStop   = function() { _simEngine.stop(); };
window.simToggle = function() { _simEngine.toggle(); };
