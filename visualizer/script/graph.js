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
import { GraphView }                                  from './GraphView.js';
import { LegendView }                                 from './LegendView.js';
import { showSpectrumMenu, showChoiceMenu }                           from './appearance.js';
import { InfoView }                                     from './InfoView.js';
import { SourceCodeView }                               from './SourceCodeView.js';
import { TimelineView }                                 from './TimelineView.js';
import { SimulationEngine }                             from './simulation.js';
import { showDarkMode } from './appearance.js';

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

// ── TraceDataProvider singleton ───────────────────────────────────────────────
// Owns the fetch pipeline, colour spectrum, and all cached response data.

var _provider = new TraceDataProvider({
  fetchGraph:    fetchGraph,
  fetchNodeData: fetchGetData,
  settings:      State.settings,      // shared reference — settings are always in sync
  colorSpectrum: State.settings.get('color-spectrum'),
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
  State._allFusionPids     = _provider.allFusionPids;
  State._regionParents     = _provider.regionParents;
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
    _legendView.render(State.settings.get('dark-mode'));

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

// The collapse state is the provider's: the Actions menu (ui.js) writes it through State, the
// double click on a region (GraphView) and the next fetch read it through the provider. They must be
// one object.
State._collapsedNodes = _provider.collapsedNodes;

// ── LegendView singleton ──────────────────────────────────────────────────────

var _legendView = new LegendView(document.getElementById('legend'), _view, { hoverGraph: function() { return _view._cy; },
  clearHighlight: function() { _view._provider._emit('clear', { background: false }); },
  onSelect: function(callback) { _view._provider.on('select', callback); }, onColorByClick: function(x, y) { showChoiceMenu(x, y, document.querySelectorAll('[data-ctrl="node-color-call"]'), function(b) { return State.settings.get('node-color-call') === b.dataset.value; }); }, onSpectrumClick: function(x, y) { showSpectrumMenu(x, y, State.settings.get('color-spectrum'), function(name) { window.colorSpectrumChanged(name); }); } });
_legendView.settings = State.settings;  // same reference as _view.settings

// Wire the legend back into the graph view so hover events can highlight entries.
_view.legendView = _legendView;

// Expose to ui.js (syncSpectrumMenuIcons) and settings.js (window.renderLegend).
State.legendView = _legendView;
window.renderLegend = function() { _legendView.render(State.settings.get('dark-mode')); };

// ── SourceCodeView singleton ──────────────────────────────────────────────────

var _sourceView = new SourceCodeView({
  initialTheme: State.settings.get('hljs-theme'),
  codePaths:    State.settings.get('code-paths'),
  pinDefinitions: State.settings.get('pin-definitions'),

  onThemeChange: function(theme) {
    State.settings.set('hljs-theme', theme);
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
export function switchTab(btn, idx) {
  _sourceView.switchTab(btn, idx);
}

export function setHljsTheme(theme) {
  _sourceView.setTheme(theme); 
}

window.switchTab       = switchTab;
window.setHljsTheme    = setHljsTheme;
window._applyHljsTheme = function(theme)     { _sourceView.restoreTheme(theme); };

// ── InfoView singleton ────────────────────────────────────────────────────────

var _infoView = new InfoView({
  provider: _provider,

  onOpaqueNodes: function(ids) {
    State.opaqueNodes = ids;
  },

  onSelectIteration: function(nodeId, iterIdx) {
    if (State.graphView) State.graphView.selectNode(nodeId, { zoom: false });
    if (State.timelineView) State.timelineView.scrollToInterval(nodeId, iterIdx);
  },

  onOpen: function() {
    if (typeof openInfoPane === 'function') openInfoPane();
  },

  onShowSource: function(place) {
    if (State.sourceView) State.sourceView.highlightSourceLine(place.line, place.file, undefined, place.end_line);
  },

  onDefinitions: function(definitions, call) {
    if (State.sourceView) State.sourceView.showDefinitions(definitions, call);
  },

  canShowSource: function(file) {
    return !!State.sourceView && State.sourceView.hasSourceFile(file);
  },

  onError: function(err) {
    if (typeof showAlert === 'function')
      showAlert('Could not load node info: ' + (err && err.message ? err.message : err));
  },
});

State.infoView = _infoView;

export function setInfoViewOnHighlightNode(func) {
  _infoView._onHighlightNode = func;
}

/** Which source files the links of the info pane can show (the editor: the ones that it has found). */
export function setInfoViewCanShowSource(func) {
  _infoView._canShowSource = func;
}

/** Sets the handler that marks the places of a call (instance, user functions, the call itself) in the editor. */
export function setInfoViewOnDefinitions(func) {
  _infoView._onDefinitions = func;
}

/** The editor shows the lines of a definition in its own file: what the links of the info pane do. */
export function setInfoViewOnShowSource(func) {
  _infoView._onShowSource = func;
}

// ── TimelineView singleton ────────────────────────────────────────────────────

var _timelineView = new TimelineView(
  document.getElementById('gantt-svg-container'), {
    paneEl:          document.getElementById('gantt-pane'),
    scaleSliderEl:   document.getElementById('gantt-scale'),
    scaleLabelEl:    document.getElementById('gantt-scale-label'),
    countEl:         document.getElementById('gantt-count'),
    initialScale:    State.settings.get('gantt-scale'),
    initialOpen:     State.settings.get('gantt-open'),
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
  State._selectedRank = ev.rank != null ? ev.rank : null;
  // traceIdx is the raw trace-file index used by InfoView for line highlighting.
  // It is distinct from iterIdx (0-based iteration) and is only present in
  // events emitted by simulation.js. (An iteration number is not a trace index.)
  var infoIdx = ev.traceIdx;
  if (State.infoView) State.infoView.open(ev.nodeId, ev.source, infoIdx, ev.reveal);
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

// ── The views follow the settings ─────────────────────────────────────────────
// A setting change reaches the views whoever makes it (menu, loadSettings, Restore Defaults).
State.settings.onChange(function(keys) {
  var get = function(key) { return State.settings.get(key); };
  if (keys.includes('gantt-open')) {
    _timelineView._paneOpen = get('gantt-open');
    // closed by the setting alone, not by close(): collapsed at once
    if (!get('gantt-open') && _timelineView._paneEl) _timelineView._paneEl.style.setProperty('--gantt-height', 0);
  }
  if (keys.includes('gantt-scale')) _timelineView._scale = parseFloat(get('gantt-scale'));
  if (keys.includes('dark-mode')) {
    showDarkMode(get('dark-mode'));   // the timeline reads the class
    _view.darkMode = get('dark-mode');
    _sourceView.refreshSelectionColors();   // the accent changed
    if (_timelineView._paneOpen) _timelineView.reRenderPreservingScroll();
  }
  if (keys.includes('color-spectrum')) {
    _provider.colorSpectrum = get('color-spectrum');   // read by spectrumColor()
    _view.setColorSpectrum(get('color-spectrum'));     // restyles now; the re-render follows
  }
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
  // the progress bar is shown when a simulation starts: the page has the setting and the menu item
  onActivate:  function() { if (!State.settings.get('progress-bar') && typeof window.setProgressBar === 'function') window.setProgressBar(true); },
});
State.simEngine = _simEngine;

export function renderImage() {
  _view.exportImage('graph.png');
}

export function stopRendering() {
  hideMenubarIndicator();
  _view.stop(); 
}

// ── Global exposure ───────────────────────────────────────────────────────────
// HTML inline handlers and worker-loader.js reach these as window globals.
// State.graphView / State.provider / State.simEngine are the canonical references.
window.fetchViewDataAndRender = function() { closePopup(); _provider.fetch(); };
window.renderImage            = renderImage;
window.zoomToFit              = function(id)  { _view.zoomToFit(id); };
window.zoomToFitGroup         = function(ids) { _view.zoomToFitGroup(ids); };

// stopRendering: abort the fetch, cancel the layout, hide the indicator.
window.stopRendering = stopRendering;

// Simulation controls for HTML inline onclick handlers.
export function simToggle() {
  _simEngine.toggle();
}

window.simStart  = function() { _simEngine.start(); };
window.simPause  = function() { _simEngine.pause(); };
window.simToggle = simToggle;
