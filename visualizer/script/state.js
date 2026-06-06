// state.js — Shared mutable state and constants, exported as an ES module.
// All modules import { State } from here; HTML inline scripts access window.State.

export const State = {
  // Cytoscape instance — set by graph.js after each render.
  cy:               null,
  daglayout:        null,

  // Render settings — source of truth for all checkbox/select settings.
  settings: {
    // checkboxes
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
    // selects
    'graph-mode':      'dependence-dag',
    'node-color-call': 'duration-total',
    'node-size-call':  'fixed',
    'direction':       'vertical',
    'edge-width':      'fixed',
    'edge-opacity':    'fixed',
  },

  // Node / edge state
  opaqueNodes:      [],   // node IDs to show at full opacity; others are dimmed
  baseZoom:         1,    // zoom level at gesturestart (pinch-zoom baseline)
  isZooming:        false,

  // Render lifecycle
  _collapsedNodes:    {},   // persistent_region_id → true/false (collapsed/expanded)
  _allRegionPids:     [],   // full list of region pids from last /graph response
  _renderSeq:         0,    // bumped on every new render; stale callbacks self-discard
  _renderAbort:       null, // AbortController for the active /graph fetch
  _currentLayout:     null, // handle returned by _startLayoutWorker (supports .stop())

  // Graph data
  _lastGraphNodes: [],
  _lastNodeCount:  0,
  _lastMinmax:        null,   // min/max ranges for colour normalisation

  // Gantt state
  _ganttPaneOpen:       true,  // overridden by loadSettings
  _ganttPaneRatio:      0,     // unused — kept for settings compatibility
  _ganttScale:          1,     // x-axis scale multiplier
  _ganttSelectedNodeId: null,
  _ganttLockedTime:     null,  // time of the locked cursor line (survives re-renders)
  _ganttHoverTime:      null,  // time under the hover cursor (null when off-chart)

  // Source pane + legend
  _sourcePaneOpen: true,  // overridden by loadSettings
  _legendOpen:     true,  // overridden by loadSettings
  _codePaths:      true,  // overridden by loadSettings

  // Colour spectrum — active key into COLOR_SPECTRUMS; overridden by loadSettings
  _colorSpectrum:  'default',

  // Dark mode
  _darkMode: false,

  // Highlight.js code theme (filename without .css extension)
  _hljsTheme: 'default',

  // Info pane
  _infoPaneRatio:  0,     // last-known open height as fraction of right pane; 0 = default

  // Debounce handles
  _settingChangedTimer: null,

  // Set to true by worker-loader.js after the first trace is loaded.
  // Controls whether the menubar render indicator is shown.
  _traceLoaded: false,
};

// ── Edge-grouping bookkeeping ─────────────────────────────────────────────────
// Both Sets are mutated in place (clear / add) — const means "never reassigned".
export const _groupedEdgeReps   = new Set(); // IDs of representative edges (modified labels)
export const _groupedEdgeHidden = new Set(); // IDs of duplicate edges hidden by our grouping

// ── Highlight / selection colour ────────────────────────────────────────────
// Used by Cytoscape (selected_gutter border, edge-highlighted), Gantt bar stroke,
// source-line background, and gutter badges.  Change both constants to retheme.
export const HIGHLIGHT_COLOR_LIGHT = '#0055cc';
export const HIGHLIGHT_COLOR_DARK  = '#e8720c';   // pumpkin orange

// Default highlight.js theme to switch to when toggling dark/light mode.
export const HLJS_THEME_DEFAULT_LIGHT = 'default';
export const HLJS_THEME_DEFAULT_DARK  = 'tokyo-night-dark';

/** Returns the active highlight colour based on the current dark-mode state. */
export function highlightColor() {
  return document.documentElement.classList.contains('dark-mode')
    ? HIGHLIGHT_COLOR_DARK
    : HIGHLIGHT_COLOR_LIGHT;
}

// ── Animation durations (ms) ─────────────────────────────────────────────────
export const ANIM_ZOOM_MS   = 400;   // graph pan/zoom-to-node, expand/collapse
export const ANIM_SCROLL_MS = 400;   // Gantt scroll, code-listing scroll

// ── Edge behaviour ───────────────────────────────────────────────────────────
// When true, selecting a node highlights its directly connected edges in blue.
export const HIGHLIGHT_CONNECTED_EDGES = true;

// ── Gantt layout constants ───────────────────────────────────────────────────
export const GANTT_LW        = 72;   // label column width (px)
export const GANTT_PAD_RIGHT = 90;   // space reserved at the right for the "end of trace" label

// Gantt row geometry (px)
export const GANTT_ROW_H       = 28;  // height of each backend / transfer / event row
export const GANTT_AXIS_H      = 36;  // bottom axis strip height; also the regions row height
export const GANTT_ROW_PAD     = 3;   // top/bottom padding inside a row bar
export const GANTT_MIN_LABEL_W = 24;  // minimum bar width (px) before drawing an inline label
export const GANTT_TICK_PX          = 90;    // target pixel spacing between time-axis ticks
export const GANTT_PINCH_SENSITIVITY = 0.08; // ctrl+wheel / touch pinch scale speed (exp factor per delta-pixel)

// ── Zoom and layout padding ──────────────────────────────────────────────────
export const ZOOM_FIT_PAD      = 40;   // padding in zoomToFit / zoomToFitGroup
export const ZOOM_NODE_DIVISOR = 2.3;  // cy.height() ÷ this = padding when fitting to a node

// ── Debounce / placement delays (ms) ────────────────────────────────────────
export const SETTING_CHANGED_DEBOUNCE_MS = 50;   // coalesce rapid setting-change calls
export const GANTT_RESIZE_DEBOUNCE_MS    = 60;   // wait after resize before re-rendering Gantt
export const BADGE_PLACEMENT_DELAY_MS    = 200;  // delay before inserting source-gutter badges

// ── Info pane ────────────────────────────────────────────────────────────────
export const INFO_PANE_MIN_H = 40;   // minimum info pane height (px)

// ── Auto-render threshold ────────────────────────────────────────────────────
export const AUTO_RENDER_THRESHOLD = 200; // node count below which changes auto-re-render

// ── Global exposure ──────────────────────────────────────────────────────────
// Classic <script> blocks (the HTML inline script) access State via window.State.
window.State = State;
