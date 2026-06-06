// ui.js — Source pane, alerts, setting-changed debounce, legend, and page init.
// Entry-point module: imports from all other modules and wires up window exports.
import { State, SETTING_CHANGED_DEBOUNCE_MS, GANTT_RESIZE_DEBOUNCE_MS,
         INFO_PANE_MIN_H, AUTO_RENDER_THRESHOLD,
         HLJS_THEME_DEFAULT_LIGHT, HLJS_THEME_DEFAULT_DARK } from './state.js';
import { saveSettings, loadSettings, applyStoredLayout, toggleSetting } from './settings.js';
import { navigateHeatmap } from './heatmap.js';
import './graph.js';   // side-effect: creates GraphView, LegendView, SimulationEngine, sets State.*, wires window globals

// ── Colour spectrum helpers ───────────────────────────────────────────────────

// Re-render the legend panel.  Delegates to State.legendView (set by graph.js).
function renderLegend() {
  if (State.legendView) State.legendView.render(State._darkMode);
}

// Sync the checkmarks in the Color Spectrum dropdown menu, then re-render the
// legend so the gradient bar reflects the newly active spectrum.
export function syncSpectrumMenuIcons() {
  document.querySelectorAll('.spectrum-radio').forEach(function(b) {
    b.querySelector('.menu-icon').textContent = b.dataset.spectrum === State._colorSpectrum ? '✓' : '';
  });
  renderLegend();
}

// Called by inline onclick handlers in index.html when the user picks a spectrum.
export function colorSpectrumChanged(name) {
  State._colorSpectrum = name;
  syncSpectrumMenuIcons();
  if (typeof saveSettings === 'function') saveSettings();
  settingChanged();
}

window.syncSpectrumMenuIcons = syncSpectrumMenuIcons;
window.colorSpectrumChanged  = colorSpectrumChanged;

// ── Gantt / timeline helpers ──────────────────────────────────────────────────

export function openGanttPane() {
  State._ganttPaneOpen = true;   // setter in graph.js syncs TimelineView._paneOpen
  if (State.timelineView) State.timelineView.open();
}

export function closeGanttPane() {
  State._ganttPaneOpen = false;
  if (State.timelineView) State.timelineView.close();
}

export function toggleGanttPane() {
  if (State._ganttPaneOpen) closeGanttPane(); else openGanttPane();
}

export function ganttScaleChanged(val) {
  State._ganttScale = parseFloat(val);   // setter in graph.js syncs TimelineView._scale
  if (State.timelineView) State.timelineView.setScale(State._ganttScale);
}

function zoomGanttToMeasuredInterval() {
  if (State.timelineView) State.timelineView.zoomToMeasuredInterval();
}

function reRenderGanttPreservingScroll() {
  if (State.timelineView) State.timelineView.reRenderPreservingScroll();
}

function initGanttPinchZoom() {
  if (State.timelineView) State.timelineView.initPinchZoom();
}

window.openGanttPane               = openGanttPane;
window.closeGanttPane              = closeGanttPane;
window.toggleGanttPane             = toggleGanttPane;
window.ganttScaleChanged           = ganttScaleChanged;
window.zoomGanttToMeasuredInterval = zoomGanttToMeasuredInterval;
window.recenterGraph               = function() { if (State.graphView) State.graphView.relayout(); };

// ── graph.js exposes render() via State.graphView; this local helper is used
// by the many call sites in this module that need closePopup() + render together.
function fetchViewDataAndRender() {
  closePopup();
  if (State.graphView) State.graphView.render();
}

export function openSourcePane() {
  State._sourcePaneOpen = true;
  document.getElementById('right_pane').style.display  = '';
  document.getElementById('split-gutter').style.display = '';
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  saveSettings();
}

export function closeSourcePane() {
  State._sourcePaneOpen = false;
  document.getElementById('right_pane').style.display  = 'none';
  document.getElementById('split-gutter').style.display = 'none';
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  saveSettings();
}

export function toggleSourcePane() {
  if (State._sourcePaneOpen) closeSourcePane(); else openSourcePane();
}

// To close popup window
export function closePopup()
{
  document.getElementById('info_window').style.flex = '0 0 0em';
}

export function showAlert(msg) {
  var bar = document.getElementById('alert-bar');
  if (!bar) return;
  bar.textContent = msg;
  bar.style.display = 'block';
}

export function clearAlert() {
  var bar = document.getElementById('alert-bar');
  if (!bar) return;
  bar.style.display = 'none';
  bar.textContent = '';
}

// Called whenever a menu setting changes.  For small graphs it triggers an immediate
// re-render; for large graphs it shows an alert reminding the user to apply manually.
// Debounced so rapid successive calls (e.g. a checkbox dispatching its own onchange
// AND menuToggleCheck calling us) coalesce into a single action.
export function settingChanged() {
  clearTimeout(State._settingChangedTimer);
  State._settingChangedTimer = setTimeout(function () {
    renderLegend();
    if (State._lastNodeCount === 0) return; // no graph loaded yet
    if (State._lastNodeCount <= AUTO_RENDER_THRESHOLD) {
      clearAlert();
      fetchViewDataAndRender();
    } else {
      showAlert('⚠ Live re-render is disabled for large graphs ('
        + State._lastNodeCount + ' nodes > threshold ' + AUTO_RENDER_THRESHOLD
        + '). Use View › Apply settings to update.');
    }
  }, SETTING_CHANGED_DEBOUNCE_MS);
}

export function toggleDarkMode() {
  State._darkMode = !State._darkMode;
  document.documentElement.classList.toggle('dark-mode', State._darkMode);
  if (State._ganttPaneOpen && State._lastGraphNodes.length) reRenderGanttPreservingScroll();
  renderLegend();
  // Switch to the canonical code theme for the new mode.
  var defaultTheme = State._darkMode ? HLJS_THEME_DEFAULT_DARK : HLJS_THEME_DEFAULT_LIGHT;
  if (typeof setHljsTheme === 'function') setHljsTheme(defaultTheme);
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  saveSettings();
}

export function toggleCodePaths() {
  State._codePaths = !State._codePaths;
  if (State.sourceView) State.sourceView.setCodePaths(State._codePaths);
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  saveSettings();
}

export function toggleLegend()
{
  State._legendOpen = !State._legendOpen;
  var el = document.getElementById('legend');
  el.style.display = State._legendOpen ? '' : 'none';
  if (State._legendOpen) renderLegend();
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  saveSettings();
}


export function load()
{
  // Load persisted settings first so all State.* values are set before the
  // DOM is manipulated below.
  loadSettings();

  // ── Custom horizontal resizer (replaces Split.js) ────────────────────────
  (function() {
    var gutter  = document.getElementById('split-gutter');
    var rightEl = document.getElementById('right_pane');
    if (!gutter || !rightEl) return;
    var _startX, _startW;

    function onMove(ev) {
      var dx   = ev.clientX - _startX;
      var cont = rightEl.parentElement;
      var minW = 120;
      var maxW = (cont ? cont.clientWidth : window.innerWidth) - 200;
      rightEl.style.flex = '0 0 ' + Math.max(minW, Math.min(_startW - dx, maxW)) + 'px';
    }
    function onUp() {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup',   onUp);
      document.body.style.cursor = '';
      saveSettings();
    }
    gutter.addEventListener('mousedown', function(ev) {
      _startX = ev.clientX;
      _startW = rightEl.offsetWidth;
      document.body.style.cursor = 'col-resize';
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup',   onUp);
      ev.preventDefault();
    });
  })();

  applyStoredLayout();
  initGanttPinchZoom();
  // Apply legend visibility now that #legend is in the DOM (_legendOpen was set by loadSettings).
  document.getElementById('legend').style.display = State._legendOpen ? '' : 'none';
  syncSpectrumMenuIcons();   // paint the legend gradient bar with the restored spectrum
  if (!State._sourcePaneOpen) closeSourcePane();
  if (State._ganttPaneOpen) openGanttPane();

  // Highlight any pre-rendered source panes (active first, rest deferred).
  if (State.sourceView)
    State.sourceView.highlightAllBlocks(document.getElementById('cpp_tab_panes'));

  // Needed to render the graph
  fetchViewDataAndRender();



  var infoWindow  = document.getElementById('info_window');
  var infoResizer = document.getElementById('info-resizer');
  var _dragStartY, _dragStartH;

  infoResizer.addEventListener('mousedown', function(e) {
    _dragStartY = e.clientY;
    _dragStartH = infoWindow.offsetHeight;
    infoWindow.classList.add('resizing');
    document.addEventListener('mousemove', _onInfoDrag);
    document.addEventListener('mouseup',   _onInfoDragEnd);
    e.preventDefault();
  });

  function _onInfoDrag(e) {
    var newH = Math.max(INFO_PANE_MIN_H, _dragStartH + (_dragStartY - e.clientY));
    infoWindow.style.flex = '0 0 ' + newH + 'px';
  }

  function _onInfoDragEnd() {
    infoWindow.classList.remove('resizing');
    document.removeEventListener('mousemove', _onInfoDrag);
    document.removeEventListener('mouseup',   _onInfoDragEnd);
    var rightEl = document.getElementById('right_pane');
    if (rightEl && infoWindow.offsetHeight > INFO_PANE_MIN_H)
      State._infoPaneRatio = infoWindow.offsetHeight / rightEl.offsetHeight;
    saveSettings();
  }


  document.addEventListener("keydown", function(event) {
    // Don't steal keypresses from inputs, and ignore modifier combos.
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
    if (event.altKey || event.ctrlKey || event.metaKey) return;

    if (event.key === " ") {
      event.preventDefault();   // prevent page scroll
      fetchViewDataAndRender();
    }
    else if (event.key === "a") { toggleSetting('container-allocations'); }
    else if (event.key === "d") { toggleSetting('container-deallocations'); }
    else if (event.key === "n") { toggleSetting('anti-deps'); }
    else if (event.key === "t") { toggleSetting('container-transfers'); }
    else if (event.key === "r") { toggleSetting('show_regions'); }
    else if (event.key === "u") { toggleSetting('container-updates'); }
    else if (event.key === "l") { toggleSetting('edge-labels'); }
    else if (event.key === "e") { expandSelectedRecursively(); }
    else if (event.key === "c") { collapseSelectedRecursively(); }
    else if (event.key === "+") { expandSelected(); }
    else if (event.key === "-") { collapseSelected(); }
    else if (event.key === "i") { toggleSetting('collapse_iteration'); }
    else if (event.key === "f") { toggleSetting('fusion_analysis'); }
    else if (event.key === "z") { zoomGanttToMeasuredInterval(); }
    else if (event.key === "g") { toggleGanttPane(); }
    else if (event.key === "m") { toggleDarkMode(); }
    else if (event.key === "s") { toggleSourcePane(); }
    else if (event.key === "p") { if (State.simEngine) State.simEngine.toggle(); }
    else if (event.key === "Tab") {
      // Tab / Shift+Tab — cycle forward / backward through source-code tabs.
      var tabs = Array.from(document.querySelectorAll('.cpp-tab'));
      if (tabs.length > 1) {
        event.preventDefault();
        var cur  = tabs.findIndex(function(t) { return t.classList.contains('active'); });
        var next = event.shiftKey
          ? (cur - 1 + tabs.length) % tabs.length   // backward
          : (cur + 1) % tabs.length;                // forward
        switchTab(tabs[next], next);
      }
    }
    else if (event.key === "ArrowLeft") {
      if (State.simEngine && State.simEngine.isActive) { State.simEngine.step(-1); event.preventDefault(); }
      else if (navigateHeatmap(-1)) event.preventDefault();
    }
    else if (event.key === "ArrowRight") {
      if (State.simEngine && State.simEngine.isActive) { State.simEngine.step(+1); event.preventDefault(); }
      else if (navigateHeatmap(1)) event.preventDefault();
    }
    else if (event.key === "x" || event.key === "Escape") {
      closePopup();
    }
  });

  // Re-render Gantt on window resize (debounced) so the 1x scale is recomputed
  // while keeping the current visible time range centred.
  var _resizeTimer = null;
  window.addEventListener('resize', function() {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(function() {
      if (State._ganttPaneOpen && State._lastGraphNodes.length)
        reRenderGanttPreservingScroll();
    }, GANTT_RESIZE_DEBOUNCE_MS);
  });
}

// ── Expand / collapse actions ─────────────────────────────────────────────────
// Collapse/expand state is stored in State._collapsedNodes (persistent_region_id
// → true/false).  Every change triggers a full re-fetch so the server returns
// the correctly filtered node set; no client-side show/hide needed.

function _setSelectedRegions(collapsed) {
  if (!State.graphView) return;
  var changed = false;
  State.graphView.selectedNodes()
    .filter(function(d) { return d.type === 'region'; })
    .forEach(function(d) {
      if (d.persistent_region_id) {
        State._collapsedNodes[d.persistent_region_id] = collapsed;
        changed = true;
      }
    });
  if (changed) fetchViewDataAndRender();
}

function _setAllRegions(collapsed) {
  var pids = State._allRegionPids || Object.keys(State._collapsedNodes);
  pids.forEach(function(pid) { State._collapsedNodes[pid] = collapsed; });
  fetchViewDataAndRender();
}

export function collapseSelectedRecursively() { _setSelectedRegions(true); }
export function expandSelectedRecursively()   { _setSelectedRegions(false); }
export function collapseSelected()            { _setSelectedRegions(true); }
export function expandSelected()              { _setSelectedRegions(false); }
export function collapseAllRegions()          { _setAllRegions(true); }
export function expandAllRegions()            { _setAllRegions(false); }
// Fusion collapse is not implemented in the server-side approach.
export function collapseAllFusions()          { }
export function expandAllFusions()            { }

// ── Global exposure ───────────────────────────────────────────────────────────
// Functions called from HTML body onload, onclick attributes, and inline scripts.
window.load                        = load;
window.openSourcePane              = openSourcePane;
window.closeSourcePane             = closeSourcePane;
window.toggleSourcePane            = toggleSourcePane;
window.toggleCodePaths             = toggleCodePaths;
window.toggleLegend                = toggleLegend;
window.closePopup                  = closePopup;
window.showAlert                   = showAlert;
window.clearAlert                  = clearAlert;
window.settingChanged              = settingChanged;
window.toggleDarkMode              = toggleDarkMode;
window.collapseSelectedRecursively = collapseSelectedRecursively;
window.expandSelectedRecursively   = expandSelectedRecursively;
window.collapseSelected            = collapseSelected;
window.expandSelected              = expandSelected;
window.collapseAllRegions          = collapseAllRegions;
window.expandAllRegions            = expandAllRegions;
window.collapseAllFusions          = collapseAllFusions;
window.expandAllFusions            = expandAllFusions;
