// ui.js — Source pane, alerts, setting-changed debounce, legend, and page init.
// Entry-point module: imports from all other modules and wires up window exports.
import { descendantsOf, drawnCollapsed } from './nesting.js';
import { State, SETTING_CHANGED_DEBOUNCE_MS, GANTT_RESIZE_DEBOUNCE_MS,
         INFO_PANE_MIN_H, AUTO_RENDER_THRESHOLD,
         HLJS_THEME_DEFAULT_LIGHT, HLJS_THEME_DEFAULT_DARK } from './state.js';
import { saveSettings, loadSettings, applyStoredLayout, toggleSetting } from './settings.js';
import { navigateHeatmap } from './heatmap.js';
import { installMenus, enableMenuItem } from './menu.js';
import { attachResizer } from './resizer.js';
import { setPaneOpen } from './panes.js';
import { reveal } from './reveal.js';
import { installShortcuts, shortcutsOpen, toggleShortcuts } from './shortcuts.js';
import { installTour } from './tour.js';
import { buildThemeMenu, buildSpectrumMenu } from './appearance.js';
import './graph.js';   // side-effect: creates GraphView, LegendView, SimulationEngine, sets State.*, wires window globals

// ── Colour spectrum helpers ───────────────────────────────────────────────────

// Re-render the legend panel.  Delegates to State.legendView (set by graph.js).
function renderLegend() {
  if (State.legendView) State.legendView.render(State.settings.get('dark-mode'));
}

// Sync the checkmarks in the Color Spectrum dropdown menu, then re-render the
// legend so the gradient bar reflects the newly active spectrum.
export function syncSpectrumMenuIcons() {
  document.querySelectorAll('.spectrum-radio').forEach(function(b) {
    b.querySelector('.menu-icon').textContent = b.dataset.spectrum === State.settings.get('color-spectrum') ? '✓' : '';
  });
  renderLegend();
}

// Called by inline onclick handlers in index.html when the user picks a spectrum.
export function colorSpectrumChanged(name) {
  State.settings.set('color-spectrum', name);
  syncSpectrumMenuIcons();
  if (typeof saveSettings === 'function') saveSettings();
  settingChanged();
}

window.syncSpectrumMenuIcons = syncSpectrumMenuIcons;
window.colorSpectrumChanged  = colorSpectrumChanged;

// ── Gantt / timeline helpers ──────────────────────────────────────────────────

export function openGanttPane() {
  State.settings.set('gantt-open', true);
  if (State.timelineView) State.timelineView.open();
}

export function closeGanttPane() {
  State.settings.set('gantt-open', false);
  if (State.timelineView) State.timelineView.close();
}

export function toggleGanttPane() {
  enablePaneAnimation();
  if (State.settings.get('gantt-open')) closeGanttPane(); else openGanttPane();
}

export function ganttScaleChanged(val) {
  State.settings.set('gantt-scale', parseFloat(val));
  if (State.timelineView) State.timelineView.setScale(State.settings.get('gantt-scale'));
}

export function zoomGanttToMeasuredInterval() {
  if (State.timelineView) State.timelineView.zoomToMeasuredInterval();
}

function reRenderGanttPreservingScroll() {
  if (State.timelineView) State.timelineView.reRenderPreservingScroll();
}

function initGanttPinchZoom() {
  if (State.timelineView) State.timelineView.initPinchZoom();
}

export function recenterGraph() {
  if (State.graphView)
    State.graphView.relayout();
}

window.openGanttPane               = openGanttPane;
window.closeGanttPane              = closeGanttPane;
window.toggleGanttPane             = toggleGanttPane;
window.ganttScaleChanged           = ganttScaleChanged;
window.zoomGanttToMeasuredInterval = zoomGanttToMeasuredInterval;
window.recenterGraph               = recenterGraph;

// ── graph.js exposes render() via State.graphView; this local helper is used
// by the many call sites in this module that need closePopup() + render together.
export function fetchViewDataAndRender() {
  closePopup();
  if (State.graphView) State.graphView.render();
}

/** Panes slide only when the user opens or closes them, not for the first trace (early.js sets the class; this removes it). */
export function enablePaneAnimation() {
  document.documentElement.classList.remove('no-pane-anim');
}

/** The page of an editor (VS Code): the source is in the editor's own files, so the right pane is only the info pane. */
export const inEditor = () => document.documentElement.classList.contains('editor');

// `animate` (true only): the pane slides; restored panes and defaults are set at once.
export function openSourcePane(animate) {
  State.settings.set('source-open', true);
  setPaneOpen(document.getElementById('right_pane'),  true, 'width', animate === true);
  setPaneOpen(document.getElementById('split-gutter'), true, 'width', animate === true);
  saveSettings();
}

export function closeSourcePane(animate) {
  State.settings.set('source-open', false);
  setPaneOpen(document.getElementById('right_pane'),  false, 'width', animate === true);
  setPaneOpen(document.getElementById('split-gutter'), false, 'width', animate === true);
  saveSettings();
}

/** Whether the info panel opens when a node is selected (View > Info Panel). Off closes it. */
export function setInfoEnabled(on) {
  State.settings.set('info-enabled', !!on);
  if (!State.settings.get('info-enabled')) closePopup();
  else if (State._ganttSelectedNodeId && State.infoView) State.infoView.open(State._ganttSelectedNodeId, 'info-panel');
  saveSettings();
}

export function toggleInfoPanel() { setInfoEnabled(!State.settings.get('info-enabled')); }

export function toggleSourcePane() {
  enablePaneAnimation();
  if (State.settings.get('source-open')) closeSourcePane(true); else openSourcePane(true);
}

// To close popup window
export function closePopup()
{
  document.getElementById('info_window').style.flex = '0 0 0em';
  if (inEditor()) {   // the right pane is only the info pane there: it closes, unless a selection that follows at once opens it again
    clearTimeout(_closeInfoTimer);
    _closeInfoTimer = setTimeout(function() { if (State.settings.get('source-open')) closeSourcePane(true); }, 120);
  }
}
let _closeInfoTimer = null;

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
  if (inEditor()) return;   // the theme of the editor decides: theme_vscode.js
  setDarkMode(!State.settings.get('dark-mode'));
}

/** Dark mode on or off, with what depends on it (the timeline, the legend, the code theme). */
export function setDarkMode(on) {
  if (State.settings.get('dark-mode') === on) return;
  State.settings.set('dark-mode', on);
  if (State.settings.get('gantt-open') && State._lastGraphNodes.length) reRenderGanttPreservingScroll();
  renderLegend();
  // Switch to the canonical code theme for the new mode.
  var defaultTheme = State.settings.get('dark-mode') ? HLJS_THEME_DEFAULT_DARK : HLJS_THEME_DEFAULT_LIGHT;
  if (typeof setHljsTheme === 'function') setHljsTheme(defaultTheme);
  saveSettings();
}

export function toggleCodePaths() {
  State.settings.set('code-paths', !State.settings.get('code-paths'));
  saveSettings();
}

export function toggleDefinitions() {
  State.settings.set('pin-definitions', !State.settings.get('pin-definitions'));
  saveSettings();
}

/** Show or hide the progress bar of the simulation (View > Progress Bar; it is turned on when a simulation starts). */
export function setProgressBar(on) {
  State.settings.set('progress-bar', !!on);
  saveSettings();
}

export function toggleProgressBar() { setProgressBar(!State.settings.get('progress-bar')); }

/** Whether a plain wheel zooms the graph instead of panning it (View > Scroll Behaviour). */
export function setWheelZoom(on) {
  State.settings.set('wheel-zoom', !!on);
  saveSettings();
}

export function setScrollPan()  { setWheelZoom(false); }
export function setScrollZoom() { setWheelZoom(true); }

export function toggleLegend()
{
  enablePaneAnimation();
  State.settings.set('legend-open', !State.settings.get('legend-open'));
  var el = document.getElementById('legend');
  if (State.settings.get('legend-open')) renderLegend();      // before it slides open, so the slide targets its height with the legend
  setPaneOpen(el, State.settings.get('legend-open'), 'height', true);
  saveSettings();
}


export function load()
{
  // The menus (script/menu.js), and the two that are the same in the designer (script/appearance.js).
  installMenus();
  installShortcuts();
  installTour();
  document.getElementById('menu-actions').addEventListener('menuopen', updateActionsMenu);
  buildThemeMenu(document.getElementById('hljs-theme-menu'), function(theme) { window.setHljsTheme(theme); });
  buildSpectrumMenu(document.getElementById('spectrum-menu'), function(name) { colorSpectrumChanged(name); });

  // Load persisted settings first so all State.* values are set before the
  // DOM is manipulated below.
  loadSettings();
  if (State.simEngine) State.simEngine.setProgressBarEnabled(State.settings.get('progress-bar'));

  // Sliding starts after the first trace is drawn, which happens with the panes already in place.
  if (State.provider) {
    var panel = document.getElementById('welcome-panel');
    State.provider.on('rendersettled', function() {
      // the page makes a render before there is a trace: that is not the first trace that is drawn
      if (!State._traceLoaded) return;
      var go = function() {
        reveal(document.documentElement, panel);       // the splash fades out: reveal.js; nothing if it is not held
        requestAnimationFrame(enablePaneAnimation);    // after the frame that has the panes in place: nothing that is pending slides
      };
      // the badges of the source pane are put in a little after the render: they would come in while the splash is half gone
      if (State.sourceView) State.sourceView.whenPlaced(go); else go();
    });
    // a first trace that fails to be drawn is no reason to keep the splash
    State.provider.on('error', function() { reveal(document.documentElement, panel); });
  }

  // ── The gutter between the graph and the right pane (script/resizer.js, as in the designer) ─────────
  var gutter  = document.getElementById('split-gutter');
  var rightEl = document.getElementById('right_pane');
  if (gutter && rightEl) attachResizer(gutter, {
    axis: 'x', min: 120,
    max:     function() { return (rightEl.parentElement ? rightEl.parentElement.clientWidth : window.innerWidth) - 200; },
    size:    function() { return rightEl.offsetWidth; },
    setSize: function(px) { rightEl.style.flex = '0 0 ' + px + 'px'; },
    onEnd:   saveSettings,
  });

  applyStoredLayout();
  initGanttPinchZoom();
  // Apply legend visibility now that #legend is in the DOM (_legendOpen was set by loadSettings).
  setPaneOpen(document.getElementById('legend'), State.settings.get('legend-open'), 'height', false);
  syncSpectrumMenuIcons();   // paint the legend gradient bar with the restored spectrum
  if (inEditor()) State.settings.set('source-open', false);   // opens with the first selection: openInfoPane
  if (!State.settings.get('source-open')) closeSourcePane();
  if (State.settings.get('gantt-open')) openGanttPane();

  // Highlight any pre-rendered source panes (active first, rest deferred).
  if (State.sourceView)
    State.sourceView.highlightAllBlocks(document.getElementById('cpp_tab_panes'));

  // Needed to render the graph
  fetchViewDataAndRender();



  var infoWindow = document.getElementById('info_window');
  attachResizer(document.getElementById('info-resizer'), {
    axis: 'y', min: INFO_PANE_MIN_H,
    size:    function() { return infoWindow.offsetHeight; },
    setSize: function(px) { infoWindow.style.flex = '0 0 ' + px + 'px'; },
    onStart: function() { infoWindow.classList.add('resizing'); },
    onEnd:   function() {
      infoWindow.classList.remove('resizing');
      var rightPane = document.getElementById('right_pane');
      if (rightPane && infoWindow.offsetHeight > INFO_PANE_MIN_H)
        State.settings.set('info-pane-ratio', infoWindow.offsetHeight / rightPane.offsetHeight);
      saveSettings();
    },
  });


  window.addEventListener("keydown", function(event) {
    // Don't steal keypresses from inputs, and ignore modifier combos.
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    // while the shortcuts are shown the keys are not for the viewer: ? closes them, Esc is shortcuts.js
    if (shortcutsOpen()) { if (event.key === "?") toggleShortcuts(); return; }

    if (event.key === "?") { toggleShortcuts(); }
    else if (event.key === " ") {
      event.preventDefault();   // prevent page scroll
      fetchViewDataAndRender();
    }
    else if (event.key === "a") { toggleSetting('container-allocations'); }
    else if (event.key === "d") { toggleSetting('container-deallocations'); }
    else if (event.key === "k") { toggleSetting('container-snapshots'); }
    else if (event.key === "n") { toggleSetting('anti-deps'); }
    else if (event.key === "t") { toggleSetting('container-transfers'); }
    else if (event.key === "r") { toggleSetting('show_regions'); }
    else if (event.key === "u") { toggleSetting('container-updates'); }
    else if (event.key === "l") { toggleSetting('edge-labels'); }
    else if (event.key === "e") { expandSelectedRecursively(); }
    else if (event.key === "c") { collapseSelectedRecursively(); }
    else if (event.key === "+") { expandSelected(); }
    else if (event.key === "-") { collapseSelected(); }
    else if (event.key === "i") { State.settings.set('aggregate', State.settings.get('aggregate') === 'none' ? 'name' : 'none'); saveSettings(); settingChanged(); }
    else if (event.key === "f") { toggleSetting('fusion_analysis'); }
    else if (event.key === "z") { zoomGanttToMeasuredInterval(); }
    else if (event.key === "g") { toggleGanttPane(); }
    else if (event.key === "m") { toggleDarkMode(); }
    else if (event.key === "s") { if (!inEditor()) toggleSourcePane(); }   // no source pane in the editor
    else if (event.key === "o") { toggleInfoPanel(); }
    else if (event.key === "p") { if (State.simEngine) State.simEngine.toggle(); }
    else if (event.key === "Tab") {
      // Tab / Shift+Tab — cycle forward / backward through source-code tabs.
      var tabs = Array.from(document.querySelectorAll('.cpp-tab'));
      if (tabs.length > 1 && !inEditor()) {
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
      else if (State.simEngine && State.simEngine.stepFromSelection(-1)) event.preventDefault();
    }
    else if (event.key === "ArrowRight") {
      if (State.simEngine && State.simEngine.isActive) { State.simEngine.step(+1); event.preventDefault(); }
      else if (navigateHeatmap(1)) event.preventDefault();
      else if (State.simEngine && State.simEngine.stepFromSelection(+1)) event.preventDefault();
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
      if (State.settings.get('gantt-open') && State._lastGraphNodes.length)
        reRenderGanttPreservingScroll();
    }, GANTT_RESIZE_DEBOUNCE_MS);
  });
}

// ── Expand / collapse actions ─────────────────────────────────────────────────
// Collapse/expand state is stored in State._collapsedNodes (persistent_region_id
// → true/false).  Every change triggers a full re-fetch so the server returns
// the correctly filtered node set; no client-side show/hide needed.

// Regions and fusion nodes are collapsed the same way: by the id that the backend gives them.
function _collapsibleId(d) {
  return d.type === 'region' || d.type === 'tree-region' ? (d.persistent_region_id === 'region_global' ? null : d.persistent_region_id)
       : d.type === 'fusion' ? d.persistent_fusion_id
       : null;
}

// With View > Regions off every region is drawn expanded, whatever is set here: say so, instead of doing nothing.
function _regionsAreShown() {
  if (State.settings.get('show_regions')) return true;
  showAlert('Regions are not shown: turn on View > Regions (R) and apply the settings first.');
  return false;
}

// Collapse or expand the regions and fusion nodes of `nodes` (data of graph nodes); with `recursive` everything inside them too (Once
// leaves the state of what is inside as it is).
function _setRegions(nodes, collapsed, recursive) {
  var changed = false;
  nodes.forEach(function(d) {
    var id = _collapsibleId(d);
    if (id && (d.type !== 'region' && d.type !== 'tree-region' || _regionsAreShown())) {
      State._collapsedNodes[id] = collapsed;
      (d.persistent_region_ids || []).forEach(function(member) { State._collapsedNodes[member] = collapsed; });
      if (recursive) descendantsOf(State._regionParents, id).forEach(function(c) { State._collapsedNodes[c] = collapsed; });
      changed = true;
    }
  });
  if (changed) fetchViewDataAndRender();
}

function _setSelectedRegions(collapsed, recursive) {
  if (State.graphView) _setRegions(State.graphView.selectedNodes(), collapsed, recursive);
}

/** Collapses or expands the region or fusion node whose data is `node` (the context menu of a node does not select it). */
export function setRegionsOf(node, collapsed, recursive) { _setRegions([node], collapsed, recursive); }

function _setAllRegions(collapsed) {
  if (!_regionsAreShown()) return;
  var pids = State._allRegionPids || Object.keys(State._collapsedNodes);
  pids.forEach(function(pid) { if (pid !== 'region_global') State._collapsedNodes[pid] = collapsed; });
  fetchViewDataAndRender();
}

// Expand the regions that are drawn collapsed, and leave the ones inside them as they are: one level.
function _expandAllRegionsOnce() {
  if (!_regionsAreShown()) return;
  var pids = State._allRegionPids || Object.keys(State._collapsedNodes);
  drawnCollapsed(State._regionParents, State._collapsedNodes, pids).forEach(function(pid) { State._collapsedNodes[pid] = false; });
  fetchViewDataAndRender();
}

function _setAllFusions(collapsed) {
  var pids = State._allFusionPids || [];
  if (!pids.length) {
    showAlert('There are no fusion nodes: turn on Nodes > Fusion Candidates and apply the settings.');
    return;
  }
  pids.forEach(function(pid) { State._collapsedNodes[pid] = collapsed; });
  fetchViewDataAndRender();
}

// ── Which of the Actions would do something now ──────────────────────────────

// Set when the Actions menu opens (menu.js says 'menuopen'): an item that would do nothing is disabled, and its title says what it needs.
// The actions themselves still tell the same, if they are called some other way.
var ACTION_NEEDS = {
  trace:     { has: function() { return State._traceLoaded; },
               why: 'Load a trace first' },
  tabs:      { has: function() { return document.querySelectorAll('.cpp-tab').length > 1; },
               why: 'There is only one source tab' },
  selection: { has: function() { return State.graphView && State.graphView.selectedNodes().some(function(d) { return _collapsibleId(d) && (d.type !== 'region' || State.settings.get('show_regions')); }); },
               why: 'Select a region or a fusion node (regions need Nodes › Regions)' },
  regions:   { has: function() { return State.settings.get('show_regions') && (State._allRegionPids || []).length > 0; },
               why: 'There are no regions to show: turn on Nodes › Regions and apply the settings' },
  fusions:   { has: function() { return (State._allFusionPids || []).length > 0; },
               why: 'There are no fusion nodes: turn on Nodes › Fusion Candidates and apply the settings' },
};

function updateActionsMenu() {
  document.querySelectorAll('#menu-actions [data-needs]').forEach(function(item) {
    var need = ACTION_NEEDS[item.dataset.needs];
    var ok = !!(need && need.has());
    enableMenuItem(item, ok);
    item.title = ok ? '' : need.why;
  });
}

export function collapseSelectedRecursively() { _setSelectedRegions(true,  true); }
export function expandSelectedRecursively()   { _setSelectedRegions(false, true); }
export function collapseSelected()            { _setSelectedRegions(true,  false); }
export function expandSelected()              { _setSelectedRegions(false, false); }
export function collapseAllRegions()          { _setAllRegions(true); }
export function expandAllRegions()            { _setAllRegions(false); }
export function expandAllRegionsOnce()        { _expandAllRegionsOnce(); }
export function collapseAllFusions()          { _setAllFusions(true); }
export function expandAllFusions()            { _setAllFusions(false); }

// ── Menu icons and the info pane (called by the other modules through window) ────

export function syncExternalMenuIcons() {
  function mark(id, on) {
    var b = document.getElementById(id);
    if (b) b.querySelector('.menu-icon').textContent = on ? '✓' : '';
  }
  mark('menu-check-legend',      State.settings.get('legend-open'));
  mark('menu-check-timeline',    State.settings.get('gantt-open'));
  mark('menu-check-source',      State.settings.get('source-open'));
  mark('menu-check-info',        State.settings.get('info-enabled'));
  mark('menu-check-info-web',    State.settings.get('info-enabled'));
  mark('menu-check-progress',    State.settings.get('progress-bar'));
  mark('menu-radio-scroll-pan',  !State.settings.get('wheel-zoom'));
  mark('menu-radio-scroll-zoom', State.settings.get('wheel-zoom'));
  mark('menu-check-code-paths',  State.settings.get('code-paths'));
  mark('menu-check-definitions', State.settings.get('pin-definitions'));
  mark('menu-check-dark',        State.settings.get('dark-mode'));
  // hljs theme icons are owned by SourceCodeView (window.setHljsTheme / window._applyHljsTheme wired in graph.js).
  if (State.sourceView) State.sourceView.syncThemeIcons();
}

/** Open the info pane to the height that was remembered, or a default one. */
export function openInfoPane() {
  if (!State.settings.get('info-enabled')) return;
  var rightEl = document.getElementById('right_pane');
  var infoEl  = document.getElementById('info_window');
  if (inEditor()) {   // the right pane is the info pane: it opens with the first selection and has the whole pane
    clearTimeout(_closeInfoTimer);
    if (!State.settings.get('source-open')) openSourcePane(true);
    infoEl.style.flex = '1 1 auto';
    return;
  }
  if (State.settings.get('info-pane-ratio') > 0 && rightEl)
    infoEl.style.flex = '0 0 ' + Math.round(State.settings.get('info-pane-ratio') * rightEl.offsetHeight) + 'px';
  else
    infoEl.style.flex = '0 0 10em';
}

// ── Global exposure ───────────────────────────────────────────────────────────
// Functions that the other modules and the controls of the markup (data-click, see actions.js) call by their names.
window.syncExternalMenuIcons       = syncExternalMenuIcons;
window.openInfoPane                = openInfoPane;
window.load                        = load;
window.enablePaneAnimation         = enablePaneAnimation;
window.openSourcePane              = openSourcePane;
window.closeSourcePane             = closeSourcePane;
window.toggleSourcePane            = toggleSourcePane;
window.toggleInfoPanel             = toggleInfoPanel;
window.setRegionsOf                = setRegionsOf;
window.toggleCodePaths             = toggleCodePaths;
window.toggleDefinitions           = toggleDefinitions;
window.toggleLegend                = toggleLegend;
window.toggleProgressBar           = toggleProgressBar;
window.setScrollPan                = setScrollPan;
window.setScrollZoom               = setScrollZoom;
window.setProgressBar              = setProgressBar;
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
window.expandAllRegionsOnce        = expandAllRegionsOnce;
window.collapseAllFusions          = collapseAllFusions;
window.expandAllFusions            = expandAllFusions;
