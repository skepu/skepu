// settings.js — Centralised settings persistence and menu sync.
// Reads/writes State.settings instead of hidden DOM form controls.
import { knownTheme, SETTINGS_KEY } from './appearance.js';
import { State } from './state.js';
import { setPaneOpen } from './panes.js';
import { setWheelZooms } from './panzoom.js';

var STORAGE_KEY = SETTINGS_KEY;

var SETTING_CHECKBOXES = [
  'container-updates', 'container-allocations', 'container-deallocations', 'container-snapshots',
  'container-transfers', 'kernel-compiles', 'communications', 'element-accesses', 'anti-deps', 'virtual-alias-edges',
  'edge-labels', 'node-labels', 'show_regions', 'collapse_iteration', 'fusion_analysis',
];
var SETTING_SELECTS = [
  'graph-mode', 'node-color-call', 'node-size-call', 'direction', 'edge-width', 'edge-opacity',
  'aggregate_by',
];

var DEFAULTS = {
  checkboxes: {
    'container-updates': false, 'container-allocations': false,
    'container-deallocations': false, 'container-snapshots': true, 'container-transfers': false, 'kernel-compiles': false, 'communications': true,
    'element-accesses': false, 'anti-deps': false, 'virtual-alias-edges': false,
    'edge-labels': false, 'node-labels': true, 'show_regions': false,
    'collapse_iteration': false, 'fusion_analysis': false,
  },
  selects: {
    'graph-mode': 'dependence-dag', 'node-color-call': 'duration-total', 'node-size-call': 'fixed',
    'direction': 'vertical', 'edge-width': 'fixed', 'edge-opacity': 'fixed', 'aggregate_by': 'name',
  },
  layout: {
    'right-pane-width': 500, 'source-open': true, 'info-pane-ratio': 0,
    'gantt-open': false, 'gantt-scale': 1, 'legend-open': false,
    'color-spectrum': 'default', 'dark-mode': false, 'hljs-theme': 'default',
    'code-paths': true, 'pin-definitions': false, 'progress-bar': false, 'wheel-zoom': false, 'info-enabled': true,
  },
};

// In an editor (VS Code) the source is in the editor's own files: the right pane (the info pane) opens with a selection, and the timeline and
// the legend start open. The dark mode is the editor's theme (theme_vscode.js).
if (typeof document !== 'undefined' && document.documentElement.classList.contains('editor'))
  Object.assign(DEFAULTS.layout, { 'source-open': false, 'gantt-open': true, 'legend-open': true });


/**
 * The width that the source pane has, or is sliding to: its flex-basis, which is set inline (a drag of the gutter, the stored width). Its
 * offsetWidth is the width at this moment, which is not what is wanted while the pane slides open (it is 0 at the start).
 */
function paneWidth(el) {
  var m = /(\d+(?:\.\d+)?)px\s*$/.exec(el.style.flex || '');
  var w = m ? Math.round(+m[1]) : el.offsetWidth;
  return w > 0 ? w : DEFAULTS.layout['right-pane-width'];   // (a pane that is not shown has no width: that is not a width to keep)
}

export function saveSettings(layoutOverrides) {
  var s = { checkboxes: {}, selects: {}, layout: {} };
  SETTING_CHECKBOXES.forEach(function(id) { s.checkboxes[id] = State.settings[id]; });
  SETTING_SELECTS.forEach(function(id)    { s.selects[id]    = State.settings[id]; });
  // Source pane width: preserve stored value when closed or in welcome mode
  // (welcome mode hides #right_pane via CSS so offsetWidth would be 0).
  if (State._sourcePaneOpen && State._traceLoaded) {
    var rightEl = document.getElementById('right_pane');
    if (rightEl) s.layout['right-pane-width'] = paneWidth(rightEl);
  } else {
    var prev = {};
    try { prev = JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch(e) {}
    s.layout['right-pane-width'] = (prev.layout && prev.layout['right-pane-width'] > 0)
      ? prev.layout['right-pane-width'] : DEFAULTS.layout['right-pane-width'];
  }
  s.layout['source-open'] = State._sourcePaneOpen;
  if (layoutOverrides && 'info-pane-ratio' in layoutOverrides) {
    s.layout['info-pane-ratio'] = layoutOverrides['info-pane-ratio'];
  } else {
    var infoEl  = document.getElementById('info_window');
    var rightEl = document.getElementById('right_pane');
    if (infoEl && rightEl && infoEl.offsetHeight > 40 && State._traceLoaded) {
      s.layout['info-pane-ratio'] = infoEl.offsetHeight / rightEl.offsetHeight;
    } else {
      var prev = {};
      try { prev = JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch(e) {}
      s.layout['info-pane-ratio'] = (prev.layout && prev.layout['info-pane-ratio'] != null)
        ? prev.layout['info-pane-ratio'] : DEFAULTS.layout['info-pane-ratio'];
    }
  }
  s.layout['gantt-open']     = State._ganttPaneOpen;
  s.layout['gantt-scale']    = State._ganttScale;
  s.layout['legend-open']    = State._legendOpen;
  s.layout['color-spectrum'] = State._colorSpectrum;
  s.layout['dark-mode']      = State._darkMode;
  s.layout['hljs-theme']     = State._hljsTheme;
  s.layout['code-paths']     = State._codePaths;
  s.layout['pin-definitions'] = State._pinDefinitions;
  s.layout['progress-bar']    = State._progressBar;
  s.layout['wheel-zoom']      = State._wheelZoom;
  s.layout['info-enabled']    = State._infoEnabled;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
}

/**
 * The options of the source view that are kept in State. The view is made with the defaults before the settings are read, so it is told
 * the values that were read.
 */
function applyToSourceView() {
  if (!State.sourceView) return;
  State.sourceView.setCodePaths(State._codePaths);
  State.sourceView.setPinDefinitions(State._pinDefinitions);
}

export function loadSettings() {
  var stored = localStorage.getItem(STORAGE_KEY);
  if (!stored) {
    if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
    return;
  }
  var s;
  try { s = JSON.parse(stored); } catch(e) {
    if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
    return;
  }
  SETTING_CHECKBOXES.forEach(function(id) {
    if (!s.checkboxes || !(id in s.checkboxes)) return;
    State.settings[id] = s.checkboxes[id];
    var btn = document.querySelector('.menu-check[data-ctrl="' + id + '"]');
    if (btn) btn.querySelector('.menu-icon').textContent = State.settings[id] ? '✓' : '';
  });
  SETTING_SELECTS.forEach(function(id) {
    if (!s.selects || !(id in s.selects)) return;
    State.settings[id] = s.selects[id];
    document.querySelectorAll('.menu-radio[data-ctrl="' + id + '"]').forEach(function(b) {
      b.querySelector('.menu-icon').textContent = b.dataset.value === State.settings[id] ? '✓' : '';
    });
  });
  if (s.layout) {
    if (s.layout['info-pane-ratio'] != null) State._infoPaneRatio  = s.layout['info-pane-ratio'];
    if (s.layout['source-open']     != null) State._sourcePaneOpen = s.layout['source-open'];
    if (s.layout['gantt-open']      != null) State._ganttPaneOpen  = s.layout['gantt-open'];
    if (s.layout['legend-open']     != null) State._legendOpen     = s.layout['legend-open'];
    if (s.layout['color-spectrum']  != null) State._colorSpectrum  = s.layout['color-spectrum'];
    if (s.layout['gantt-scale']     != null) {
      State._ganttScale = s.layout['gantt-scale'];
      var sl = document.getElementById('gantt-scale');
      if (sl) sl.value = State._ganttScale;
      var lb = document.getElementById('gantt-scale-label');
      if (lb) lb.textContent = parseFloat(State._ganttScale).toFixed(0) + '×';
    }
    if (s.layout['dark-mode'] != null) {
      State._darkMode = s.layout['dark-mode'];
      document.documentElement.classList.toggle('dark-mode', State._darkMode);
    }
    if (s.layout['hljs-theme'] != null) {
      State._hljsTheme = knownTheme(s.layout['hljs-theme']);
      if (typeof _applyHljsTheme === 'function') _applyHljsTheme(State._hljsTheme);
    }
    if (s.layout['code-paths'] != null) {
      State._codePaths = s.layout['code-paths'];
    }
    if (s.layout['progress-bar'] != null) State._progressBar = s.layout['progress-bar'];
    if (s.layout['info-enabled'] != null) State._infoEnabled = s.layout['info-enabled'];
    if (s.layout['wheel-zoom'] != null) { State._wheelZoom = s.layout['wheel-zoom']; setWheelZooms(State._wheelZoom); }
    if (s.layout['pin-definitions'] != null) {
      State._pinDefinitions = s.layout['pin-definitions'];
    }
  }
  applyToSourceView();
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  if (typeof syncSpectrumMenuIcons === 'function') syncSpectrumMenuIcons();
}

export function applyStoredLayout() {
  var stored = localStorage.getItem(STORAGE_KEY);
  if (!stored) return;
  var s;
  try { s = JSON.parse(stored); } catch(e) { return; }
  if (!s.layout) return;
  if (s.layout['right-pane-width'] > 0) {
    var rightEl = document.getElementById('right_pane');
    if (rightEl) rightEl.style.flex = '0 0 ' + s.layout['right-pane-width'] + 'px';
  }
}

export function toggleSetting(id) {
  State.settings[id] = !State.settings[id];
  var btn = document.querySelector('.menu-check[data-ctrl="' + id + '"]');
  if (btn) btn.querySelector('.menu-icon').textContent = State.settings[id] ? '✓' : '';
  saveSettings();
  if (typeof settingChanged === 'function') settingChanged();
}

export function menuToggleCheck(btn) {
  var id = btn.dataset.ctrl;
  State.settings[id] = !State.settings[id];
  btn.querySelector('.menu-icon').textContent = State.settings[id] ? '✓' : '';
  saveSettings();
  if (typeof menuClose === 'function') menuClose();
  if (typeof settingChanged === 'function') settingChanged();
}

export function menuSelectRadio(btn) {
  var ctrlId = btn.dataset.ctrl;
  var value  = btn.dataset.value;
  State.settings[ctrlId] = value;
  document.querySelectorAll('.menu-radio[data-ctrl="' + ctrlId + '"]').forEach(function(b) {
    b.querySelector('.menu-icon').textContent = b.dataset.value === value ? '✓' : '';
  });
  saveSettings();
  if (typeof menuClose === 'function') menuClose();
  if (typeof settingChanged === 'function') settingChanged();
}

export function menuRestoreDefaults() {
  SETTING_CHECKBOXES.forEach(function(id) {
    State.settings[id] = DEFAULTS.checkboxes[id];
    var btn = document.querySelector('.menu-check[data-ctrl="' + id + '"]');
    if (btn) btn.querySelector('.menu-icon').textContent = State.settings[id] ? '✓' : '';
  });
  SETTING_SELECTS.forEach(function(id) {
    State.settings[id] = DEFAULTS.selects[id];
    document.querySelectorAll('.menu-radio[data-ctrl="' + id + '"]').forEach(function(b) {
      b.querySelector('.menu-icon').textContent = b.dataset.value === DEFAULTS.selects[id] ? '✓' : '';
    });
  });
  var rightEl = document.getElementById('right_pane');
  if (rightEl) rightEl.style.flex = '0 0 ' + DEFAULTS.layout['right-pane-width'] + 'px';
  State._sourcePaneOpen = DEFAULTS.layout['source-open'];
  State._infoPaneRatio  = DEFAULTS.layout['info-pane-ratio'];
  State._ganttPaneOpen  = DEFAULTS.layout['gantt-open'];
  State._legendOpen     = DEFAULTS.layout['legend-open'];
  State._colorSpectrum  = DEFAULTS.layout['color-spectrum'];
  State._codePaths      = DEFAULTS.layout['code-paths'];
  State._pinDefinitions = DEFAULTS.layout['pin-definitions'];
  State._progressBar    = DEFAULTS.layout['progress-bar'];
  State._infoEnabled    = DEFAULTS.layout['info-enabled'];
  State._wheelZoom      = DEFAULTS.layout['wheel-zoom'];
  setWheelZooms(State._wheelZoom);
  if (State.simEngine) State.simEngine.setProgressBarEnabled(State._progressBar);
  applyToSourceView();
  document.getElementById('info_window').style.flex = '0 0 0em';
  setPaneOpen(document.getElementById('legend'), State._legendOpen, 'height', false);
  if (typeof setHljsTheme === 'function') setHljsTheme(DEFAULTS.layout['hljs-theme']);
  if (State._sourcePaneOpen) { if (typeof openSourcePane === 'function') openSourcePane(); }
  else { if (typeof closeSourcePane === 'function') closeSourcePane(); }
  if (State._ganttPaneOpen) { if (typeof openGanttPane === 'function') openGanttPane(); }
  else { if (typeof closeGanttPane === 'function') closeGanttPane(); }
  if (typeof ganttScaleChanged === 'function') ganttScaleChanged(DEFAULTS.layout['gantt-scale']);
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  if (typeof syncSpectrumMenuIcons === 'function') syncSpectrumMenuIcons();
  saveSettings({ 'info-pane-ratio': DEFAULTS.layout['info-pane-ratio'] });
  if (typeof menuClose === 'function') menuClose();
}

// ── Global exposure ───────────────────────────────────────────────────────────
window.saveSettings        = saveSettings;
window.loadSettings        = loadSettings;
window.applyStoredLayout   = applyStoredLayout;
window.toggleSetting       = toggleSetting;
window.menuToggleCheck     = menuToggleCheck;
window.menuSelectRadio     = menuSelectRadio;
window.menuRestoreDefaults = menuRestoreDefaults;
