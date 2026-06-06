// settings.js — Centralised settings persistence and menu sync.
// Reads/writes State.settings instead of hidden DOM form controls.
import { State } from './state.js';

var STORAGE_KEY = 'skepu-visualizer-settings-v1';

var SETTING_CHECKBOXES = [
  'container-updates', 'container-allocations', 'container-deallocations',
  'container-transfers', 'element-accesses', 'anti-deps', 'virtual-alias-edges',
  'edge-labels', 'node-labels', 'show_regions', 'collapse_iteration', 'fusion_analysis',
];
var SETTING_SELECTS = [
  'graph-mode', 'node-color-call', 'node-size-call', 'direction', 'edge-width', 'edge-opacity',
];

var DEFAULTS = {
  checkboxes: {
    'container-updates': false, 'container-allocations': false,
    'container-deallocations': false, 'container-transfers': false,
    'element-accesses': false, 'anti-deps': false, 'virtual-alias-edges': false,
    'edge-labels': false, 'node-labels': true, 'show_regions': false,
    'collapse_iteration': false, 'fusion_analysis': false,
  },
  selects: {
    'graph-mode': 'dependence-dag', 'node-color-call': 'duration-total', 'node-size-call': 'fixed',
    'direction': 'vertical', 'edge-width': 'fixed', 'edge-opacity': 'fixed',
  },
  layout: {
    'right-pane-width': 500, 'source-open': true, 'info-pane-ratio': 0,
    'gantt-open': false, 'gantt-scale': 1, 'legend-open': false,
    'color-spectrum': 'default', 'dark-mode': false, 'hljs-theme': 'default',
    'code-paths': true,
  },
};

export function saveSettings(layoutOverrides) {
  var s = { checkboxes: {}, selects: {}, layout: {} };
  SETTING_CHECKBOXES.forEach(function(id) { s.checkboxes[id] = State.settings[id]; });
  SETTING_SELECTS.forEach(function(id)    { s.selects[id]    = State.settings[id]; });
  // Source pane width: preserve stored value when closed or in welcome mode
  // (welcome mode hides #right_pane via CSS so offsetWidth would be 0).
  if (State._sourcePaneOpen && State._traceLoaded) {
    var rightEl = document.getElementById('right_pane');
    if (rightEl) s.layout['right-pane-width'] = rightEl.offsetWidth;
  } else {
    var prev = {};
    try { prev = JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch(e) {}
    s.layout['right-pane-width'] = (prev.layout && prev.layout['right-pane-width'] != null)
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
  localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
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
      State._hljsTheme = s.layout['hljs-theme'];
      if (typeof _applyHljsTheme === 'function') _applyHljsTheme(State._hljsTheme);
    }
    if (s.layout['code-paths'] != null) {
      State._codePaths = s.layout['code-paths'];
    }
  }
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  if (typeof syncSpectrumMenuIcons === 'function') syncSpectrumMenuIcons();
}

export function applyStoredLayout() {
  var stored = localStorage.getItem(STORAGE_KEY);
  if (!stored) return;
  var s;
  try { s = JSON.parse(stored); } catch(e) { return; }
  if (!s.layout) return;
  if (s.layout['right-pane-width'] != null) {
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
  document.getElementById('info_window').style.flex = '0 0 0em';
  document.getElementById('legend').style.display = State._legendOpen ? '' : 'none';
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
