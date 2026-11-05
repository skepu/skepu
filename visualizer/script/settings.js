// settings.js — Centralised settings persistence and menu sync.
// Reads/writes State.settings instead of hidden DOM form controls.
import { knownTheme, SETTINGS_KEY } from './appearance.js';
import { State } from './state.js';
import { setPaneOpen } from './panes.js';
import { setWheelZooms } from './panzoom.js';

var STORAGE_KEY = SETTINGS_KEY;

/**
 * The source pane's width, or the width it is sliding to: its inline flex-basis (a gutter drag, the stored width). offsetWidth is the width now, which
 * is wrong while the pane slides open (0 at the start).
 */
function paneWidth(el) {
  var m = /(\d+(?:\.\d+)?)px\s*$/.exec(el.style.flex || '');
  var w = m ? Math.round(+m[1]) : el.offsetWidth;
  return w > 0 ? w : 0;
}

/** Pane sizes are read from the page into the settings; a hidden pane has none and keeps the stored one. */
function measurePanes() {
  var rightEl = document.getElementById('right_pane');
  var infoEl  = document.getElementById('info_window');
  if (State.settings.get('source-open') && State._traceLoaded && rightEl) {
    var w = paneWidth(rightEl);
    if (w > 0) State.settings.set('right-pane-width', w);
  }
  if (infoEl && rightEl && infoEl.offsetHeight > 40 && State._traceLoaded)
    State.settings.set('info-pane-ratio', infoEl.offsetHeight / rightEl.offsetHeight);
}

export function saveSettings() {
  measurePanes();
  localStorage.setItem(STORAGE_KEY, JSON.stringify(State.settings.persisted()));
}

/** The source view is created before the settings load, so it is told their values. */
function applyToSourceView() {
  if (!State.sourceView) return;
  State.sourceView.setCodePaths(State.settings.get('code-paths'));
  State.sourceView.setPinDefinitions(State.settings.get('pin-definitions'));
}

export function loadSettings() {
  var stored = localStorage.getItem(STORAGE_KEY);
  var s;
  try { s = stored && JSON.parse(stored); } catch(e) { s = null; }
  if (s) {
    var kept = {};
    [['check', s.checkboxes], ['select', s.selects], ['layout', s.layout]].forEach(function(section) {
      State.settings.keys(section[0]).forEach(function(id) {
        if (section[1] && section[1][id] != null) kept[id] = section[1][id];
      });
    });
    // predates the single Aggregate setting: an on/off check plus the key to aggregate by
    if (s.checkboxes && 'collapse_iteration' in s.checkboxes && !(s.selects && 'aggregate' in s.selects))
      kept.aggregate = s.checkboxes.collapse_iteration ? ((s.selects && s.selects.aggregate_by) || 'name') : 'none';
    // a choice the menu no longer offers (a removed colour scheme) is dropped
    State.settings.keys('select').forEach(function(id) {
      var items = Array.from(document.querySelectorAll('.menu-radio[data-ctrl="' + id + '"]'));
      if (id in kept && items.length && !items.some(function(b) { return b.dataset.value === kept[id]; })) delete kept[id];
    });
    if ('hljs-theme' in kept) kept['hljs-theme'] = knownTheme(kept['hljs-theme']);
    State.settings.update(kept);
    if (typeof _applyHljsTheme === 'function') _applyHljsTheme(State.settings.get('hljs-theme'));
  }
  syncSettingsMenu();
  applyToSourceView();
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  if (typeof syncSpectrumMenuIcons === 'function') syncSpectrumMenuIcons();
}

/** Applies a layout setting to the page, whoever changed it. */
State.settings.onChange(function(keys) {
  var get = function(key) { return State.settings.get(key); };
  if (keys.includes('gantt-scale')) {
    var sl = document.getElementById('gantt-scale');
    if (sl) sl.value = get('gantt-scale');
    var lb = document.getElementById('gantt-scale-label');
    if (lb) lb.textContent = parseFloat(get('gantt-scale')).toFixed(0) + '×';
  }
  if (keys.includes('wheel-zoom')) setWheelZooms(get('wheel-zoom'));
  if (keys.includes('progress-bar') && State.simEngine) State.simEngine.setProgressBarEnabled(get('progress-bar'));
  if (keys.includes('code-paths') || keys.includes('pin-definitions')) applyToSourceView();
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
});

/** `data-mpi` items show for MPI traces; a choice already made stays shown. */
function syncMpiItems() {
  document.querySelectorAll('[data-mpi]').forEach(function(btn) {
    var chosen = btn.classList.contains('menu-radio') && State.settings.get(btn.dataset.ctrl) === btn.dataset.value;
    btn.hidden = !State._isMpi && !chosen;
  });
}

/**
 /** Menu state from the settings: effective values of checks and radios; an overridden setting is disabled with the reason as tooltip. Runs on every effective change. */
export function syncSettingsMenu() {
  var settings = State.settings;
  syncMpiItems();
  settings.keys('check').forEach(function(id) {
    var btn = document.querySelector('.menu-check[data-ctrl="' + id + '"]');
    if (!btn) return;
    var lock = settings.lock(id);
    btn.querySelector('.menu-icon').textContent = settings.get(id) ? '✓' : '';
    btn.disabled = !!lock;
    if (lock) { btn.title = lock; btn.dataset.locked = '1'; }
    else if (btn.dataset.locked) { btn.title = ''; delete btn.dataset.locked; }
  });
  settings.keys('select').forEach(function(id) {
    document.querySelectorAll('.menu-radio[data-ctrl="' + id + '"]').forEach(function(b) {
      b.querySelector('.menu-icon').textContent = b.dataset.value === settings.get(id) ? '✓' : '';
    });
  });
}
State.settings.onChange(syncSettingsMenu);

export function applyStoredLayout() {
  var rightEl = document.getElementById('right_pane');
  if (rightEl) rightEl.style.flex = '0 0 ' + State.settings.get('right-pane-width') + 'px';
}

export function toggleSetting(id) {
  if (!State.settings.toggle(id)) return;
  saveSettings();
  if (typeof settingChanged === 'function') settingChanged();
}

export function menuToggleCheck(btn) {
  if (!State.settings.toggle(btn.dataset.ctrl)) return;
  saveSettings();
  if (typeof menuClose === 'function') menuClose();
  if (typeof settingChanged === 'function') settingChanged();
}

export function menuSelectRadio(btn) {
  State.settings.set(btn.dataset.ctrl, btn.dataset.value);
  saveSettings();
  if (typeof menuClose === 'function') menuClose();
  if (typeof settingChanged === 'function') settingChanged();
}

export function menuRestoreDefaults() {
  State.settings.restoreDefaults();
  var get = function(key) { return State.settings.get(key); };
  syncSettingsMenu();
  applyStoredLayout();
  document.getElementById('info_window').style.flex = '0 0 0em';
  setPaneOpen(document.getElementById('legend'), get('legend-open'), 'height', false);
  if (typeof setHljsTheme === 'function') setHljsTheme(get('hljs-theme'));
  if (get('source-open')) { if (typeof openSourcePane === 'function') openSourcePane(); }
  else { if (typeof closeSourcePane === 'function') closeSourcePane(); }
  if (get('gantt-open')) { if (typeof openGanttPane === 'function') openGanttPane(); }
  else { if (typeof closeGanttPane === 'function') closeGanttPane(); }
  if (typeof ganttScaleChanged === 'function') ganttScaleChanged(get('gantt-scale'));
  if (typeof syncSpectrumMenuIcons === 'function') syncSpectrumMenuIcons();
  saveSettings();
  if (typeof menuClose === 'function') menuClose();
}

// ── Global exposure ───────────────────────────────────────────────────────────
window.saveSettings        = saveSettings;
window.syncSettingsMenu    = syncSettingsMenu;
window.loadSettings        = loadSettings;
window.applyStoredLayout   = applyStoredLayout;
window.toggleSetting       = toggleSetting;
window.menuToggleCheck     = menuToggleCheck;
window.menuSelectRadio     = menuSelectRadio;
window.menuRestoreDefaults = menuRestoreDefaults;
