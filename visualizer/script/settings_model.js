// settings_model.js — the viewer's settings: a table of definitions and the object holding the values. No DOM.
//
// Readers use `get(key)`, the effective value. A definition may override or derive it, so rules between settings live in definitions, not readers:
//   override(s)  { value, reason } while another setting forces this one, else null; the setting is then locked (`lock(key)` is the reason)
//   derive(s)    value computed from other settings (kind 'derived': not stored, not set)
// `raw(key)` is the stored preference, unchanged by an override and applying again when it ends; persistence and the menu use it.
//
// Kinds: 'check' and 'select' are in the menu; 'layout' is other remembered state (panes, sizes, appearance). `editorDefault`: default in an editor.
// `restore: false` survives restoring the defaults.

export const SETTING_DEFS = {
  'container-updates':       { kind: 'check', default: false },
  'container-allocations':   { kind: 'check', default: false },
  'container-deallocations': { kind: 'check', default: false },
  'container-snapshots':     { kind: 'check', default: false },
  'container-transfers':     { kind: 'check', default: false },
  'kernel-compiles':         { kind: 'check', default: false },
  'communications':          { kind: 'check', default: true },
  'element-accesses':        { kind: 'check', default: false },
  'anti-deps':               { kind: 'check', default: false },
  'virtual-alias-edges':     { kind: 'check', default: false },
  'edge-labels':             { kind: 'check', default: false },
  'node-labels':             { kind: 'check', default: true },
  'show_regions':            { kind: 'check', default: false,
    override: s => s.get('graph-mode') === 'program-tree' ? { value: true, reason: 'Always on in Program Tree mode' } : null },
  'fusion_analysis':         { kind: 'check', default: false },
  'fusion-ignore-regions':   { kind: 'check', default: false },
  'fusion-ignore-snapshots': { kind: 'check', default: false },

  'graph-mode':      { kind: 'select', default: 'dependence-dag' },
  'node-color-call': { kind: 'select', default: 'duration-total' },
  'node-size-call':  { kind: 'select', default: 'fixed' },
  'direction':       { kind: 'select', default: 'vertical' },
  'edge-width':      { kind: 'select', default: 'fixed' },
  'edge-opacity':    { kind: 'select', default: 'fixed' },
  'aggregate':       { kind: 'select', default: 'none' },   // none | name | name-location

  'right-pane-width': { kind: 'layout', default: 500 },
    'info-pane-ratio':  { kind: 'layout', default: 0 },   // open height as a fraction of the right pane; 0: default height
  'source-open':      { kind: 'layout', default: true, editorDefault: false },
  'gantt-open':       { kind: 'layout', default: false, editorDefault: true },
    'gantt-height':     { kind: 'layout', default: 0 },   // px chosen by the user; 0 fits the trace
    'gantt-scale':      { kind: 'layout', default: 1 },   // 1 fits the trace in the pane
  'legend-open':      { kind: 'layout', default: true },
    'info-enabled':     { kind: 'layout', default: true },   // the info panel opens on selection
  'code-paths':       { kind: 'layout', default: true },
    'pin-definitions':  { kind: 'layout', default: false },   // marks of all definitions stay in the source
    'progress-bar':     { kind: 'layout', default: false },   // turned on when a simulation starts
    'wheel-zoom':       { kind: 'layout', default: false },   // a plain wheel zooms instead of panning
  'color-spectrum':   { kind: 'layout', default: 'default' },
  'hljs-theme':       { kind: 'layout', default: 'default' },
  'dark-mode':        { kind: 'layout', default: false, restore: false },   // an editor's theme decides: theme_vscode.js

  'expand_all_regions': { kind: 'derived', derive: s => !s.get('show_regions') },
  'collapse_iteration': { kind: 'derived', derive: s => s.get('aggregate') !== 'none' },
  'aggregate_by':       { kind: 'derived', derive: s => s.get('aggregate') === 'name-location' ? 'name-location' : 'name' },
  'data_as_edges':      { kind: 'derived', derive: s => !s.get('container-updates') },
};

export class Settings {
  /** `initial`: raw values over the defaults. Keys without a definition are kept as given: readable and settable, not persisted. */
  constructor(defs, initial, { editor = false } = {}) {
    this._defs = defs;
    this._values = {};
    this._listeners = [];
    this._resolving = [];
    this._editor = editor;
    for (const [key, def] of Object.entries(defs)) if (def.kind !== 'derived') this._values[key] = this._default(key, editor);
    Object.assign(this._values, initial || {});
    // property access (`settings['key']`) would give undefined: throws with the call to use
    const isKey = (target, prop) => typeof prop === 'string' && !(prop in target) && (prop in target._defs || prop in target._values);
    return new Proxy(this, {
      get(target, prop, receiver) {
        if (isKey(target, prop)) throw new Error("settings: use get('" + prop + "'), not a property");
        return Reflect.get(target, prop, receiver);
      },
      set(target, prop, value, receiver) {
        if (isKey(target, prop)) throw new Error("settings: use set('" + prop + "', value), not a property");
        return Reflect.set(target, prop, value, receiver);
      },
    });
  }

  get(key) {
    const def = this._defs[key];
    if (def && def.kind === 'derived') return this._resolve(key, () => def.derive(this));
    const forced = this.forced(key);
    return forced ? forced.value : this._values[key];
  }

  raw(key) { return this._values[key]; }

  forced(key) {
    const def = this._defs[key];
    return def && def.override ? this._resolve(key, () => def.override(this)) || null : null;
  }

  lock(key) {
    const forced = this.forced(key);
    return forced ? forced.reason : null;
  }

  /** Stores a preference; listeners hear of keys whose effective value changed. */
  set(key, value) { return this.update({ [key]: value }); }

  /** Stores several preferences with one notification; returns the keys whose effective value changed. */
  update(patch) {
    const before = this._snapshot();
    for (const [key, value] of Object.entries(patch)) {
      const def = this._defs[key];
      if (def && def.kind === 'derived') throw new Error('settings: ' + key + ' is derived and cannot be set');
      this._values[key] = value;
    }
    return this._notify(before);
  }

  /** Flips a check; false and no change while locked. */
  toggle(key) {
    if (this.lock(key)) return false;
    this.set(key, !this.get(key));
    return true;
  }

  restoreDefaults() {
    const patch = {};
    for (const key of this.keys()) if (this._defs[key].restore !== false) patch[key] = this._default(key, this._editor);
    this.update(patch);
  }

  _default(key, editor) {
    const def = this._defs[key];
    return editor && 'editorDefault' in def ? def.editorDefault : def.default;
  }

  /** Stored keys of a kind ('check', 'select', 'layout') or all. */
  keys(kind) {
    return Object.keys(this._defs).filter(key => this._defs[key].kind !== 'derived' && (!kind || this._defs[key].kind === kind));
  }

  persisted() {
    const out = { checkboxes: {}, selects: {}, layout: {} };
    for (const key of this.keys('check')) out.checkboxes[key] = this._values[key];
    for (const key of this.keys('select')) out.selects[key] = this._values[key];
    for (const key of this.keys('layout')) out.layout[key] = this._values[key];
    return out;
  }

  /** `listener(changedKeys)` after an effective value changes; returns a function that removes it. */
  onChange(listener) {
    this._listeners.push(listener);
    return () => { this._listeners = this._listeners.filter(l => l !== listener); };
  }

  _resolve(key, compute) {
    if (this._resolving.includes(key)) throw new Error('settings: ' + [...this._resolving, key].join(' > '));
    this._resolving.push(key);
    try { return compute(); } finally { this._resolving.pop(); }
  }

  _snapshot() {
    const out = {};
    for (const key of Object.keys(this._defs)) out[key] = this.get(key);
    return out;
  }

  _notify(before) {
    const changed = Object.keys(before).filter(key => this.get(key) !== before[key]);
    if (changed.length) for (const listener of this._listeners.slice()) listener(changed);
    return changed;
  }
}

export const appSettings = (initial, options) => new Settings(SETTING_DEFS, initial, options);

/** `settings` as a Settings: one already, or a plain object of raw values (tests). */
export const toSettings = settings => settings instanceof Settings ? settings : appSettings(settings);
