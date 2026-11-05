// test_settings_model.js — the settings object (script/settings_model.js): effective values, overrides and locks, derived values, notification, persistence.
import { describe, it, expect } from 'vitest';
import { Settings, SETTING_DEFS, appSettings, toSettings } from '../../script/settings_model.js';

// Definitions of its own, so that the mechanism is tested apart from the viewer's settings.
const defs = {
    mode:    { kind: 'select', default: 'a' },
    x:       { kind: 'check', default: false },
    y:       { kind: 'check', default: false, override: s => s.get('mode') === 'b' ? { value: true, reason: 'forced by mode b' } : null },
    z:       { kind: 'check', default: true },
    not_y:   { kind: 'derived', derive: s => !s.get('y') },
};

describe('values', () => {
    it('start at the defaults, over which the initial values go; unknown keys are kept as given', () => {
        const s = new Settings(defs, { x: true, other: 7 });
        expect([s.get('mode'), s.get('x'), s.get('z'), s.get('other')]).toEqual(['a', true, true, 7]);
    });

    it('are set one at a time or several at once', () => {
        const s = new Settings(defs);
        s.set('mode', 'c');
        s.update({ x: true, z: false });
        expect([s.get('mode'), s.get('x'), s.get('z')]).toEqual(['c', true, false]);
    });

    it('are the stored preferences, in the form that is saved, and are restored to the defaults', () => {
        const s = new Settings(defs, { x: true, other: 7 });
        expect(s.persisted()).toEqual({ checkboxes: { x: true, y: false, z: true }, selects: { mode: 'a' }, layout: {} });
        s.update({ mode: 'c', z: false });
        s.restoreDefaults();
        expect(s.persisted()).toEqual({ checkboxes: { x: false, y: false, z: true }, selects: { mode: 'a' }, layout: {} });
    });
});

describe('an override', () => {
    it('gives the forced value to every reader and the reason as a lock, while the preference is kept', () => {
        const s = new Settings(defs, { y: false });
        expect([s.get('y'), s.lock('y')]).toEqual([false, null]);
        s.set('mode', 'b');
        expect([s.get('y'), s.raw('y'), s.lock('y')]).toEqual([true, false, 'forced by mode b']);
        s.set('mode', 'a');
        expect([s.get('y'), s.lock('y')]).toEqual([false, null]);
    });

    it('is respected by a toggle: a locked setting does not change, an unlocked one flips', () => {
        const s = new Settings(defs);
        expect(s.toggle('x')).toBe(true);
        expect(s.get('x')).toBe(true);
        s.set('mode', 'b');
        expect(s.toggle('y')).toBe(false);
        expect([s.get('y'), s.raw('y')]).toEqual([true, false]);
    });

    it('does not store a changed preference over the forced value, which returns when the override ends', () => {
        const s = new Settings(defs);
        s.set('mode', 'b');
        s.set('y', true);
        s.set('mode', 'a');
        expect(s.get('y')).toBe(true);
    });

    it('is persisted as the preference, not the forced value', () => {
        const s = new Settings(defs);
        s.set('mode', 'b');
        expect(s.persisted().checkboxes.y).toBe(false);
    });
});

describe('a derived value', () => {
    it('follows the effective values it is made from', () => {
        const s = new Settings(defs);
        expect(s.get('not_y')).toBe(true);
        s.set('mode', 'b');
        expect(s.get('not_y')).toBe(false);
    });

    it('is not stored, and cannot be set', () => {
        const s = new Settings(defs);
        expect(s.keys()).not.toContain('not_y');
        expect(() => s.set('not_y', true)).toThrow(/derived/);
    });
});

describe('notification', () => {
    it('names the keys whose effective value changed, those that follow an override and derived ones included', () => {
        const s = new Settings(defs);
        const heard = [];
        s.onChange(keys => heard.push(keys.slice().sort()));
        s.set('mode', 'b');
        s.set('x', true);
        expect(heard).toEqual([['mode', 'not_y', 'y'], ['x']]);
    });

    it('is silent when nothing changed, once for an update of several, and stops after the listener is removed', () => {
        const s = new Settings(defs);
        const heard = [];
        const off = s.onChange(keys => heard.push(keys));
        s.set('x', false);
        s.update({ x: true, z: false });
        expect(heard).toHaveLength(1);
        off();
        s.set('x', false);
        expect(heard).toHaveLength(1);
    });

    it('does not name a key whose preference changed under an override', () => {
        const s = new Settings(defs);
        s.set('mode', 'b');
        const heard = [];
        s.onChange(keys => heard.push(keys));
        s.set('y', true);
        expect(heard).toEqual([]);
    });
});

describe('access by property', () => {
    it('is an error, in both directions, with the call to use instead', () => {
        const s = new Settings(defs, { other: 7 });
        expect(() => s['mode']).toThrow(/use get\('mode'\)/);
        expect(() => s['other']).toThrow(/use get\('other'\)/);
        expect(() => { s['mode'] = 'c'; }).toThrow(/use set\('mode'/);
        expect(s.get('mode')).toBe('a');
    });

    it('leaves what is not a setting alone: methods, other properties and the checks of a promise or a printer', () => {
        const s = new Settings(defs);
        expect([typeof s.get, s.then, s.toJSON, typeof s[Symbol.iterator]]).toEqual(['function', undefined, undefined, 'undefined']);
        expect(() => JSON.stringify(s)).not.toThrow();
    });
});

describe('definitions', () => {
    it('with a cycle are reported with the chain', () => {
        const cyclic = { a: { kind: 'derived', derive: s => s.get('b') }, b: { kind: 'derived', derive: s => s.get('a') } };
        expect(() => new Settings(cyclic).get('a')).toThrow(/a > b > a/);
    });

    it('of the viewer: Program Tree mode shows regions and the regions are then not collapsed by default', () => {
        const s = appSettings();
        expect([s.get('show_regions'), s.get('expand_all_regions'), s.lock('show_regions')]).toEqual([false, true, null]);
        s.set('graph-mode', 'program-tree');
        expect([s.get('show_regions'), s.get('expand_all_regions'), s.lock('show_regions')]).toEqual([true, false, 'Always on in Program Tree mode']);
        s.set('graph-mode', 'dependence-dag');
        expect(s.get('show_regions')).toBe(false);
    });

    it('of the viewer: the updates of a container are edges unless they are shown as nodes', () => {
        const s = appSettings();
        expect(s.get('data_as_edges')).toBe(true);
        s.set('container-updates', true);
        expect(s.get('data_as_edges')).toBe(false);
    });

    it('of the viewer: snapshots are off and communications on, until the user says otherwise', () => {
        const s = appSettings();
        expect([s.get('container-snapshots'), s.get('communications')]).toEqual([false, true]);
    });

    it('of the viewer have a default for every stored key, and a kind', () => {
        for (const [key, def] of Object.entries(SETTING_DEFS)) {
            expect(['check', 'select', 'layout', 'derived'], key).toContain(def.kind);
            if (def.kind !== 'derived') expect(def.default, key).not.toBeUndefined();
        }
    });
});

describe('layout settings', () => {
    it('are persisted as layout, with the same defaults for a fresh page and for Restore Defaults', () => {
        const s = appSettings();
        expect([s.get('legend-open'), s.get('gantt-open'), s.get('source-open')]).toEqual([true, false, true]);
        s.update({ 'legend-open': false, 'right-pane-width': 321 });
        expect(s.persisted().layout['right-pane-width']).toBe(321);
        s.restoreDefaults();
        expect([s.get('legend-open'), s.get('right-pane-width')]).toEqual([true, 500]);
    });

    it('have the default of an editor where there is one, also after Restore Defaults', () => {
        const s = appSettings(undefined, { editor: true });
        expect([s.get('source-open'), s.get('gantt-open')]).toEqual([false, true]);
        s.set('source-open', true);
        s.restoreDefaults();
        expect(s.get('source-open')).toBe(false);
    });

    it('can be kept when the defaults are restored', () => {
        const s = appSettings({ 'dark-mode': true });
        s.restoreDefaults();
        expect(s.get('dark-mode')).toBe(true);
    });
});

describe('toSettings', () => {
    it('takes a Settings as it is, and makes one of a plain object of raw values', () => {
        const s = appSettings();
        expect(toSettings(s)).toBe(s);
        const wrapped = toSettings({ 'graph-mode': 'program-tree', 'node-labels': false });
        expect([wrapped.get('graph-mode'), wrapped.get('node-labels'), wrapped.get('show_regions')]).toEqual(['program-tree', false, true]);
        expect(toSettings(undefined).get('graph-mode')).toBe('dependence-dag');
    });
});

describe('aggregation', () => {
    it('is one setting, from which the two options of the backend are derived', () => {
        const s = appSettings();
        expect([s.get('aggregate'), s.get('collapse_iteration'), s.get('aggregate_by')]).toEqual(['none', false, 'name']);
        s.set('aggregate', 'name');
        expect([s.get('collapse_iteration'), s.get('aggregate_by')]).toEqual([true, 'name']);
        s.set('aggregate', 'name-location');
        expect([s.get('collapse_iteration'), s.get('aggregate_by')]).toEqual([true, 'name-location']);
    });
});
