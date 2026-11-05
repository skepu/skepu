// test_settings_source_view.js — the options of the source view that are kept in State (Code Paths, Definitions) are given to the view when the
// settings are loaded and when the defaults are restored.
import { describe, it, expect, beforeEach } from 'vitest';

const store = {};
const element = () => {
    const e = { style: {}, value: '', textContent: '', dataset: {}, classList: { toggle() {}, add() {}, remove() {}, contains: () => false } };
    e.querySelector = () => element();
    return e;
};
const elements = {};   // the elements that a test gives to getElementById (the others are any element)
globalThis.window = globalThis;
globalThis.localStorage = { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } };
globalThis.document = {
    querySelector: () => null, querySelectorAll: () => [], getElementById: id => elements[id] ?? element(),
    documentElement: { classList: { toggle() {}, contains: () => false } },
};

const { State } = await import('../../script/state.js');
const { loadSettings, saveSettings, menuRestoreDefaults } = await import('../../script/settings.js');
const KEY = (await import('../../script/appearance.js')).SETTINGS_KEY;

let view;
beforeEach(() => {
    for (const k of Object.keys(store)) delete store[k];
    view = { codePaths: true, pin: false, setCodePaths(v) { this.codePaths = !!v; }, setPinDefinitions(v) { this.pin = !!v; } };
    State.sourceView = view;
    State.settings.set('code-paths', true);
    State.settings.set('pin-definitions', false);
});

describe('Code Paths and Definitions after the page is loaded', () => {
    it('are given to the source view when they are read from the stored settings', () => {
        store[KEY] = JSON.stringify({ layout: { 'code-paths': false, 'pin-definitions': true } });
        loadSettings();
        expect(State.settings.get('code-paths')).toBe(false);
        expect(State.settings.get('pin-definitions')).toBe(true);
        expect(view.codePaths).toBe(false);   // it was left at true: the arrows were drawn while the menu said no
        expect(view.pin).toBe(true);
    });

    it('keep what was saved: save, load again, the same', () => {
        State.settings.set('code-paths', false);
        State.settings.set('pin-definitions', true);
        saveSettings();
        State.settings.set('code-paths', true);
        State.settings.set('pin-definitions', false);
        loadSettings();
        expect([State.settings.get('code-paths'), State.settings.get('pin-definitions'), view.codePaths, view.pin]).toEqual([false, true, false, true]);
        expect(JSON.parse(store[KEY]).layout['code-paths']).toBe(false);
    });

    it('stay as they are (the defaults) without stored settings', () => {
        loadSettings();
        expect([State.settings.get('code-paths'), view.codePaths, State.settings.get('pin-definitions'), view.pin]).toEqual([true, true, false, false]);
    });
});

describe('Restore Defaults', () => {
    it('also restores Code Paths and Definitions, in State, in the view and in what is stored', () => {
        State.settings.set('code-paths', false);
        State.settings.set('pin-definitions', true);
        view.codePaths = false;
        view.pin = true;
        menuRestoreDefaults();
        expect([State.settings.get('code-paths'), State.settings.get('pin-definitions')]).toEqual([true, false]);
        expect([view.codePaths, view.pin]).toEqual([true, false]);
        const layout = JSON.parse(store[KEY]).layout;
        expect([layout['code-paths'], layout['pin-definitions']]).toEqual([true, false]);
    });
});

describe('the width of the source pane that is stored', () => {
    it('is the width that the pane has (its flex-basis), also while the pane slides open and has no width yet', () => {
        const right = element();
        right.offsetWidth = 0;                       // a pane that has just begun to slide open
        right.style.flex = '0 0 420px';
        elements.right_pane = right;
        State.settings.set('source-open', true);
        State._traceLoaded = true;
        saveSettings();
        expect(JSON.parse(store[KEY]).layout['right-pane-width']).toBe(420);
        delete elements.right_pane;
    });

    it('is the offsetWidth of a pane that has no flex-basis set', () => {
        const right = element();
        right.offsetWidth = 377;
        elements.right_pane = right;
        State.settings.set('source-open', true);
        State._traceLoaded = true;
        saveSettings();
        expect(JSON.parse(store[KEY]).layout['right-pane-width']).toBe(377);
        delete elements.right_pane;
    });
});
