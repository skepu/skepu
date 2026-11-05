// test_settings_menu.js — settings in the page: the menu follows effective values and disables a locked item with its reason; shortcuts respect the lock;
// a stored preference returns when the lock ends; backend requests use effective values.
import { describe, it, expect, beforeEach } from 'vitest';

const store = {};
const element = () => {
    const e = { style: {}, value: '', textContent: '', dataset: {}, classList: { toggle() {}, add() {}, remove() {}, contains: () => false } };
    e.querySelector = () => element();
    return e;
};
const regionsItem = { disabled: false, title: '', dataset: { ctrl: 'show_regions' }, icon: { textContent: '' }, querySelector() { return this.icon; } };
globalThis.window = globalThis;
globalThis.localStorage = { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } };
globalThis.document = {
    querySelector: s => (s === '.menu-check[data-ctrl="show_regions"]' ? regionsItem : null),
    querySelectorAll: () => [], getElementById: () => element(),
    documentElement: { classList: { toggle() {}, contains: () => false } },
};

const { State } = await import('../../script/state.js');
const { toggleSetting, menuSelectRadio, menuRestoreDefaults, loadSettings, saveSettings, syncSettingsMenu } = await import('../../script/settings.js');
const { TraceDataProvider } = await import('../../script/TraceDataProvider.js');
const KEY = (await import('../../script/appearance.js')).SETTINGS_KEY;

const mode = value => menuSelectRadio({ dataset: { ctrl: 'graph-mode', value } });

describe('the request to the backend', () => {
    const request = async settings => {
        const requests = [];
        const provider = new TraceDataProvider({ fetchGraph: async p => { requests.push(p); return { nodes: [], edges: [], timeline_nodes: [] }; }, settings });
        await provider.fetch();
        return requests[0];
    };

    it('expands every region only when they are not shown', async () => {
        expect((await request({ 'graph-mode': 'dependence-dag', show_regions: false })).expand_all_regions).toBe(true);
        expect((await request({ 'graph-mode': 'dependence-dag', show_regions: true })).expand_all_regions).toBe(false);
        expect((await request({ 'graph-mode': 'program-tree', show_regions: false })).expand_all_regions).toBe(false);
    });

    it('has the updates of a container as edges unless they are shown as nodes', async () => {
        expect((await request({})).data_as_edges).toBe(true);
        expect((await request({ 'container-updates': true })).data_as_edges).toBe(false);
    });
});

describe('the Regions menu item and its shortcut', () => {
    beforeEach(() => {
        for (const k of Object.keys(store)) delete store[k];
        State.settings.restoreDefaults();
        Object.assign(regionsItem, { disabled: false, title: '' });
        regionsItem.icon.textContent = '';
        delete regionsItem.dataset.locked;
    });

    it('is checked and disabled in Program Tree mode, with the reason, and is as it was after leaving it', () => {
        mode('program-tree');
        expect([regionsItem.disabled, regionsItem.icon.textContent, regionsItem.title]).toEqual([true, '✓', 'Always on in Program Tree mode']);
        mode('dependence-dag');
        expect([regionsItem.disabled, regionsItem.icon.textContent, regionsItem.title]).toEqual([false, '', '']);
    });

    it('keeps the stored choice: the shortcut does nothing in Program Tree mode, and the choice returns afterwards', () => {
        toggleSetting('show_regions');
        expect(State.settings.raw('show_regions')).toBe(true);
        mode('program-tree');
        toggleSetting('show_regions');
        expect(State.settings.raw('show_regions')).toBe(true);
        mode('dependence-dag');
        expect([State.settings.get('show_regions'), regionsItem.icon.textContent]).toEqual([true, '✓']);
        toggleSetting('show_regions');
        expect([State.settings.get('show_regions'), regionsItem.icon.textContent]).toEqual([false, '']);
    });

    it('is enabled again by Restore Defaults', () => {
        mode('program-tree');
        menuRestoreDefaults();
        expect([State.settings.get('graph-mode'), regionsItem.disabled]).toEqual(['dependence-dag', false]);
    });

    it('is as stored when the settings are loaded, and the stored preference is the one saved, not the forced value', () => {
        store[KEY] = JSON.stringify({ checkboxes: { show_regions: false }, selects: { 'graph-mode': 'program-tree' } });
        loadSettings();
        expect([State.settings.get('show_regions'), State.settings.raw('show_regions'), regionsItem.disabled, regionsItem.icon.textContent]).toEqual([true, false, true, '✓']);
        saveSettings();
        expect(JSON.parse(store[KEY]).checkboxes.show_regions).toBe(false);
    });

    it('does not touch the title of an item that is not locked by a rule', () => {
        regionsItem.title = 'a title of its own';
        syncSettingsMenu();
        expect(regionsItem.title).toBe('a title of its own');
    });
});

describe('Restore Defaults', () => {
    it('opens the legend and turns snapshots off', () => {
        State.settings.set('legend-open', false);
        State.settings.set('container-snapshots', true);
        menuRestoreDefaults();
        expect([State.settings.get('legend-open'), State.settings.get('container-snapshots')]).toEqual([true, false]);
    });
});

describe('a stored choice that the menu no longer offers', () => {
    it('is dropped on load, and the default stays', () => {
        const items = ['duration-total', 'elements'].map(value => ({ dataset: { value }, querySelector: () => ({}) }));
        const real = document.querySelectorAll;
        document.querySelectorAll = sel => (sel === '.menu-radio[data-ctrl="node-color-call"]' ? items : []);
        store[KEY] = JSON.stringify({ selects: { 'node-color-call': 'fan-in' } });
        loadSettings();
        document.querySelectorAll = real;
        expect(State.settings.get('node-color-call')).toBe('duration-total');
    });
});

describe('the items for an MPI program', () => {
    const item = (cls, ctrl, value) => ({ hidden: false, dataset: { ctrl, value }, classList: { contains: c => c === cls } });
    const withItems = (items, run) => {
        const real = document.querySelectorAll;
        document.querySelectorAll = sel => (sel === '[data-mpi]' ? items : []);
        try { run(); } finally { document.querySelectorAll = real; }
    };

    it('are hidden for a trace without ranks, and shown for one with ranks', () => {
        const imbalance = item('menu-radio', 'node-color-call', 'imbalance'), comms = item('menu-check', 'communications');
        State._isMpi = false;
        withItems([imbalance, comms], () => { syncSettingsMenu(); });
        expect([imbalance.hidden, comms.hidden]).toEqual([true, true]);
        State._isMpi = true;
        withItems([imbalance, comms], () => { syncSettingsMenu(); });
        expect([imbalance.hidden, comms.hidden]).toEqual([false, false]);
    });

    it('keep the colour that is chosen shown', () => {
        const imbalance = item('menu-radio', 'node-color-call', 'imbalance');
        State._isMpi = false;
        State.settings.set('node-color-call', 'imbalance');
        withItems([imbalance], () => { syncSettingsMenu(); });
        expect(imbalance.hidden).toBe(false);
        State.settings.set('node-color-call', 'duration-total');
        withItems([imbalance], () => { syncSettingsMenu(); });
        expect(imbalance.hidden).toBe(true);
    });
});

describe('stored settings from before the Aggregate setting', () => {
    const load = stored => { store[KEY] = JSON.stringify(stored); State.settings.set('aggregate', 'none'); loadSettings(); return State.settings.get('aggregate'); };

    it('become it', () => {
        expect(load({ checkboxes: { collapse_iteration: true }, selects: { aggregate_by: 'name-location' } })).toBe('name-location');
        expect(load({ checkboxes: { collapse_iteration: true }, selects: {} })).toBe('name');
        expect(load({ checkboxes: { collapse_iteration: false }, selects: { aggregate_by: 'name-location' } })).toBe('none');
    });
});
