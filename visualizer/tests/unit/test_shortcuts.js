// test_shortcuts.js — the dialog of the keyboard shortcuts (script/shortcuts.js): made from the menus, so what the keyboard does and what the menus
// say must be the same: every key that the page handles is in a menu, and every key of a menu is handled.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { groupShortcuts, EXTRA_SHORTCUTS } from '../../script/shortcuts.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const page = readFileSync(join(ROOT, 'index.html'), 'utf8');
const ui = readFileSync(join(ROOT, 'script/ui.js'), 'utf8');

const menuKeys = [...page.matchAll(/<span class="menu-key">([^<]+)<\/span>/g)].map(m => m[1]);
const handledKeys = [...ui.matchAll(/event\.key === "([^"]+)"/g)].map(m => m[1]);

describe('the groups of the dialog', () => {
  it('keep the order of first appearance', () => {
    const g = groupShortcuts([
      { group: 'View', keys: 'G', label: 'Timeline' }, { group: 'Nodes', keys: 'T', label: 'Transfers' }, { group: 'View', keys: 'S', label: 'Source' },
    ]);
    expect(g.map(x => x.group)).toEqual(['View', 'Nodes']);
    expect(g[0].entries.map(e => e.keys)).toEqual(['G', 'S']);
  });
  it('have the keys that no menu has', () => {
    expect(EXTRA_SHORTCUTS.some(e => e.keys === '?')).toBe(true);
    expect(EXTRA_SHORTCUTS.some(e => e.keys === 'Esc')).toBe(true);
  });
});

describe('the keys of the page and of the menus', () => {
  it('are found', () => {
    expect(menuKeys.length).toBeGreaterThan(10);
    expect(handledKeys.length).toBeGreaterThan(10);
  });
  it('every key of a menu item is handled by the keyboard', () => {
    const handled = new Set(handledKeys.map(k => (k === ' ' ? 'space' : k.toLowerCase())));
    for (const key of menuKeys) expect(handled.has(key.toLowerCase()), `${key} is in a menu and not handled`).toBe(true);
  });
  it('every key that the keyboard handles has a menu item (or is in the extra list)', () => {
    const inMenus = new Set(menuKeys.map(k => k.toLowerCase()));
    const extra = new Set(EXTRA_SHORTCUTS.flatMap(e => (e.handles ?? [e.keys]).map(k => k.toLowerCase())));
    for (const key of handledKeys) {
      const name = key === ' ' ? 'space' : key.toLowerCase();
      expect(inMenus.has(name) || extra.has(name), `${key} is handled and not in a menu or the extra list`).toBe(true);
    }
  });
});

describe('the markup', () => {
  it('has the ? button to the left of the ⓘ, and the dialog', () => {
    const help = page.indexOf('menu-help-btn'), info = page.indexOf('menu-info-btn');
    expect(help).toBeGreaterThan(0);
    expect(help).toBeLessThan(info);
    expect(page).toContain('data-click="showShortcuts"');
    expect(page).toContain('id="shortcuts-modal"');
  });
});
