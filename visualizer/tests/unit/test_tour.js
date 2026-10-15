// test_tour.js — the steps of the guided tour (script/tour.js) point at things that are in the page, say something, and are not repeated.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const page = readFileSync(join(ROOT, 'index.html'), 'utf8');
const source = readFileSync(join(ROOT, 'script/tour.js'), 'utf8');

// the steps are data: read them from the source, as the text between STEPS = [ and the closing ];
const block = source.slice(source.indexOf('export const STEPS = ['), source.indexOf('// ── The tour'));
const ids = [...block.matchAll(/\{ id: '([^']+)'/g)].map(m => m[1]);
const selectors = [...block.matchAll(/target: '([^']+)'/g)].map(m => m[1]);
const menus = [...block.matchAll(/dropdownOf\('([^']+)'\)/g)].map(m => m[1]);
const texts = [...block.matchAll(/\n\s{4}text: '((?:[^'\\]|\\.)*)'/g)].map(m => m[1]);
const tasks = [...block.matchAll(/task: \{ text: '((?:[^'\\]|\\.)*)', listen: (\w+)\(/g)].map(m => ({ text: m[1], listen: m[2] }));

describe('the steps of the tour', () => {
  it('are there, with different ids', () => {
    expect(ids.length).toBeGreaterThanOrEqual(6);
    expect(new Set(ids).size).toBe(ids.length);
  });
  it('have a text each, that is short', () => {
    expect(texts.length).toBe(ids.length);
    for (const t of texts) { expect(t.length).toBeGreaterThan(20); expect(t.length).toBeLessThan(260); }
  });
  it('ask the user to try what they show, in the same words', () => {
    expect(tasks.length).toBeGreaterThanOrEqual(3);
    for (const t of tasks) {
      expect(t.text.startsWith('Try it: ')).toBe(true);
      expect(['onSelect', 'onWheel']).toContain(t.listen);
    }
  });
  it('point at ids and classes that the page has', () => {
    for (const s of selectors) {
      const name = s.slice(1);
      const pattern = s[0] === '#' ? new RegExp(`id="${name}"`) : new RegExp(`class="[^"]*\\b${name}\\b`);
      expect(pattern.test(page), `${s} is not in index.html`).toBe(true);
    }
  });
  it('open menus that the page has', () => {
    expect(menus.length).toBeGreaterThan(0);
    for (const m of menus) expect(page.includes(`<button class="menu-title">${m}</button>`), `no menu ${m}`).toBe(true);
  });
  it('have an action in the markup that starts them', () => {
    expect(page).toContain('data-click="startTour"');
  });
});
