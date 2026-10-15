// test_source_scroll.js — a source pane keeps its scroll position when its tab is not shown (a pane that is display: none loses it), so that the
// smooth scroll to a highlighted line starts from where the pane was, not from the top.
import { describe, it, expect, vi } from 'vitest';

globalThis.requestAnimationFrame = f => f();
const { SourceCodeView } = await import('../../script/SourceCodeView.js');

function pane(name) {
  const classes = new Set();
  return {
    name, classes, _scroll: 0,
    classList: { add: c => classes.add(c), remove: c => classes.delete(c), contains: c => classes.has(c) },
    // (the browser: a pane that is not shown has no scroll position)
    get scrollTop() { return classes.has('active') ? this._scroll : 0; },
    set scrollTop(v) { if (classes.has('active')) this._scroll = v; },
    _hide() { this._scroll = 0; },
  };
}

function view(panes) {
  const v = Object.create(SourceCodeView.prototype);
  const tabs = panes.map(() => ({ classList: { add() {}, remove() {} }, scrollIntoView() {} }));
  v._tabBarEl = { querySelectorAll: () => tabs };
  v._tabPanesEl = { querySelectorAll: s => (s === '.cpp-tab-pane' ? panes : []), querySelector: () => null };
  v._drawBadgeArrows = vi.fn();
  return { v, tabs };
}

describe('switching source tabs', () => {
  it('gives a pane its scroll position back', () => {
    const a = pane('a'), b = pane('b');
    const { v, tabs } = view([a, b]);
    v.switchTab(tabs[0], 0);
    a.scrollTop = 420;
    v.switchTab(tabs[1], 1);
    expect(a.scrollTop).toBe(0);         // (not shown)
    b.scrollTop = 80;
    v.switchTab(tabs[0], 0);
    expect(a.scrollTop).toBe(420);
    v.switchTab(tabs[1], 1);
    expect(b.scrollTop).toBe(80);
  });
  it('leaves a pane that was never scrolled at the top, and the same tab where it is', () => {
    const a = pane('a'), b = pane('b');
    const { v, tabs } = view([a, b]);
    v.switchTab(tabs[1], 1);
    expect(b.scrollTop).toBe(0);
    b.scrollTop = 300;
    v.switchTab(tabs[1], 1);
    expect(b.scrollTop).toBe(300);
  });
});
