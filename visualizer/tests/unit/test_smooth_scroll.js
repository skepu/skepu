// test_smooth_scroll.js — the scroll of a pane to a selected line (script/smooth_scroll.js): from where the pane is, in a time that grows a little with
// the distance, taken over by the next one from where it is, and ended by the user.
import { describe, it, expect } from 'vitest';
import { scrollDuration, ease, smoothScrollTo, cancelScroll, MIN_MS, MAX_MS } from '../../script/smooth_scroll.js';

// a pane and a clock: frames are run by hand
function setup(scrollTop = 0, scrollHeight = 2000, clientHeight = 400) {
  const listeners = {};
  const pane = {
    scrollTop, scrollHeight, clientHeight,
    addEventListener: (t, f) => { (listeners[t] ||= new Set()).add(f); },
    removeEventListener: (t, f) => { listeners[t]?.delete(f); },
    fire: t => [...(listeners[t] || [])].forEach(f => f()),
    listeners,
  };
  let time = 0, frames = [], timers = [];
  const env = {
    now: () => time, raf: f => frames.push(f),
    setTimeout: (f, ms) => { const t = { f, at: time + ms }; timers.push(t); return t; },
    clearTimeout: t => { timers = timers.filter(x => x !== t); },
  };
  const run = ms => {   // advance the clock, running the frames (each 16 ms) and the timers that are due
    const end = time + ms;
    while (time < end) {
      time = Math.min(end, time + 16);
      const f = frames; frames = []; f.forEach(x => x());
      const due = timers.filter(t => t.at <= time); timers = timers.filter(t => t.at > time); due.forEach(t => t.f());
    }
  };
  return { pane, env, run };
}

describe('the duration', () => {
  it('grows with the distance, between the least and the most', () => {
    expect(scrollDuration(0)).toBe(MIN_MS);
    expect(scrollDuration(300)).toBeGreaterThan(scrollDuration(50));
    expect(scrollDuration(100000)).toBe(MAX_MS);
    expect(scrollDuration(-300)).toBe(scrollDuration(300));
  });
  it('has an easing from 0 to 1, slow at the ends', () => {
    expect(ease(0)).toBe(0); expect(ease(1)).toBe(1); expect(ease(0.5)).toBeCloseTo(0.5, 5);
    expect(ease(0.1)).toBeLessThan(0.1); expect(ease(0.9)).toBeGreaterThan(0.9);
    expect(ease(-1)).toBe(0); expect(ease(2)).toBe(1);
  });
});

describe('the scroll of a pane', () => {
  it('goes from where the pane is to the new place, down or up, without a visit to the top', () => {
    for (const [from, to] of [[300, 900], [900, 300], [1500, 20]]) {
      const { pane, env, run } = setup(from);
      const seen = [];
      smoothScrollTo(pane, to, {}, env);
      for (let i = 0; i < 40; i++) { run(16); seen.push(pane.scrollTop); }
      expect(pane.scrollTop).toBe(to);
      const lo = Math.min(from, to), hi = Math.max(from, to);
      for (const s of seen) { expect(s).toBeGreaterThanOrEqual(lo - 1e-9); expect(s).toBeLessThanOrEqual(hi + 1e-9); }
      // (monotonic: it moves one way)
      for (let i = 1; i < seen.length; i++) expect(to > from ? seen[i] >= seen[i - 1] : seen[i] <= seen[i - 1]).toBe(true);
    }
  });
  it('is kept inside the pane', () => {
    const { pane, env, run } = setup(100);
    smoothScrollTo(pane, 99999, {}, env);
    run(1000);
    expect(pane.scrollTop).toBe(1600);
    smoothScrollTo(pane, -50, {}, env);
    run(1000);
    expect(pane.scrollTop).toBe(0);
  });
  it('jumps when it is not to animate, or there is nothing to move', () => {
    const { pane, env } = setup(100);
    smoothScrollTo(pane, 700, { animate: false }, env);
    expect(pane.scrollTop).toBe(700);
    smoothScrollTo(pane, 700.4, {}, env);
    expect(pane.scrollTop).toBeCloseTo(700.4, 5);   // (less than a pixel away: nothing to animate)
    expect(pane._scrollAnim).toBeFalsy();
  });
  it('is taken over by the next one, from where the pane is at that moment', () => {
    const { pane, env, run } = setup(0);
    smoothScrollTo(pane, 1000, {}, env);
    run(128);
    const mid = pane.scrollTop;
    expect(mid).toBeGreaterThan(0); expect(mid).toBeLessThan(1000);
    smoothScrollTo(pane, 0, {}, env);
    run(16);
    expect(pane.scrollTop).toBeLessThanOrEqual(mid);   // (it turns from mid, it does not jump to the top or to 1000 first)
    run(1000);
    expect(pane.scrollTop).toBe(0);
  });
  it('is not thrown off by a reset of the position in the middle: it goes on from its own path', () => {
    const { pane, env, run } = setup(600);
    smoothScrollTo(pane, 1200, {}, env);
    run(96);
    pane.scrollTop = 0;   // (something resets the pane)
    run(16);
    expect(pane.scrollTop).toBeGreaterThan(600);   // the next frame puts it back on the path
    run(1000);
    expect(pane.scrollTop).toBe(1200);
  });
  it('ends where the line is also if no frames are run (a page that is hidden)', () => {
    const { pane, run } = setup(0);
    const env = { now: () => 0, raf: () => {}, setTimeout: (f, ms) => { setTimeout(f, 0); return 0; }, clearTimeout() {} };
    smoothScrollTo(pane, 800, {}, env);
    return new Promise(r => setTimeout(r, 10)).then(() => expect(pane.scrollTop).toBe(800));
  });
  it('is ended by the wheel, touch or keys of the user, and leaves no listeners', () => {
    const { pane, env, run } = setup(0);
    smoothScrollTo(pane, 1000, {}, env);
    run(96);
    const at = pane.scrollTop;
    pane.fire('wheel');
    run(500);
    expect(pane.scrollTop).toBe(at);
    expect([...Object.values(pane.listeners)].every(s => s.size === 0)).toBe(true);
  });
  it('can be cancelled', () => {
    const { pane, env, run } = setup(0);
    smoothScrollTo(pane, 1000, {}, env);
    run(48);
    cancelScroll(pane);
    const at = pane.scrollTop;
    run(500);
    expect(pane.scrollTop).toBe(at);
  });
});

describe('the default environment', () => {
  it('works with the timer and frame functions of a page, which throw when they are called as methods of another object', async () => {
    const calls = [];
    const strict = name => function(...args) { if (this !== undefined && this !== globalThis) throw new TypeError('Illegal invocation'); calls.push(name); return globalThis['__real_' + name](...args); };
    globalThis.__real_setTimeout = globalThis.setTimeout; globalThis.__real_clearTimeout = globalThis.clearTimeout;
    const saved = { setTimeout: globalThis.setTimeout, clearTimeout: globalThis.clearTimeout, raf: globalThis.requestAnimationFrame };
    globalThis.setTimeout = strict('setTimeout'); globalThis.clearTimeout = strict('clearTimeout');
    globalThis.requestAnimationFrame = f => globalThis.__real_setTimeout(() => f(), 0);
    try {
      const { pane } = setup(0);
      smoothScrollTo(pane, 800, {});
      await new Promise(r => globalThis.__real_setTimeout(r, 700));
      expect(pane.scrollTop).toBe(800);
      expect(calls).toContain('setTimeout');
    } finally {
      globalThis.setTimeout = saved.setTimeout; globalThis.clearTimeout = saved.clearTimeout; globalThis.requestAnimationFrame = saved.raf;
    }
  });
});

import { linesInView } from '../../script/smooth_scroll.js';

describe('whether the lines are in view already', () => {
  const view = { top: 100, bottom: 600 };   // 500 px high: the margin is 48 px
  const line = (top, h = 12) => ({ top, bottom: top + h });
  it('is yes for a line well inside the pane', () => {
    expect(linesInView([line(300)], line(300), view)).toBe(true);
  });
  it('is no for a line outside, or at the very edge (cut off, or too close to it)', () => {
    expect(linesInView([line(700)], line(700), view)).toBe(false);
    expect(linesInView([line(40)], line(40), view)).toBe(false);
    expect(linesInView([line(104)], line(104), view)).toBe(false);      // (4 px from the top)
    expect(linesInView([line(590)], line(590), view)).toBe(false);      // (cut off at the bottom)
    expect(linesInView([line(100 + 48)], line(100 + 48), view)).toBe(true);   // (just inside the margin)
  });
  it('needs all of a range of lines that fits in the pane', () => {
    const range = [line(300), line(312), line(560)];
    expect(linesInView(range, range[0], view)).toBe(false);   // (the last one is within the margin of the bottom)
    expect(linesInView([line(300), line(330)], line(300), view)).toBe(true);
  });
  it('needs only the target line of a range that is more than the pane shows', () => {
    const tall = [line(150), line(900)];
    expect(linesInView(tall, tall[0], view)).toBe(true);
    expect(linesInView(tall, tall[1], view)).toBe(false);
  });
  it('is no for a pane that is not shown (no height), and for no lines', () => {
    expect(linesInView([line(300)], line(300), { top: 0, bottom: 0 })).toBe(false);
    expect(linesInView([], line(300), view)).toBe(false);
  });
});
