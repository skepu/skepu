// test_sim_selection.js — the arrow keys with a node selected and no simulation (SimulationEngine.stepFromSelection): the simulation starts, paused, at
// the selection and takes one step; nothing is done if the point is not unique.
import { describe, it, expect } from 'vitest';
import { SimulationEngine } from '../../script/simulation.js';

function setup(nodes) {
  const listeners = {}, selects = [];
  const provider = {
    lastGraphNodes: nodes,
    on(name, f) { (listeners[name] ||= []).push(f); return this; },
    _emit(name, ev) { if (name === 'select') selects.push(ev); (listeners[name] || []).forEach(f => f(ev)); },
  };
  const engine = new SimulationEngine(provider, {});
  return { engine, provider, selects };
}
const node = (id, order, extra = {}) => ({ data: { id, type: 'skeleton_call', total_order: order, trace_index: order, ...extra } });
const three = () => [node('a', 100), node('b', 200), node('c', 300)];
const select = (provider, ev) => provider._emit('select', { source: 'graph', allIds: [ev.nodeId], ...ev });
const shown = selects => selects.filter(s => s.source === 'simulation').map(s => s.nodeId);

describe('the arrow keys from a selected node', () => {
  it('start a paused simulation one step from the selection', () => {
    const { engine, provider, selects } = setup(three());
    select(provider, { nodeId: 'b' });
    expect(engine.stepFromSelection(+1)).toBe(true);
    expect(shown(selects)).toEqual(['c']);
    expect(engine.isActive).toBe(true);
    expect(engine._running).toBe(false);   // paused, as when the arrow keys are used while it plays
    expect(engine._paused).toBe(true);
  });
  it('go back with the other key, and go on from there with the usual steps', () => {
    const { engine, provider, selects } = setup(three());
    select(provider, { nodeId: 'b' });
    expect(engine.stepFromSelection(-1)).toBe(true);
    engine.step(+1);
    engine.step(+1);
    expect(shown(selects)).toEqual(['a', 'b', 'c']);
  });
  it('do nothing at the ends of the sequence, and do not start a simulation', () => {
    const { engine, provider } = setup(three());
    select(provider, { nodeId: 'c' });
    expect(engine.stepFromSelection(+1)).toBe(false);
    expect(engine.isActive).toBe(false);
    select(provider, { nodeId: 'a' });
    expect(engine.stepFromSelection(-1)).toBe(false);
  });
  it('do nothing without a selection, or after it was cleared or the graph was drawn again', () => {
    const { engine, provider } = setup(three());
    expect(engine.stepFromSelection(+1)).toBe(false);
    select(provider, { nodeId: 'a' }); provider._emit('clear');
    expect(engine.stepFromSelection(+1)).toBe(false);
    select(provider, { nodeId: 'a' }); provider._emit('fetchstart');
    expect(engine.stepFromSelection(+1)).toBe(false);
  });
  it('need a unique point: a node of iterations selected as a whole is not, one iteration of it is', () => {
    const iterated = node('i', 100, { total_orders: [100, 300, 500], trace_indices: [1, 3, 5] });
    const { engine, provider, selects } = setup([iterated, node('x', 200), node('y', 400)]);
    select(provider, { nodeId: 'i' });
    expect(engine.stepFromSelection(+1)).toBe(false);
    select(provider, { nodeId: 'i', iterIdx: 1 });
    expect(engine.stepFromSelection(+1)).toBe(true);
    expect(shown(selects)).toEqual(['y']);
  });
  it('need a unique node: a gutter badge of several nodes is not', () => {
    const { engine, provider } = setup(three());
    provider._emit('select', { source: 'code-listing', nodeId: 'a', focusId: 'a', allIds: ['a', 'b'] });
    expect(engine.stepFromSelection(+1)).toBe(false);
  });
  it('do nothing for a node that is not drawn', () => {
    const { engine, provider } = setup(three());
    select(provider, { nodeId: 'gone' });
    expect(engine.stepFromSelection(+1)).toBe(false);
  });
  it('leave a running or paused simulation to the usual steps', () => {
    const { engine, provider } = setup(three());
    select(provider, { nodeId: 'a' });
    engine.stepFromSelection(+1);
    expect(engine.stepFromSelection(+1)).toBe(false);   // (active: step() does it)
  });
});

// ── The progress bar ──────────────────────────────────────────────────────────

function withBar(nodes, extra = {}) {
  const handlers = {};
  const bar = { style: {}, classList: { add() {}, remove() {} }, addEventListener(t, f) { handlers[t] = f; }, getBoundingClientRect: () => ({ left: 0, width: 100 }) };
  const fill = { style: {} }, handle = { style: {} };
  const listeners = {};
  const provider = {
    lastGraphNodes: nodes,
    on(name, f) { (listeners[name] ||= []).push(f); return this; },
    _emit(name, ev) { (listeners[name] || []).forEach(f => f(ev)); },
  };
  let activated = 0;
  const engine = new SimulationEngine(provider, { progressEl: bar, fillEl: fill, handleEl: handle, onActivate: () => { activated++; }, ...extra });
  return { engine, provider, bar, fill, handlers, activated: () => activated };
}

describe('the progress bar', () => {
  it('is shown when the setting says so, simulation or not', () => {
    const { engine, bar } = withBar(three());
    engine.setProgressBarEnabled(false);
    expect(bar.style.display).toBe('none');
    engine.setProgressBarEnabled(true);
    expect(bar.style.display).toBe('block');
  });
  it('says where a node that is selected is in the sequence, without a simulation', () => {
    const { engine, provider, fill } = withBar(three());
    engine.setProgressBarEnabled(true);
    select(provider, { nodeId: 'b' });
    expect(parseFloat(fill.style.width)).toBeCloseTo(200 / 3, 5);
    select(provider, { nodeId: 'c' });
    expect(parseFloat(fill.style.width)).toBeCloseTo(100, 5);
    expect(engine.isActive).toBe(false);
  });
  it('is left as it is for a selection that is not unique, and empty when the selection is cleared or the graph is drawn again', () => {
    const iterated = node('i', 100, { total_orders: [100, 300], trace_indices: [1, 3] });
    const { engine, provider, fill } = withBar([iterated, node('x', 200), node('y', 400)]);
    engine.setProgressBarEnabled(true);
    select(provider, { nodeId: 'x' });
    expect(parseFloat(fill.style.width)).toBeCloseTo(50, 5);   // (x is the second of 4 entries)
    select(provider, { nodeId: 'i' });                           // (the node of two iterations as a whole)
    expect(parseFloat(fill.style.width)).toBeCloseTo(50, 5);
    select(provider, { nodeId: 'i', iterIdx: 1 });
    expect(parseFloat(fill.style.width)).toBeCloseTo(75, 5);
    provider._emit('clear');
    expect(fill.style.width).toBe('0%');
    select(provider, { nodeId: 'x' });
    provider._emit('fetchstart');
    expect(fill.style.width).toBe('0%');
  });
  it('follows the simulation while there is one, and the selection after it stopped', () => {
    const { engine, provider, fill } = withBar(three());
    engine.setProgressBarEnabled(true);
    select(provider, { nodeId: 'a' });
    engine.stepFromSelection(+1);
    expect(parseFloat(fill.style.width)).toBeCloseTo(200 / 3, 5);
    engine.stop();
    expect(parseFloat(fill.style.width)).toBeCloseTo(200 / 3, 5);   // (the simulation showed b: the selection is b)
  });
  it('tells the page when a simulation starts: the arrow keys, play, the bar itself', () => {
    const a = withBar(three());
    select(a.provider, { nodeId: 'a' });
    a.engine.stepFromSelection(+1);
    expect(a.activated()).toBe(1);
    const b = withBar(three());
    b.engine.start();
    b.engine.stop();
    expect(b.activated()).toBe(1);
    globalThis.document = { addEventListener() {}, removeEventListener() {} };
    const c = withBar(three());
    c.handlers.mousedown({ clientX: 10, preventDefault() {} });
    expect(c.engine.isActive).toBe(true);
    expect(c.engine._running).toBe(false);
    expect(c.activated()).toBe(1);
    delete globalThis.document;
  });
});

describe('Simulate from the selection, and the deselection that stops it', () => {
  it('starts from a unique selection: the next node is the first step', () => {
    const { engine, provider, selects } = setup(three());
    select(provider, { nodeId: 'a' });
    engine.start();
    expect(shown(selects)).toEqual(['b']);
    expect(engine._running).toBe(true);
    engine.stop();
  });
  it('starts from the beginning without a selection, with one that is not unique, or at the last node', () => {
    for (const make of [
      () => null,
      p => select(p, { nodeId: 'c' }),
      p => p._emit('select', { source: 'code-listing', nodeId: 'b', focusId: 'b', allIds: ['b', 'c'] }),
    ]) {
      const { engine, provider, selects } = setup(three());
      make(provider);
      engine.start();
      expect(shown(selects)).toEqual(['a']);
      engine.stop();
    }
  });
  it('is stopped by a click in the canvas outside of every item, which is also a deselection', () => {
    const { engine, provider } = setup(three());
    select(provider, { nodeId: 'a' });
    engine.start();
    expect(engine.isActive).toBe(true);
    provider._emit('clear', { background: true });
    expect(engine.isActive).toBe(false);
    expect(engine._running).toBe(false);
    expect(engine.stepFromSelection(+1)).toBe(false);   // (no selection any more)
  });
  it('is not stopped by the clear that a click on a node makes before its selection', () => {
    const { engine, provider } = setup(three());
    select(provider, { nodeId: 'a' });
    engine.start();
    engine.pause();
    provider._emit('clear', { background: false });
    select(provider, { nodeId: 'c' });
    expect(engine.isActive).toBe(true);
    engine.stop();
  });
  it('is stopped also when it is paused', () => {
    const { engine, provider } = setup(three());
    select(provider, { nodeId: 'a' });
    engine.stepFromSelection(+1);
    provider._emit('clear', { background: true });
    expect(engine.isActive).toBe(false);
  });
});
