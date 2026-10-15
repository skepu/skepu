// tour.js — the guided tour: a spotlight on one part of the viewer at a time, with a short text and Back / Next / Skip. The steps are data
// (STEPS): what to show, the text, and `prepare`, which puts the viewer in the state of the step. The tour makes no lasting change: saved
// settings and opened panes are put back when it ends. What is lit can be used: a step may ask the user to try it (`task`).
//
// It starts from the ? dialog or the welcome panel, and is offered once after the first trace is drawn. Without a trace it loads an example
// (web only: window.loadTourExample).
import { State } from './state.js';
import { SETTINGS_KEY } from './appearance.js';
import { hideShortcuts } from './shortcuts.js';
import { visibleSteps, stepIndex, spotRect, placePopover } from './tour_core.js';
import { closeGanttPane, openGanttPane, openSourcePane, closeSourcePane, toggleLegend } from './ui.js';

const STORAGE_KEY = 'skevu-tour';   // 'offered' (the offer has been made) or 'done' (the tour was finished)
const PAD = 6;
const SWAP_MS = 130;    // the text fades out (main.css: .tour-card)
const GLIDE_MS = 200;   // the spotlight moves to the next part (main.css: .tour-spot)

// ── What a step can do to the viewer ──────────────────────────────────────────

const wait = ms => new Promise(r => setTimeout(r, ms));

function menuNamed(title) {
  return Array.from(document.querySelectorAll('#menubar .menu')).find(function(m) {
    const t = m.querySelector(':scope > .menu-title');
    return t && t.textContent.trim() === title;
  });
}
function closeMenus() { document.querySelectorAll('.menu.open').forEach(m => m.classList.remove('open')); }
function openMenu(title) { closeMenus(); const m = menuNamed(title); if (m) m.classList.add('open'); }
function dropdownOf(title) { const m = menuNamed(title); return m && m.querySelector(':scope > .dropdown'); }

function firstCallId() {
  const nodes = State.graphView ? State.graphView._lastGraphNodes : [];
  const n = nodes.find(x => (x.data || x).type === 'skeleton_call');
  return n ? (n.data || n).id : null;
}

/** For a task: `listen(done)` starts to wait for what the user does and returns the function that stops it. */
function onSelect(source) {   // a node is selected in a view (the events of the views, TraceDataProvider 'select')
  return function(done) {
    const h = function(ev) { if (!source || ev.source === source) done(); };
    State.provider.on('select', h);
    return function() { State.provider.off('select', h); };
  };
}
function onWheel(selector) {   // the wheel or a pinch on an element
  return function(done) {
    const e = document.querySelector(selector);
    if (!e) return function() {};
    e.addEventListener('wheel', done, { passive: true });
    e.addEventListener('gesturechange', done);
    return function() { e.removeEventListener('wheel', done); e.removeEventListener('gesturechange', done); };
  };
}
/** Nothing selected (what a selection shows can then be seen coming). */
function clearSelection() {
  if (State.graphView && State.graphView._cy) State.graphView._cy.nodes().removeClass('selected_gutter');
  if (State.provider) State.provider._emit('clear');
}
function hasSelection() { return !!(State.graphView && State.graphView._cy && State.graphView._cy.nodes('.selected_gutter').length); }

/** The panes as the tour found them (to put them back). */
let _found = null;
function ensureSource() { if (!State._sourcePaneOpen) openSourcePane(false); }
function ensureTimeline() { if (!State._ganttPaneOpen) openGanttPane(); }
function ensureLegend() { if (!State._legendOpen) toggleLegend(); }

// ── The steps ─────────────────────────────────────────────────────────────────
// target: a selector or a function that gives the element (no target: the text is in the middle); prepare: puts the viewer in the state of the
// step (it may wait); when: false leaves the step out (context: { vscode }); side: where the text goes, if it fits.

export const STEPS = [
  { id: 'graph', target: '#cy', title: 'The graph',
    text: 'Each node is a skeleton call or data event from the trace; edges are the dependences between them.',
    task: { text: 'Try it: scroll to pan, or Ctrl or ⌘ + scroll (or pinch) to zoom', listen: onWheel('#cy') } },
  { id: 'select', target: '#cy', title: 'Select a node', side: 'right',
    text: 'A selected node lights up in every view: the info pane, the timeline, and the lines of the program that it came from.',
    task: { text: 'Try it: click a node', listen: onSelect('graph') },
    prepare: async function() { ensureSource(); clearSelection(); await wait(250); } },
  { id: 'source', target: '#right_pane', title: 'The same node in the source', side: 'left',
    text: 'Its lines are marked, with a badge in the gutter. Click a badge to select that node instead; the graph and the timeline follow.',
    task: { text: 'Try it: click a badge in the gutter', listen: onSelect('code-listing') },
    prepare: async function() {
      ensureSource(); await wait(250);
      const id = firstCallId();
      if (!hasSelection() && id && State.graphView) State.graphView.selectNode(id, { zoom: false });
    } },
  { id: 'timeline', target: '#gantt-pane', title: 'The timeline', side: 'top',
    text: 'Time runs to the right, with a row for each backend. Z zooms to the measured interval, G shows or hides this pane.',
    task: { text: 'Try it: click a bar', listen: onSelect('gantt') },
    prepare: async function() { ensureTimeline(); await wait(350); } },
  { id: 'nodes', target: () => dropdownOf('Nodes'), title: 'What to draw', side: 'right',
    text: 'Show or hide kinds of nodes: allocations, transfers, kernel builds, communication. The graph is drawn again at once for small traces.',
    prepare: async function() { openMenu('Nodes'); await wait(120); } },
  { id: 'edges', target: () => dropdownOf('Edges'), title: 'Edges and layout', side: 'right',
    text: 'Anti-dependences, labels, and the direction of the layout.',
    prepare: async function() { openMenu('Edges'); await wait(120); } },
  { id: 'legend', target: '#legend', title: 'The legend', side: 'top',
    text: 'It explains the colours, shapes and edge styles of what is on the screen now.',
    prepare: async function() { closeMenus(); ensureLegend(); await wait(300); } },
  { id: 'simulate', target: () => dropdownOf('Actions'), title: 'Replay the run', side: 'right',
    text: 'Simulate plays the run as an animation, node by node (P plays and pauses). The arrow keys step through it from the selected node. The other actions work on the selected node.',
    prepare: async function() { openMenu('Actions'); await wait(120); } },
  { id: 'files', target: () => dropdownOf('Files'), title: 'Your own trace', side: 'right', when: ctx => !ctx.vscode,
    text: 'Load a trace of your own here, with its source files, or open another example.',
    prepare: async function() { openMenu('Files'); await wait(120); } },
  { id: 'help', target: '.menu-help-title', title: 'Shortcuts', side: 'bottom',
    text: 'Press ? at any time for the list of keyboard shortcuts. The tour is started again from there.',
    prepare: async function() { closeMenus(); await wait(60); } },
];

// ── The tour ──────────────────────────────────────────────────────────────────

let _active = null;   // { steps, index, layer, spot, card, ... } while the tour runs

export function tourActive() { return !!_active; }

function readStorage() { try { return localStorage.getItem(STORAGE_KEY); } catch (e) { return null; } }
function writeStorage(v) { try { localStorage.setItem(STORAGE_KEY, v); } catch (e) { /* (no storage) */ } }

function el(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text != null) e.textContent = text;
  return e;
}

function targetElement(step) {
  if (!step.target) return null;
  const t = typeof step.target === 'function' ? step.target() : document.querySelector(step.target);
  return t && t.getBoundingClientRect().width > 0 ? t : null;
}

function position() {
  if (!_active) return;
  const step = _active.steps[_active.index];
  const viewport = { width: window.innerWidth, height: window.innerHeight };
  const target = targetElement(step);
  const rect = target ? spotRect(target.getBoundingClientRect(), PAD, viewport) : null;
  const spot = _active.spot;
  if (rect) {
    spot.hidden = false;
    spot.style.left = rect.left + 'px'; spot.style.top = rect.top + 'px';
    spot.style.width = rect.width + 'px'; spot.style.height = rect.height + 'px';
    // (the page around the spotlight is dimmed by four panels: a shadow that is larger than the screen is not drawn everywhere)
    const [top, bottom, left, right] = _active.dims;
    const box = (e, l, t, w, h) => { e.style.left = l + 'px'; e.style.top = t + 'px'; e.style.width = Math.max(0, w) + 'px'; e.style.height = Math.max(0, h) + 'px'; e.hidden = false; };
    box(top, 0, 0, viewport.width, rect.top);
    box(bottom, 0, rect.top + rect.height, viewport.width, viewport.height - rect.top - rect.height);
    box(left, 0, rect.top, rect.left, rect.height);
    box(right, rect.left + rect.width, rect.top, viewport.width - rect.left - rect.width, rect.height);
  } else {
    spot.hidden = true;
    _active.dims.forEach(d => { d.hidden = true; });
  }
  _active.layer.classList.toggle('tour-dim', !rect);   // (without a spotlight the layer itself dims the page)
  const card = _active.card;
  const p = placePopover(rect, { width: card.offsetWidth, height: card.offsetHeight }, viewport, step.side || 'auto');
  card.style.left = p.left + 'px'; card.style.top = p.top + 'px';
}

function stopTask() {
  if (_active && _active.stop) { _active.stop(); _active.stop = null; }
}

async function show(index) {
  if (!_active) return;
  const token = ++_active.token;
  stopTask();
  _active.index = index;
  const step = _active.steps[index], last = index === _active.steps.length - 1;
  const card = _active.card;
  // The text goes away, the spotlight moves to the next part (it glides), and the text comes back there: it does not change before the move.
  card.classList.add('tour-swapping');
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  await Promise.all([wait(still ? 0 : SWAP_MS), step.prepare ? step.prepare() : null]);
  if (!_active || token !== _active.token) return;   // (the user went on while the step was being prepared)
  card.querySelector('.tour-title').textContent = step.title;
  card.querySelector('.tour-text').textContent = step.text;
  card.querySelector('.tour-count').textContent = (index + 1) + ' / ' + _active.steps.length;
  card.querySelector('.tour-back').disabled = index === 0;
  card.querySelector('.tour-next').textContent = last ? 'Done' : 'Next';
  card.querySelector('.tour-next').classList.remove('tour-ready');
  const taskEl = card.querySelector('.tour-task');
  taskEl.hidden = !step.task;
  taskEl.classList.remove('tour-task-done');
  if (step.task) {
    taskEl.textContent = step.task.text;
    stopTask();
    // (what the user does while the text is on the screen; once is enough)
    _active.stop = step.task.listen(function() {
      if (!_active || token !== _active.token || taskEl.classList.contains('tour-task-done')) return;
      taskEl.classList.add('tour-task-done');
      taskEl.textContent = 'Done: ' + step.task.text.replace(/^Try it: /, '');
      card.querySelector('.tour-next').classList.add('tour-ready');
    });
  }
  position();
  await wait(still ? 0 : GLIDE_MS);
  if (!_active || token !== _active.token) return;
  card.classList.remove('tour-swapping');
  card.querySelector('.tour-next').focus({ preventScroll: true });
}

function go(delta) {
  if (!_active) return;
  const i = stepIndex(_active.index, delta, _active.steps.length);
  if (i === null) { if (delta > 0) endTour(true); return; }
  show(i);
}

function onKey(event) {
  if (!_active) return;
  // (the viewer is in use while the tour runs, so its keys work: the tour has Esc, and the arrows when its own buttons have the focus)
  const k = event.key, inCard = _active.card.contains(document.activeElement);
  if (k === 'Escape') endTour(false);
  else if (inCard && k === 'ArrowRight') go(1);
  else if (inCard && k === 'ArrowLeft') go(-1);
  else return;
  event.preventDefault();
  event.stopPropagation();
}

function build() {
  const layer = el('div', 'tour-layer');
  layer.setAttribute('role', 'dialog');
  layer.setAttribute('aria-label', 'Guided tour');
  // (a click on the text or on the dimmed page is not for the page: it closes no menu, which a step may have opened; a click in the light is)
  const dims = [0, 1, 2, 3].map(function() { return el('div', 'tour-dimmer'); });
  layer.addEventListener('click', function(e) { if (e.target !== layer || layer.classList.contains('tour-dim')) e.stopPropagation(); });
  const spot = el('div', 'tour-spot');
  const card = el('div', 'tour-card tour-swapping');
  card.appendChild(el('h3', 'tour-title'));
  card.appendChild(el('p', 'tour-text'));
  card.appendChild(el('p', 'tour-task'));
  const foot = el('div', 'tour-foot');
  foot.appendChild(el('span', 'tour-count'));
  const skip = el('button', 'tour-skip', 'Skip');
  const back = el('button', 'tour-back', 'Back');
  const next = el('button', 'tour-next', 'Next');
  skip.addEventListener('click', function() { endTour(false); });
  back.addEventListener('click', function() { go(-1); });
  next.addEventListener('click', function() { go(1); });
  foot.appendChild(skip); foot.appendChild(back); foot.appendChild(next);
  card.appendChild(foot);
  dims.forEach(d => layer.appendChild(d));
  layer.appendChild(spot); layer.appendChild(card);
  document.body.appendChild(layer);
  return { layer, spot, card, dims };
}

/** The panes and the saved settings as they were, put back (the tour makes no lasting change). */
function restore(found) {
  closeMenus();
  if (!found.source && State._sourcePaneOpen) closeSourcePane(false);
  if (found.source && !State._sourcePaneOpen) openSourcePane(false);
  if (!found.timeline && State._ganttPaneOpen) closeGanttPane();
  if (!found.legend && State._legendOpen) toggleLegend();
  // (those functions save the settings: what was saved before is written back last)
  try {
    if (found.stored === null) localStorage.removeItem(SETTINGS_KEY); else localStorage.setItem(SETTINGS_KEY, found.stored);
  } catch (e) { /* (no storage) */ }
}

export function endTour(finished) {
  if (!_active) return;
  const { layer, found } = _active;
  stopTask();
  _active = null;
  window.removeEventListener('keydown', onKey, true);
  window.removeEventListener('resize', position);
  layer.remove();
  restore(found);
  writeStorage(finished ? 'done' : (readStorage() || 'offered'));
}

/** Starts the tour (with an example if there is no trace, in a browser). Returns whether it started. */
export async function startTour() {
  if (_active) return false;
  hideShortcuts();
  hideOffer();
  if (!State._traceLoaded) {
    if (typeof window.loadTourExample !== 'function' || !(await window.loadTourExample())) return false;
  }
  let stored = null;
  try { stored = localStorage.getItem(SETTINGS_KEY); } catch (e) { /* (no storage) */ }
  _found = { source: !!State._sourcePaneOpen, timeline: !!State._ganttPaneOpen, legend: !!State._legendOpen, stored };
  const ctx = { vscode: !!(window.skevu && window.skevu.vscode) || typeof acquireVsCodeApi === 'function' };
  const steps = visibleSteps(STEPS, ctx);
  const ui = build();
  _active = { steps, index: 0, token: 0, found: _found, ...ui };
  window.addEventListener('keydown', onKey, true);
  window.addEventListener('resize', position);
  await show(0);
  return true;
}

// ── The offer: once, after the first trace has been drawn ─────────────────────

function hideOffer() { const o = document.getElementById('tour-offer'); if (o) o.remove(); }

function offer() {
  if (readStorage() || _active || document.getElementById('tour-offer')) return;
  writeStorage('offered');
  const box = el('div', 'tour-offer');
  box.id = 'tour-offer';
  box.appendChild(el('span', 'tour-offer-text', 'New to SkeVU? A one-minute tour shows the main parts.'));
  const yes = el('button', 'tour-offer-yes', 'Take the tour');
  const no = el('button', 'tour-offer-no', 'No thanks');
  yes.addEventListener('click', function() { startTour(); });
  no.addEventListener('click', hideOffer);
  box.appendChild(yes); box.appendChild(no);
  document.body.appendChild(box);
}

/** Once the page is made: the offer after the first trace that is drawn, and the action for the markup. */
export function installTour() {
  window.startTour = startTour;
  if (readStorage()) return;
  if (State.provider) {
    let offered = false;
    State.provider.on('rendersettled', function() {
      if (offered || !State._traceLoaded) return;
      offered = true;
      setTimeout(offer, 1500);
    });
  }
}
