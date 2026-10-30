// panzoom.js — panning and zooming of the SkeVU graph (shared by GraphView and the designer): wheel and pinch, and smooth pan and zoom to a node,
// region or group.
//
// Events are read as they say what is meant, with no device detection:
//  * a plain wheel or a two-finger trackpad swipe pans (Shift: sideways); with View > Scroll Behaviour > Zoom (`setWheelZooms`) a plain wheel zooms
//    and Shift pans;
//  * Ctrl or Cmd with the wheel zooms around the pointer. Chromium and Firefox (so also an editor's webview) report a trackpad pinch the same way:
//    a wheel event with ctrlKey;
//  * Safari reports a pinch as gesturestart / gesturechange, which zoom around the pointer; the wheel is ignored meanwhile.
// Cytoscape's own wheel zoom is off. Use `PAN_ZOOM_OPTIONS` in cytoscape({...}) and `bindPanZoom(cy, element)` for the element's events.

/** Options for cytoscape(): wheel zoom off (the wheel pans), zoom range, touch zoom. */
export const PAN_ZOOM_OPTIONS = { wheelSensitivity: 0.0, minZoom: 0.01, maxZoom: 5, userZoomingEnabled: true };

/** Pan per wheel unit; the sign makes it follow the fingers. */
const PAN_SPEED = -1.5;
/** A Safari pinch zooms by the gesture scale to this power. */
const PINCH_EXPONENT = 1.2;
/** A wheel event zooms by e to the power of this per unit (pixel); a Chromium pinch gives a few units per event (d3-zoom's rate). */
const WHEEL_ZOOM_RATE = 0.02;
/** Cap on the units one wheel event zooms by: a mouse notch is 100 or more, which would jump by a factor of 7 or more. */
const WHEEL_ZOOM_MAX_DELTA = 12;

/** A wheel delta in pixels: a mouse wheel may report lines (Firefox) or pages. */
export function wheelPixels(delta, deltaMode, pageSize) {
  return deltaMode === 1 ? delta * 16 : deltaMode === 2 ? delta * (pageSize || 600) : delta;
}

/** The zoom factor of a Ctrl/Cmd wheel event (or Chromium trackpad pinch): above 1 for a turn away from the user. */
export function wheelZoomFactor(deltaY, deltaMode, pageSize) {
  var d = Math.max(-WHEEL_ZOOM_MAX_DELTA, Math.min(WHEEL_ZOOM_MAX_DELTA, wheelPixels(deltaY, deltaMode, pageSize)));
  return Math.exp(-d * WHEEL_ZOOM_RATE);
}

/** The pan of a wheel event in graph pixels: { x, y }. Shift makes a vertical-only wheel pan sideways. */
export function wheelPan(event, pageSize) {
  var dx = wheelPixels(event.deltaX, event.deltaMode, pageSize), dy = wheelPixels(event.deltaY, event.deltaMode, pageSize);
  if (event.shiftKey && dx === 0) { dx = dy; dy = 0; }
  return { x: dx * PAN_SPEED, y: dy * PAN_SPEED };
}

var wheelZooms = false;

/** Whether a plain wheel zooms instead of panning (for mice; a trackpad swipe is a wheel event too). Ctrl and Cmd zoom either way. */
export function setWheelZooms(on) { wheelZooms = !!on; }

/** What a wheel event does: 'zoom' or 'pan'. */
export function wheelAction(event) {
  return (event.ctrlKey || event.metaKey || (wheelZooms && !event.shiftKey)) ? 'zoom' : 'pan';
}

/** Pans and zooms `cy` from the events of `el`. Returns a function that removes the listeners. */
export function bindPanZoom(cy, el) {
  var baseZoom = 1, zooming = false;

  function onWheel(event) {
    event.preventDefault();
    if (zooming) return;                          // during a Safari pinch the gesture zooms; its wheel events do not pan
    var rect = el.getBoundingClientRect();
    if (wheelAction(event) === 'zoom') {
      cy.zoom({
        level: cy.zoom() * wheelZoomFactor(event.deltaY, event.deltaMode, rect.height),
        renderedPosition: { x: event.clientX - rect.left, y: event.clientY - rect.top },
      });
    } else {
      cy.panBy(wheelPan(event, rect.height));
    }
  }
  function onGestureStart()  { baseZoom = cy.zoom(); zooming = true; }
  function onGestureEnd()    { zooming = false; }
  function onGestureChange(e) {
    var rect = e.target.getBoundingClientRect();
    cy.zoom({
      level: Math.pow(e.scale, PINCH_EXPONENT) * baseZoom,
      renderedPosition: { x: e.clientX - rect.left, y: e.clientY - rect.top },
    });
  }

  el.addEventListener('wheel', onWheel, { passive: false });
  el.addEventListener('gesturestart', onGestureStart);
  el.addEventListener('gestureend', onGestureEnd);
  el.addEventListener('gesturechange', onGestureChange);
  return function unbind() {
    el.removeEventListener('wheel', onWheel);
    el.removeEventListener('gesturestart', onGestureStart);
    el.removeEventListener('gestureend', onGestureEnd);
    el.removeEventListener('gesturechange', onGestureChange);
  };
}

// ── Panning and zooming to nodes ─────────────────────────────────────────────

export const ANIM_ZOOM_MS      = 400;
export const ZOOM_FIT_PAD      = 40;
export const ZOOM_NODE_DIVISOR = 2.3;      // a node gets 1/this of the view as room

/**
 * The room around a single node when the view is fitted to it: a fraction of the shorter side of the view (the height alone would leave a narrow
 * view no width for the node, and the zoom would drop to its minimum).
 */
export function roomAround(cy) {
  return Math.min(cy.width(), cy.height()) / ZOOM_NODE_DIVISOR;
}

/** Pans and zooms the graph smoothly so that `eles` fill the view with `padding` around them. */
export function animateFit(cy, eles, padding) {
  cy.animate({ fit: { eles: eles, padding: padding } }, { duration: ANIM_ZOOM_MS });
}

/**
 * Margin in px from each edge that something must be inside to be clearly in view. It must be smaller than the padding a fitted graph leaves at the
 * edges (Reset View and each layout: LAYOUT_FIT_PAD, 30 px in GraphView), or the first and last nodes of a just-fitted graph would not count as in
 * view and a highlight would move it. The margin is for what is half behind an edge, not for what is near one.
 */
export const VIEW_MARGIN_PX = 16;
/** Too small to recognize: its longer side is under this fraction of the view's shorter side, or under VISIBLE_MIN_PX. */
export const VISIBLE_MIN_FRACTION = 0.03;
export const VISIBLE_MIN_PX       = 28;

/**
 * Whether something is clearly in view: wholly inside it (VIEW_MARGIN_PX from the edges) and not too small. `box` is { x1, y1, x2, y2 } in view
 * pixels; `view` is { width, height }.
 */
export function isClearlyInView(box, view) {
  var shorter = Math.min(view.width, view.height);
  var margin  = VIEW_MARGIN_PX;
  var minSize = Math.max(VISIBLE_MIN_PX, VISIBLE_MIN_FRACTION * shorter);
  if (box.x1 < margin || box.y1 < margin || box.x2 > view.width - margin || box.y2 > view.height - margin) return false;
  return Math.max(box.x2 - box.x1, box.y2 - box.y1) >= minSize;
}

/**
 * Shows a highlight from another view with animateFit, unless it is clearly in view already (the view stays still rather than move for what the eye
 * already has). Returns whether the view moved.
 */
export function animateFitUnlessInView(cy, eles, padding) {
  var b = eles.renderedBoundingBox({ includeLabels: false, includeOverlays: false });
  if (isClearlyInView(b, { width: cy.width(), height: cy.height() })) return false;
  animateFit(cy, eles, padding);
  return true;
}

// ── Keeping the graph fitted ─────────────────────────────────────────────────

/** Calls `callback` when the size of `el` changes (before the next paint); returns a function that stops it. Nothing happens without ResizeObserver. */
function observeSize(el, callback) {
  if (typeof ResizeObserver === 'undefined') return function() {};
  var observer = new ResizeObserver(callback);
  observer.observe(el);
  return function() { observer.disconnect(); };
}

/** How long the element's size must be still before the graph is fitted to it again. */
export const FIT_SETTLE_MS = 120;

/** whenSettled: the size must have been still this long; the wait is at most FIT_WAIT_MAX_MS. */
export const FIT_QUIET_MS    = 50;
export const FIT_WAIT_MAX_MS = 800;

/**
 * Keeps the graph fitted in its element while the view is the fitted one. `fit()` fits the graph (Reset View, each layout) and records that the view is
 * the fitted one; any other pan or zoom (the user, a highlight) clears that. When the element changes size (the legend and timeline drawn after a
 * layout, a dragged pane, the window), Cytoscape is told and the graph is fitted again if the view is still the fitted one; otherwise it stays fitted
 * to the old size and clips when the element shrinks.
 * This waits until the size has been still for FIT_SETTLE_MS instead of reacting to each change: a dragged pane changes it on every pointer move, and
 * a Cytoscape resize (its canvas is recreated) plus a fit for each would flicker.
 */
export class FitKeeper {
  /** `observe(el, callback)` is for tests: calls the callback when the size of `el` changes and returns a function that stops it. */
  constructor(el, padding, observe) {
    this._padding = padding;
    this._timer   = null;
    this._changedAt = 0;
    this._waiting = 0;       // the number of whenSettled that wait (destroy ends them)
    this._cy      = null;
    this._fitted  = false;
    this._moving  = false;   // fit() or resize() is what moves the view just now
    this._stop    = (observe || observeSize)(el, this._onResize.bind(this));
  }

  /** For the new graph of each render: its view is not the fitted one until fit() runs. */
  attach(cy) {
    var self = this;
    this._cy = cy;
    this._fitted = false;
    cy.on('viewport', function() { if (!self._moving) self._fitted = false; });
  }

  /** Fits the graph with the padding and keeps it fitted until something else moves the view. */
  fit() {
    if (!this._cy) return;
    this._moving = true;
    try { this._cy.fit(undefined, this._padding); } finally { this._moving = false; }
    this._fitted = true;
  }

  /** The element's size changed: wait until it is still; a further change restarts the wait. */
  _onResize() {
    this._changedAt = Date.now();
    clearTimeout(this._timer);
    this._timer = setTimeout(this._settle.bind(this), FIT_SETTLE_MS);
  }

  /**
   * Calls `callback` once the element has kept its size for FIT_QUIET_MS (the panes a render brings in, the legend and timeline, are in place), or
   * FIT_WAIT_MAX_MS after the call, whichever comes first, with Cytoscape told of the size and the graph fitted to it immediately (the ordinary
   * refit's wait is for a size that keeps changing). The graph can stay hidden until then, so that nobody sees it fitted to a size that does not last.
   */
  whenSettled(callback) {
    var self = this, start = Date.now();
    this._waiting++;
    function check() {
      if (!self._waiting) return;
      var now = Date.now();
      if (now - Math.max(self._changedAt, start) >= FIT_QUIET_MS || now - start >= FIT_WAIT_MAX_MS) {
        self._waiting--;
        clearTimeout(self._timer);
        self._settle();
        callback();
      } else {
        setTimeout(check, 16);
      }
    }
    setTimeout(check, FIT_QUIET_MS);
  }

  _settle() {
    this._timer = null;
    if (!this._cy) return;
    this._moving = true;
    try { this._cy.resize(); } finally { this._moving = false; }
    if (this._fitted) this.fit();
  }

  destroy() {
    this._stop();
    clearTimeout(this._timer);
    this._timer = null;
    this._waiting = 0;
    this._cy = null;
  }
}
