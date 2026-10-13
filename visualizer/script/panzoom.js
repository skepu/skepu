// panzoom.js — how the graph of SkeVU is panned and zoomed, shared by GraphView and the designer: the wheel and the pinch, and the smooth pan and
// zoom to a node, a region or a group of nodes (when something is chosen in the code, say).
//
// No guess is made of the device, as the events say what is meant:
//  * a plain wheel, or a two-finger swipe on a trackpad, pans the graph (Shift: sideways, for a wheel that only turns up and down);
//  * Ctrl or Cmd with the wheel zooms around the pointer. This is also how Chromium and Firefox (and so an editor's webview) report the pinch
//    of a trackpad: a wheel event with ctrlKey set;
//  * Safari and WebKit report the pinch as gesturestart / gesturechange, which zoom around the pointer too (the wheel is ignored meanwhile).
// Cytoscape's own wheel zoom is off. Use `PAN_ZOOM_OPTIONS` in the options of cytoscape({...}), and `bindPanZoom(cy, element)` for the events of
// the element that the graph is in.

/** Options of cytoscape() for the zoom: no zoom by the wheel (it pans), a range of zoom, and the zoom of touch. */
export const PAN_ZOOM_OPTIONS = { wheelSensitivity: 0.0, minZoom: 0.01, maxZoom: 5, userZoomingEnabled: true };

/** How far the graph pans for a unit of the wheel (the sign makes it follow the fingers). */
const PAN_SPEED = -1.5;
/** The zoom of a Safari pinch is the scale of the gesture to this power. */
const PINCH_EXPONENT = 1.2;
/** The zoom of a wheel event is e to the power of this for each unit (pixel) of it: a pinch of Chromium is some units for each event (the rate of d3-zoom for it). */
const WHEEL_ZOOM_RATE = 0.02;
/** The most that a single wheel event zooms by, in units: the notch of a mouse wheel is 100 or more, and would jump by a factor of 7 or more. */
const WHEEL_ZOOM_MAX_DELTA = 12;

/** The pixels of a wheel delta: the wheel of a mouse can report lines (Firefox) or pages. */
export function wheelPixels(delta, deltaMode, pageSize) {
  return deltaMode === 1 ? delta * 16 : deltaMode === 2 ? delta * (pageSize || 600) : delta;
}

/** The factor that a wheel event with Ctrl or Cmd (or the pinch of a trackpad in Chromium) zooms by: more than 1 for a turn away from the user. */
export function wheelZoomFactor(deltaY, deltaMode, pageSize) {
  var d = Math.max(-WHEEL_ZOOM_MAX_DELTA, Math.min(WHEEL_ZOOM_MAX_DELTA, wheelPixels(deltaY, deltaMode, pageSize)));
  return Math.exp(-d * WHEEL_ZOOM_RATE);
}

/** How far a wheel event pans, in pixels of the graph: { x, y }. Shift turns a wheel that only goes up and down sideways. */
export function wheelPan(event, pageSize) {
  var dx = wheelPixels(event.deltaX, event.deltaMode, pageSize), dy = wheelPixels(event.deltaY, event.deltaMode, pageSize);
  if (event.shiftKey && dx === 0) { dx = dy; dy = 0; }
  return { x: dx * PAN_SPEED, y: dy * PAN_SPEED };
}

/** Pans and zooms `cy` from the events of `el`. Returns a function that takes the listeners away again. */
export function bindPanZoom(cy, el) {
  var baseZoom = 1, zooming = false;

  function onWheel(event) {
    event.preventDefault();
    if (zooming) return;                          // (a pinch of Safari: its gesture zooms, and the wheel events that come with it do not pan)
    var rect = el.getBoundingClientRect();
    if (event.ctrlKey || event.metaKey) {
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

export const ANIM_ZOOM_MS      = 400;      // the time of a smooth pan and zoom
export const ZOOM_FIT_PAD      = 40;       // the room around a region or a group that is fitted
export const ZOOM_NODE_DIVISOR = 2.3;      // a node gets room of this fraction of the view around it

/**
 * The room that a single node has around it when the view is fitted to it: a fraction of the shorter side of the view (of the height alone,
 * a narrow view would have no width left for the node, and the zoom would go to its minimum).
 */
export function roomAround(cy) {
  return Math.min(cy.width(), cy.height()) / ZOOM_NODE_DIVISOR;
}

/** Pans and zooms the graph smoothly so that `eles` fill the view with `padding` around them. */
export function animateFit(cy, eles, padding) {
  cy.animate({ fit: { eles: eles, padding: padding } }, { duration: ANIM_ZOOM_MS });
}

/**
 * The room that something has to be inside the view, to be clearly in it (pixels, from each edge). It has to be less than what the graph leaves
 * at the edges when it is fitted (Reset View and each layout: 30 px, LAYOUT_FIT_PAD of GraphView), or the first and the last nodes of a graph that
 * has just been fitted would not be "in view" and a highlight would move it. The margin is for what is half behind an edge, not for a near one.
 */
export const VIEW_MARGIN_PX = 16;
/** What is too small to be seen as what it is: less than this fraction of the shorter side of the view, or than VISIBLE_MIN_PX, in its longer side. */
export const VISIBLE_MIN_FRACTION = 0.03;
export const VISIBLE_MIN_PX       = 28;

/**
 * Whether something is clearly in the view: all of it is inside the view (VIEW_MARGIN_PX from the edges, so that it is not half behind one), and it
 * is not too small to see. `box` is where it is drawn, in pixels of the view: { x1, y1, x2, y2 }; `view` is { width, height }.
 */
export function isClearlyInView(box, view) {
  var shorter = Math.min(view.width, view.height);
  var margin  = VIEW_MARGIN_PX;
  var minSize = Math.max(VISIBLE_MIN_PX, VISIBLE_MIN_FRACTION * shorter);
  if (box.x1 < margin || box.y1 < margin || box.x2 > view.width - margin || box.y2 > view.height - margin) return false;
  return Math.max(box.x2 - box.x1, box.y2 - box.y1) >= minSize;
}

/**
 * What a highlight of something in another view does to the graph: it is shown with animateFit, but only if it is not clearly in view already
 * (the view then stays still: it would move and zoom for what the eye already has). Returns whether the view moved.
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

/**
 * Keeps the graph fitted in its element while it is the fitted view. `fit()` fits the graph (Reset View, each layout) and remembers that the view
 * is the fitted one; a pan or a zoom of anything else (the user, a highlight) forgets it. When the element changes size (the legend and the
 * timeline that are drawn after the layout, a pane that is dragged, the window), Cytoscape is told, and the graph is fitted again if the view still
 * is the fitted one: without that the graph is fitted to the size that the element had before, and clips when the element gets smaller.
 * This is done once the size has been still for FIT_SETTLE_MS, not for each change: a dragged pane changes it for every pointer move, and a resize
 * of Cytoscape (its canvas is made again) and a fit for each of them is a flicker.
 */
/** How long the size of the element has to be still before the graph is fitted to it again. */
export const FIT_SETTLE_MS = 120;

/** `whenSettled`: the size of the element has to have been still for this long, and it is waited for at most FIT_WAIT_MAX_MS. */
export const FIT_QUIET_MS    = 50;
export const FIT_WAIT_MAX_MS = 800;

export class FitKeeper {
  /** `observe(el, callback)` is for the tests: it makes the callback be called when the size of `el` changes, and returns what stops it. */
  constructor(el, padding, observe) {
    this._padding = padding;
    this._timer   = null;
    this._changedAt = 0;     // when the element last changed size
    this._waiting = 0;       // the number of whenSettled that wait (destroy ends them)
    this._cy      = null;
    this._fitted  = false;   // the view is the one that fit() made
    this._moving  = false;   // fit() or resize() is what moves the view just now
    this._stop    = (observe || observeSize)(el, this._onResize.bind(this));
  }

  /** For the graph that is made (a new one for each render): the view of a new graph is not the fitted one until it is fitted. */
  attach(cy) {
    var self = this;
    this._cy = cy;
    this._fitted = false;
    cy.on('viewport', function() { if (!self._moving) self._fitted = false; });
  }

  /** Fits the graph, with the padding, and keeps it fitted while the view is not moved by anything else. */
  fit() {
    if (!this._cy) return;
    this._moving = true;
    try { this._cy.fit(undefined, this._padding); } finally { this._moving = false; }
    this._fitted = true;
  }

  /** The size of the element changed: what is to be done for it waits until the size is still (a change in the meantime starts the wait again). */
  _onResize() {
    this._changedAt = Date.now();
    clearTimeout(this._timer);
    this._timer = setTimeout(this._settle.bind(this), FIT_SETTLE_MS);
  }

  /**
   * Calls `callback` when the element has been the same size for FIT_QUIET_MS (the panes that a render brings in, the legend and the timeline, are in
   * place), or FIT_WAIT_MAX_MS after, whatever comes first, with Cytoscape told of the size and the graph fitted to it at once (as the wait of the
   * ordinary refit is for a size that goes on changing). The graph can be kept hidden until then, so that nobody sees it fitted to a size that
   * the element does not keep.
   */
  whenSettled(callback) {
    var self = this, start = Date.now();
    this._waiting++;
    function check() {
      if (!self._waiting) return;                              // (destroyed)
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

  /** Stops watching the element. */
  destroy() {
    this._stop();
    clearTimeout(this._timer);
    this._timer = null;
    this._waiting = 0;
    this._cy = null;
  }
}
