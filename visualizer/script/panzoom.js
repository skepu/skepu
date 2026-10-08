// panzoom.js — how the graph of SkeVU is panned and zoomed, shared by GraphView and the designer: the wheel and the pinch, and the smooth pan and
// zoom to a node, a region or a group of nodes (when something is chosen in the code, say).
//
// The scroll wheel (and a two-finger swipe on a trackpad) pans the graph; the pinch gesture (Safari: gesturestart/gesturechange) zooms it
// around the pointer; Cytoscape's own wheel zoom is off. Use `PAN_ZOOM_OPTIONS` in the options of cytoscape({...}), and
// `bindPanZoom(cy, element)` for the events of the element that the graph is in.

/** Options of cytoscape() for the zoom: no zoom by the wheel (it pans), a range of zoom, and the zoom of touch. */
export const PAN_ZOOM_OPTIONS = { wheelSensitivity: 0.0, minZoom: 0.01, maxZoom: 5, userZoomingEnabled: true };

/** How far the graph pans for a unit of the wheel (the sign makes it follow the fingers). */
const PAN_SPEED = -1.5;
/** The zoom of a pinch is the scale of the gesture to this power. */
const PINCH_EXPONENT = 1.2;

/** Pans and zooms `cy` from the events of `el`. Returns a function that takes the listeners away again. */
export function bindPanZoom(cy, el) {
  var baseZoom = 1, zooming = false;

  function onWheel(event) {
    event.preventDefault();
    if (!zooming) cy.panBy({ x: event.deltaX * PAN_SPEED, y: event.deltaY * PAN_SPEED });
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
