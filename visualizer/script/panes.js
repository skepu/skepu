// panes.js — panes that open and close (the legend, the source pane and the gutter beside it) slide like the timeline: for PANE_SLIDE_MS, with the
// easing main.css gives all of them (--pane-slide). A closed pane has the class `pane-closed` (display: none).
//
// Only a pane the user opens or closes slides (`animate`); panes as the page makes them or as the settings restore them are not animated. The
// transition exists only while a pane slides (class `pane-sliding`), so a pane being dragged larger does not lag behind the pointer.

export const PANE_SLIDE_MS = 150;

// Height and width panes differ in the CSS property that limits the size, how the pane is measured, and the properties that shrink to nothing with it
// (padding and border would show at size 0).
var AXES = {
  height: { max: 'maxHeight', natural: function(el) { return el.scrollHeight; }, current: function(el) { return el.offsetHeight; },
            collapse: ['paddingTop', 'paddingBottom', 'borderTopWidth'] },
  width:  { max: 'maxWidth',  natural: function(el) { return el.offsetWidth; },  current: function(el) { return el.offsetWidth; },
            collapse: ['paddingLeft', 'paddingRight', 'borderLeftWidth'] },
};

function reducedMotion() {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Whether the pane is open: not closed and not closing (a pane sliding closed is still on the page until it has closed). */
export function isPaneOpen(el) {
  return !el.classList.contains('pane-closed') && !el._paneClosing;
}

function endSlide(el, axis) {
  el.style[axis.max] = '';
  axis.collapse.forEach(function(p) { el.style[p] = ''; });
  el.classList.remove('pane-sliding');
  el._paneClosing = false;
}

/**
 * @param {HTMLElement} el
 * @param {boolean} open
 * @param {'height'|'width'} axis  the size that it slides in
 * @param {boolean} animate  slide (the user acted) or jump to the final state (page made, settings restored)
 */
export function setPaneOpen(el, open, axis, animate) {
  if (!el) return;
  var a = AXES[axis];
  clearTimeout(el._paneTimer);
  endSlide(el, a);                                     // finish an unfinished slide: the pane is as it was going to be
  if (open === !el.classList.contains('pane-closed')) return;
  if (!animate || reducedMotion()) { el.classList.toggle('pane-closed', !open); return; }

  el.classList.add('pane-sliding');
  if (open) {
    el.classList.remove('pane-closed');                // measure the size it slides to
    var natural = a.natural(el);
    el.style[a.max] = '0px';
    a.collapse.forEach(function(p) { el.style[p] = '0px'; });
    void el.offsetWidth;                               // force a reflow so the transition starts from size 0
    el.style[a.max] = natural + 'px';
    a.collapse.forEach(function(p) { el.style[p] = ''; });
    el._paneTimer = setTimeout(function() { endSlide(el, a); }, PANE_SLIDE_MS + 60);
  } else {
    el._paneClosing = true;
    el.style[a.max] = a.current(el) + 'px';            // pin the current size: the transition runs from it to 0
    void el.offsetWidth;
    el.style[a.max] = '0px';
    a.collapse.forEach(function(p) { el.style[p] = '0px'; });
    el._paneTimer = setTimeout(function() { el.classList.add('pane-closed'); endSlide(el, a); }, PANE_SLIDE_MS + 60);
  }
}
