// panes.js — the panes that open and close (the legend, the source pane and the gutter beside it) do it the way the timeline does: they slide,
// for PANE_SLIDE_MS, with the easing that main.css gives all of them (--pane-slide). A closed pane has the class `pane-closed` (display: none).
//
// The sliding is for a pane that the user opens or closes (`animate`); what is made when the page is (the panes as they were left) is not
// animated, and nothing is animated all the time: the transition is there only while a pane slides (the class `pane-sliding`), so that a pane
// that is dragged larger does not lag behind the pointer.

export const PANE_SLIDE_MS = 150;

// What differs between a pane that slides in its height and one that slides in its width: the CSS property that limits the size, how to measure the
// pane, and the properties that go to nothing with the size (the pane has padding and a border that would show when its size is 0).
var AXES = {
  height: { max: 'maxHeight', natural: function(el) { return el.scrollHeight; }, current: function(el) { return el.offsetHeight; },
            collapse: ['paddingTop', 'paddingBottom', 'borderTopWidth'] },
  width:  { max: 'maxWidth',  natural: function(el) { return el.offsetWidth; },  current: function(el) { return el.offsetWidth; },
            collapse: ['paddingLeft', 'paddingRight', 'borderLeftWidth'] },
};

function reducedMotion() {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Whether the pane is open (not closed, and not about to be: a pane that slides closed is still on the page until it has). */
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
 * Opens or closes a pane.
 * @param {HTMLElement} el
 * @param {boolean} open
 * @param {'height'|'width'} axis  the size that it slides in
 * @param {boolean} animate  slide (the user did it) or just be as it is to be (the page is made, the settings are restored)
 */
export function setPaneOpen(el, open, axis, animate) {
  if (!el) return;
  var a = AXES[axis];
  clearTimeout(el._paneTimer);
  endSlide(el, a);                                     // (a slide that was not finished: the pane is as it was going to be, and goes on from there)
  if (open === !el.classList.contains('pane-closed')) return;
  if (!animate || reducedMotion()) { el.classList.toggle('pane-closed', !open); return; }

  el.classList.add('pane-sliding');
  if (open) {
    el.classList.remove('pane-closed');                // (it has its size now: what it slides to)
    var natural = a.natural(el);
    el.style[a.max] = '0px';
    a.collapse.forEach(function(p) { el.style[p] = '0px'; });
    void el.offsetWidth;                               // (a reflow: the sizes of 0 are where the transition starts from)
    el.style[a.max] = natural + 'px';
    a.collapse.forEach(function(p) { el.style[p] = ''; });
    el._paneTimer = setTimeout(function() { endSlide(el, a); }, PANE_SLIDE_MS + 60);
  } else {
    el._paneClosing = true;
    el.style[a.max] = a.current(el) + 'px';            // (a limit that is where it is: from this the transition goes to 0)
    void el.offsetWidth;
    el.style[a.max] = '0px';
    a.collapse.forEach(function(p) { el.style[p] = '0px'; });
    el._paneTimer = setTimeout(function() { el.classList.add('pane-closed'); endSlide(el, a); }, PANE_SLIDE_MS + 60);
  }
}
