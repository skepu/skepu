// reveal.js — from the splash (the welcome panel) to the viewer with a trace in it: the splash fades out over the panes.
//
// While the first trace is drawn the panes are in their places, under the splash (class reveal-pending), so the graph is fitted to its final
// size. The splash is moved to the body as a fixed overlay with its old place and size, since the panes would shrink its area. When the
// first render has settled it fades out and is hidden. Without a splash (a second trace) nothing is held.

export const REVEAL_FADE_MS     = 300;    // as --reveal-fade in main.css
/** The splash does not stay for ever if the first render never settles (a trace that takes very long, or fails). */
export const REVEAL_MAX_WAIT_MS = 8000;

var _timer = null;

/**
 * The first trace is being drawn: keep the splash, and the panes transparent, until `reveal`. Returns whether it is held (there is a splash).
 * Call it before the panes take their places (before welcome mode is left): the splash is measured as it is now.
 * @param {HTMLElement} html   the page (document.documentElement)
 * @param {HTMLElement} panel  the welcome panel
 * @param {HTMLElement} body   where the panel goes as an overlay (document.body)
 */
export function holdForReveal(html, panel, body) {
  if (!panel || panel.classList.contains('hidden')) return false;
  // (pixels: the overlay does not follow the layout that is to change)
  var box = panel.getBoundingClientRect();
  panel.style.top    = box.top    + 'px';
  panel.style.left   = box.left   + 'px';
  panel.style.width  = box.width  + 'px';
  panel.style.height = box.height + 'px';
  panel.classList.add('welcome-overlay');
  body.appendChild(panel);
  html.classList.add('reveal-pending');
  clearTimeout(_timer);
  _timer = setTimeout(function() { reveal(html, panel); }, REVEAL_MAX_WAIT_MS);
  return true;
}

/**
 * Fades the splash out, to show the viewer that is under it (if the splash is held: otherwise nothing happens). The splash is hidden when its
 * fade has ended (transitionend), not after a time that the fade could be late for when the page is busy; the time is for when no transition runs.
 */
export function reveal(html, panel) {
  clearTimeout(_timer);
  _timer = null;
  if (!panel || !html.classList.contains('reveal-pending')) return;
  html.classList.remove('reveal-pending');
  // No splash (a reload that restores a trace, an editor: early.js): nothing to cross-fade.
  if (html.classList.contains('no-splash')) {
    panel.classList.add('hidden');
    return;
  }
  panel.classList.add('fading');

  var finished = false, fallback = null;
  function finish() {
    if (finished) return;
    finished = true;
    clearTimeout(fallback);
    panel.removeEventListener('transitionend', onEnd);
    panel.classList.add('hidden');
    panel.classList.remove('fading');
  }
  function onEnd(event) {
    if (event.target === panel && event.propertyName === 'opacity') finish();
  }
  panel.addEventListener('transitionend', onEnd);
  fallback = setTimeout(finish, REVEAL_FADE_MS * 3);
}
