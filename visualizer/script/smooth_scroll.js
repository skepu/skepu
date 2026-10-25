// smooth_scroll.js — scrolls a pane to a selected line from where the pane is now, whatever the direction and distance. It is done here
// (frames set the position) and not by the browser's smooth scroll, which can start from a position that a layout change or tab switch
// just reset. A new scroll takes over from the current position; the user's wheel, touch or keys end it.
'use strict';

export const MIN_MS = 140, MAX_MS = 420;

/** How long the scroll of `distance` px takes: a little more for a long way, not much (a long way is not a long wait). */
export function scrollDuration(distance) {
  return Math.max(MIN_MS, Math.min(MAX_MS, MIN_MS + Math.abs(distance) * 0.3));
}

/** The progress (0 to 1) at the fraction t of the time: slow at the ends. */
export function ease(t) {
  const c = Math.max(0, Math.min(1, t));
  return c < 0.5 ? 2 * c * c : 1 - Math.pow(-2 * c + 2, 2) / 2;
}

/**
 * Whether the lines are already in view, so that nothing needs to scroll: `lines` are the rectangles ({ top, bottom }) of the highlighted lines, `target`
 * the one that is scrolled to, `view` the rectangle of the pane. The lines have to be inside it with a margin (a line at the very edge is not
 * "in view": it would be read cut off); if they are more than the pane can show, the target line has to be. A pane that has no height (it is not
 * shown) shows nothing.
 */
export function linesInView(lines, target, view) {
  const height = view.bottom - view.top;
  if (!(height > 0) || !lines.length) return false;
  const margin = Math.min(48, height * 0.15);
  const inside = r => r.top >= view.top + margin && r.bottom <= view.bottom - margin;
  const all = { top: Math.min(...lines.map(l => l.top)), bottom: Math.max(...lines.map(l => l.bottom)) };
  return all.bottom - all.top <= height - 2 * margin ? inside(all) : inside(target);
}

/**
 * The size in px of a flex item that has the flex-basis of `flex` (the style of an element: "0 0 120px", "0 0 10em"), or null if it is not
 * a size ("1 1 auto"). `fontSizePx` is the font size of the element (for em).
 */
export function flexBasisPx(flex, fontSizePx) {
  const m = /^\s*[\d.]+\s+[\d.]+\s+([\d.]+)(px|em)\s*$/.exec(flex || '');
  if (!m) return null;
  return m[2] === 'px' ? +m[1] : +m[1] * fontSizePx;
}

/**
 * How far from the top of the pane (px) the line that is scrolled to goes: a third of the way down, so that there is code to read above it and
 * below it, and at least `CONTEXT_LINES` lines from the top and from the bottom. `view` is the height of the pane, `lineHeight` that of a line, `span`
 * the height from the top of the line to the end of the last of the lines that are selected with it. If the pane has hardly room for the
 * lines, they are in the middle.
 */
export const CONTEXT_LINES = 3;
export function scrollOffset(view, lineHeight, span) {
  const context = Math.min(CONTEXT_LINES * lineHeight, view * 0.2);
  const free = view - Math.max(span, lineHeight);
  if (free <= 2 * context) return Math.max(0, free / 2);
  return Math.max(context, Math.min(free / 3, free - context));
}

function maxTop(pane) { return Math.max(0, pane.scrollHeight - pane.clientHeight); }

function reducedMotion() {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Stops the scroll that is going on in the pane, if there is one (it stays where it is). */
export function cancelScroll(pane) {
  if (pane._scrollAnim) { pane._scrollAnim.cancel(); pane._scrollAnim = null; }
}

/**
 * Scrolls the pane to `top` (px; kept inside the pane, which can be scrolled `extraRange` px further than it can now: it is going to get smaller).
 * `animate: false`, a pane that is not shown, and a system that asks for less motion jump instead. Returns the place that it scrolls to.
 * `env` is for tests: { now, raf, setTimeout, clearTimeout }.
 */
export function smoothScrollTo(pane, top, options, env) {
  options = options || {};
  // (arrow functions: the timer functions of a page throw when they are called as methods of another object)
  env = env || { now: () => performance.now(), raf: f => requestAnimationFrame(f), setTimeout: (f, ms) => setTimeout(f, ms), clearTimeout: t => clearTimeout(t) };
  cancelScroll(pane);
  const to = Math.max(0, Math.min(maxTop(pane) + (options.extraRange || 0), top));
  const from = pane.scrollTop;
  if (options.animate === false || reducedMotion() || Math.abs(to - from) < 1) {
    pane.scrollTop = to;
    return to;
  }
  const duration = scrollDuration(to - from);
  const start = env.now();
  let done = false, timer = null;
  const finish = function() {
    if (done) return;
    done = true;
    env.clearTimeout(timer);
    pane.removeEventListener('wheel', cancel);
    pane.removeEventListener('touchstart', cancel);
    pane.removeEventListener('keydown', cancel);
    if (pane._scrollAnim === handle) pane._scrollAnim = null;
  };
  const cancel = function() { finish(); };
  const handle = { cancel: cancel };
  pane._scrollAnim = handle;
  pane.addEventListener('wheel', cancel, { passive: true });
  pane.addEventListener('touchstart', cancel, { passive: true });
  pane.addEventListener('keydown', cancel);
  const frame = function() {
    if (done) return;
    const t = (env.now() - start) / duration;
    pane.scrollTop = from + (to - from) * ease(t);
    if (t >= 1) finish(); else env.raf(frame);
  };
  // (a page that does not run frames, as a hidden one does not, still ends up where the line is)
  timer = env.setTimeout(function() { if (!done) { pane.scrollTop = to; finish(); } }, duration + 80);
  env.raf(frame);
  return to;
}
