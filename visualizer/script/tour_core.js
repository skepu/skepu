// tour_core.js — the parts of the guided tour (tour.js) that are plain computation: which steps there are, where the next one is, and where the
// popover goes. No DOM, so that they can be tested.
'use strict';

/** The steps that apply in a context ({ vscode, ... }): a step with `when` is left out if it says no. */
export function visibleSteps(steps, ctx) {
  return steps.filter(s => !s.when || s.when(ctx));
}

/** The step after or before `index` of `count`, or null at the ends (the way out of the tour is the last step's Done). */
export function stepIndex(index, delta, count) {
  const i = index + delta;
  return i < 0 || i >= count ? null : i;
}

/** The rectangle of a target made larger by `pad` on every side, kept inside the viewport. */
export function spotRect(target, pad, viewport) {
  const left = Math.max(0, target.left - pad), top = Math.max(0, target.top - pad);
  const right = Math.min(viewport.width, target.left + target.width + pad), bottom = Math.min(viewport.height, target.top + target.height + pad);
  return { left, top, width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
}

/**
 * Where the popover (size: { width, height }) goes, beside the spotlight `rect` ({ left, top, width, height }, or null: the middle of the
 * viewport). `prefer` is the side to try first; the others follow (below, above, right, left). If none has room, which happens for a target that
 * fills the viewport, the popover is inside the target at the bottom. Always inside the viewport, `margin` from its edges.
 * Returns { left, top, side }.
 */
export function placePopover(rect, size, viewport, prefer = 'auto', gap = 14, margin = 12) {
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const maxLeft = viewport.width - size.width - margin, maxTop = viewport.height - size.height - margin;
  if (!rect) return { left: clamp((viewport.width - size.width) / 2, margin, maxLeft), top: clamp((viewport.height - size.height) / 2, margin, maxTop), side: 'center' };

  const cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
  const candidates = {
    bottom: { left: cx - size.width / 2, top: rect.top + rect.height + gap },
    top:    { left: cx - size.width / 2, top: rect.top - size.height - gap },
    right:  { left: rect.left + rect.width + gap, top: cy - size.height / 2 },
    left:   { left: rect.left - size.width - gap, top: cy - size.height / 2 },
  };
  const fits = c => c.left >= margin && c.top >= margin && c.left <= maxLeft && c.top <= maxTop;
  const order = prefer === 'auto' ? ['bottom', 'top', 'right', 'left'] : [prefer, ...['bottom', 'top', 'right', 'left'].filter(s => s !== prefer)];
  for (const side of order) {
    const c = candidates[side];
    // (along the side it may be moved to fit: a popover below a target at the edge of the viewport is shifted, not moved to another side)
    const moved = { left: clamp(c.left, margin, maxLeft), top: clamp(c.top, margin, maxTop) };
    if (side === 'bottom' || side === 'top' ? fits({ left: moved.left, top: c.top }) : fits({ left: c.left, top: moved.top }))
      return { ...moved, side };
  }
  return { left: clamp(cx - size.width / 2, margin, maxLeft), top: clamp(rect.top + rect.height - size.height - margin, margin, maxTop), side: 'inside' };
}

/**
 * Where the popover (size: { width, height }) goes in the bottom-left corner of `rect`, `margin` from its edges, kept inside the viewport.
 * Returns { left, top, side: 'corner' }.
 */
export function placeInCorner(rect, size, viewport, margin = 12) {
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  return {
    left: clamp(rect.left + margin, margin, viewport.width - size.width - margin),
    top:  clamp(rect.top + rect.height - size.height - margin, margin, viewport.height - size.height - margin),
    side: 'corner',
  };
}
