// tooltip.js — the tooltip that shows what a node or an edge of a graph is when the pointer rests on it, and the text of it, for SkeVU and the
// designer. One tooltip for the page: only one element can be pointed at at a time.

import { formatDuration } from './format.js';

var _tip   = null;
var _timer = null;

function ensureTooltip() {
  if (!_tip) {
    _tip = document.createElement('div');
    _tip.className = 'gv-tooltip';
    _tip.style.cssText =
      'position:fixed;display:none;background:rgba(30,30,30,0.92);color:#eee;' +
      'font:12px/1.4 monospace;padding:6px 9px;border-radius:4px;' +
      'white-space:pre;pointer-events:none;z-index:9999;max-width:320px;' +
      'box-shadow:0 2px 8px rgba(0,0,0,0.4);';
    document.body.appendChild(_tip);
  }
  return _tip;
}

/** Show `content` near the pointer at (x, y) (page coordinates), after a short rest. */
export function showTooltip(content, x, y) {
  var tip = ensureTooltip();
  clearTimeout(_timer);
  _timer = setTimeout(function() {
    tip.textContent = content;
    tip.style.display = 'block';
    positionTooltip(x, y);
  }, 500);
}

export function hideTooltip() {
  clearTimeout(_timer);
  if (_tip) _tip.style.display = 'none';
}

/** Move the tooltip with the pointer, to the side of it where there is room. */
export function positionTooltip(x, y) {
  var tip = _tip;
  if (!tip) return;
  var pad = 14;
  tip.style.left = '-9999px';
  tip.style.top  = '-9999px';
  var tw = tip.offsetWidth, th = tip.offsetHeight;
  var left = x + pad, top = y + pad;
  if (left + tw > window.innerWidth  - 8) left = x - tw - pad;
  if (top  + th > window.innerHeight - 8) top  = y - th - pad;
  tip.style.left = left + 'px';
  tip.style.top  = top  + 'px';
}

/** The text for a node, from its data: label, time, size, backend, pattern and place in the source. */
export function nodeTooltipText(d) {
  var lines = [];
  if (d.label) lines.push(d.label);
  var dur = formatDuration(d.duration);
  if (dur) lines.push('time     ' + dur);
  if (d.elements && d.elements.length) lines.push('size     ' + d.elements.join(' \xd7 '));
  if (d.backend) lines.push('backend  ' + d.backend);
  if (d.pattern) lines.push('pattern  ' + d.pattern);
  if (d.file && d.line != null && d.line !== -1)
    lines.push(d.file.split('/').pop().split('\\').pop() + ':' + d.line);
  return lines.join('\n');
}

/** The text for an edge, from its data: the kind of dependence, what it is for and how it is used. */
export function edgeTooltipText(d) {
  var typeLabel = d.type === 'forward-dep'   ? 'Data dependence'
                : d.type === 'anti-dep'      ? 'Anti-dependence'
                : d.type === 'virtual-alias' ? 'Virtual alias: ' + (d.virtual_label || '')
                : d.type === 'prng-dep'      ? 'PRNG dep: '      + (d.prng_label   || '')
                : d.type === 'setup-dep'     ? 'Setup: the kernel is built before the first call'
                : d.type || 'Edge';
  var lines = [typeLabel];
  if (d.label)       lines.push(d.label);
  if (d.access_mode) lines.push('mode  ' + d.access_mode);
  return lines.join('\n');
}
