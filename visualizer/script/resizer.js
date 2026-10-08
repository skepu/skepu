// resizer.js — a bar that is dragged to change the size of a pane (the gutter between the graph and the right pane, the bar between the source
// and the info pane), for SkeVU and the designer. The pointer drags it; the arrow keys move it when it has the focus.

/**
 * @param {HTMLElement} bar
 * @param {object} options
 * @param {'x'|'y'} options.axis      'x': a bar between panes side by side; 'y': one above the other
 * @param {number}  [options.direction=-1]  -1: moving the bar left (x) or up (y) makes the pane larger; 1: the other way
 * @param {function} options.size     () => the size of the pane now, in px
 * @param {function} options.setSize  (px) => set the size of the pane (the bar clamps it between `min` and `max`)
 * @param {number}  [options.min=0]
 * @param {function} [options.max]    () => the largest size, in px
 * @param {number}  [options.step=30] the size that an arrow key changes it by
 * @param {function} [options.onStart]  a drag begins
 * @param {function} [options.onEnd]    a drag ends, or a key moved the bar
 */
export function attachResizer(bar, options) {
  var horizontal = options.axis === 'x';
  var direction  = options.direction == null ? -1 : options.direction;
  var min = options.min || 0;
  var step = options.step || 30;
  var clamp = function(px) { return Math.round(Math.max(min, Math.min(px, options.max ? options.max() : Infinity))); };

  bar.setAttribute('role', 'separator');
  bar.setAttribute('aria-orientation', horizontal ? 'vertical' : 'horizontal');
  bar.tabIndex = 0;
  bar.style.touchAction = 'none';

  bar.addEventListener('pointerdown', function(event) {
    event.preventDefault();
    var start = horizontal ? event.clientX : event.clientY, startSize = options.size();
    bar.setPointerCapture(event.pointerId);
    document.body.style.cursor = horizontal ? 'col-resize' : 'ns-resize';
    if (options.onStart) options.onStart();
    function move(e) { options.setSize(clamp(startSize + direction * ((horizontal ? e.clientX : e.clientY) - start))); }
    function up() {
      bar.removeEventListener('pointermove', move);
      bar.removeEventListener('pointerup', up);
      bar.removeEventListener('pointercancel', up);
      document.body.style.cursor = '';
      if (options.onEnd) options.onEnd();
    }
    bar.addEventListener('pointermove', move);
    bar.addEventListener('pointerup', up);
    bar.addEventListener('pointercancel', up);
  });

  bar.addEventListener('keydown', function(event) {
    var grow = horizontal ? 'ArrowLeft' : 'ArrowUp', shrink = horizontal ? 'ArrowRight' : 'ArrowDown';
    if (event.key !== grow && event.key !== shrink) return;
    event.preventDefault();
    var more = (event.key === grow) === (direction < 0);       // does the key make the pane larger
    options.setSize(clamp(options.size() + (more ? step : -step)));
    if (options.onEnd) options.onEnd();
  });
}
