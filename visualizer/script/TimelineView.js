'use strict';

// TimelineView.js — the timeline pane: the calls, transfers, builds, communication and events of the trace on a time axis.
//
// The model is timeline/model.js, the drawing timeline/draw.js. The content is as wide as the zoom says, in a native scroller (system scrolling and
// inertia), but only a window of it is drawn: OVERSPILL viewports to each side of the visible one, drawn again when the viewport nears its edge.
// What does not depend on the scroll is in layers that stay in the viewport, so that a fast scroll does not blank them.
//
//   .tl-body                   the size of the content
//     .tl-rows                 stripes of the rows; viewport-wide, fixed
//     svg.tl-labels            label column; stays at the left, over the stripes
//     .tl-sticky               labels of bars that start before the plot; viewport-wide, fixed
//     .tl-beyond               after the end of the trace
//     svg.tl-plot              the window: the items; the grid is its background
//     .tl-axis                 stays at the bottom; content-wide, no background; one grid cell that its children share
//       .tl-axis-bg            background and baseline; viewport-wide, fixed
//       svg.tl-axis-svg        the window: tick times; the marks are its background
//       .tl-corner             over the window, stays at the left

import { timelineNodesToShow } from './badges.js';
import { buildTimeline, itemOfNode, stackRows, fitRowHeight, forEachItemIn } from './timeline/model.js';
import { makeView, drawLabels, drawPlot, drawAxis, tickStepPx, scaleToFit, labelColor } from './timeline/draw.js';
import { METRICS as M, palette } from './timeline/style.js';
import { formatDuration } from './timeline/axis.js';
import { Cursors } from './timeline/cursors.js';
import { ZoomBar, MIN_SCALE, MAX_SCALE } from './timeline/zoombar.js';
import { releaseVelocity, decayVelocity, isMoving } from './timeline/fling.js';

/** Viewports drawn beyond the visible one, at each side. */
const OVERSPILL = 2;
/** The window is drawn again, when idle, once the viewport is within this many viewports of its edge. */
const REDRAW_WITHIN = 1;
const IDLE_REDRAW_MS = 120;
const PINCH_SENSITIVITY = 0.08;   // ctrl+wheel: log of the scale per px of delta
const PINCH_DAMPING = 0.6;        // factor on the log of the scale of a pinch (wheel and touch)
const PIN_MS = 600;               // a bar that a gesture started on is kept this long after its last event
const MIN_CONTENT_W = 200;
const DRAG_THRESHOLD = 4;         // px that a press moves before it is a drag and not a click
const MIN_RANGE_DRAG = 8;         // least width of a range that zooms, px
const FLING_SAMPLES = 8;          // last scroll positions of a drag, for the release speed
const FLING_MAX_DT = 50;          // ms: longest frame step
const FLING_BOUND_PX = 1.5;       // px: a scroll short of the target by more than this has reached the end
const MIN_PANE_H = 100;           // the zoom strip needs room
const MAX_PANE_WINDOW = 0.8;      // of the window height: the tallest pane the user can set
const MAX_PANE_FRACTION = 0.45;   // of the window height, at least MAX_PANE_FLOOR; a taller timeline scrolls
const MAX_PANE_FLOOR = 160;
const FAR_JUMP_VIEWPORTS = 3;     // viewports: a scroll to a bar farther away is not animated

const svgEl = (cls) => { const e = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); e.setAttribute('class', cls); return e; };
const isDark = () => document.documentElement.classList.contains('dark-mode');
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export class TimelineView {

  /**
   * @param {Element}  containerEl         The scroller (#gantt-svg-container).
   * @param {object}   [options]
   * @param {Element}  [options.paneEl]    The flex pane; its height is set to fit the timeline.
   * @param {Element}  [options.zoomEl]    Where the zoom strip goes.
   * @param {Element}  [options.resizerEl] The bar above the pane; it is hidden while the pane has no height.
   * @param {number}   [options.initialHeight=0] The height that the user chose, px; 0: the pane fits the trace.
   * @param {Element}  [options.countEl]   Gets the number of bars.
   * @param {number}   [options.initialScale=1]
   * @param {boolean}  [options.initialOpen=true]
   * @param {Function} [options.onPaneToggle]     After open and close.
   * @param {Function} [options.onSettingsChange] After a change that is saved with the settings.
   * @param {Function} [options.onScaleChange]    (scale) after the user zooms.
   * @param {Function} [options.onSimBarClick]    (nodeId, iterIdx) → true if the click is taken.
   * @param {Function} [options.getSimDragging]   () → true while a simulation is dragged: the scroll is not animated.
   * @param {number}   [options.animScrollMs=400] Zero: no animation.
   */
  constructor(containerEl, options) {
    options = options || {};
    if (!containerEl) throw new Error('TimelineView: containerEl is required');
    this._containerEl = containerEl;
    this._paneEl = options.paneEl || null;
    this._countEl = options.countEl || null;
    this._resizerEl = options.resizerEl || null;
    this._chosenHeight = options.initialHeight > 0 ? options.initialHeight : 0;
    this._onPaneToggle = options.onPaneToggle || null;
    this._onSettingsChange = options.onSettingsChange || null;
    this._onScaleChange = options.onScaleChange || null;
    this._onSimBarClick = options.onSimBarClick || null;
    this._getSimDragging = options.getSimDragging || (() => false);
    this._animScrollMs = options.animScrollMs != null ? options.animScrollMs : 400;

    this._scale = clamp(options.initialScale != null ? options.initialScale : 1, MIN_SCALE, MAX_SCALE);
    this._paneOpen = options.initialOpen != null ? options.initialOpen : true;

    this._selectedNodeId = null;
    this._lockedTime = null;   // µs of the trace file, as the data has them
    this._hoverTime = null;
    this._commExpanded = false;
    this._lastTimelineNodes = [];

    this._model = null;
    this._rowH = M.rowH;        // current height of a flexible row
    this._view = null;
    this._win = null;          // { x0, x1 } of the content that is drawn
    this._pal = palette(false);
    this._idleTimer = null;
    this._noSelect = this._noSelect.bind(this);
    this._drag = null;         // the press that may become a pan or a range zoom: { id, x, y, left, top, range, active }
    this._suppressClick = false;
    this._samples = [];        // scroll positions of the last moves of a pan
    this._flingFrame = 0;
    this._stickyLabels = new Map();  // item index → { svg, div }: bars that start before the plot, labelled in the sticky layer
    this._labelEls = null;       // item index → label svg of the window; built on first use
    this._pinned = null;       // an item that a wheel or touch gesture started on; see _pin
    this._pinTimer = null;

    this._provider = null;
    this._boundOnData = null;
    this._boundOnSelect = null;
    this._boundOnClear = null;

    this._zoomBar = options.zoomEl ? new ZoomBar(options.zoomEl, { onChange: s => this._zoomFromBar(s) }) : null;
    if (this._zoomBar) this._zoomBar.setValue(this._scale);

    containerEl.addEventListener('scroll', () => this._onScroll(), { passive: true });
    containerEl.addEventListener('click', ev => this._onClick(ev));
    containerEl.addEventListener('mousemove', ev => this._onMouseMove(ev));
    containerEl.addEventListener('mouseleave', () => this._hideHover());
    containerEl.addEventListener('mousedown', ev => this._onMouseDown(ev));
    containerEl.addEventListener('pointerdown', ev => this._onPointerDown(ev));
    containerEl.addEventListener('pointermove', ev => this._onPointerMove(ev));
    containerEl.addEventListener('pointerup', ev => this._endDrag(true, ev));
    containerEl.addEventListener('pointercancel', ev => this._endDrag(false, ev));
    if (typeof ResizeObserver !== 'undefined') {
      let width = containerEl.clientWidth, height = containerEl.clientHeight;
      new ResizeObserver(() => {
        const wider = containerEl.clientWidth !== width, taller = containerEl.clientHeight !== height;
        width = containerEl.clientWidth;
        height = containerEl.clientHeight;
        if (!this._paneOpen || !this._model) return;
        if (wider) this._layout(this._centerAnchor());
        else if (taller && this._fitRows()) this._layout(null);   // only the rows change: the scroll stays
      }).observe(containerEl);
    }
  }

  // ── Provider ────────────────────────────────────────────────────────────────

  /** 'data' redraws; 'select' and 'clear' follow the other views (this view's clicks have source 'gantt'). */
  attachProvider(provider) {
    if (this._provider) this.detachProvider();
    this._provider = provider;

    this._boundOnData = rawData => {
      this._lastTimelineNodes = rawData.timeline_nodes || [];
      if (this._paneOpen && this._lastTimelineNodes.length) this._render();
    };
    this._boundOnSelect = ev => {
      if (ev.source === 'gantt') { this.highlight(ev.nodeId, false); return; }
      const targetId = ev.focusId || ev.nodeId;
      if (ev.iterIdx != null && ev.iterIdx > 0) this.scrollToInterval(targetId, ev.iterIdx);
      else this.highlight(targetId, true);
    };
    this._boundOnClear = () => this.clearSelection();

    provider.on('data', this._boundOnData);
    provider.on('select', this._boundOnSelect);
    provider.on('clear', this._boundOnClear);
    return this;
  }

  detachProvider() {
    if (!this._provider) return this;
    this._provider.off('data', this._boundOnData);
    this._provider.off('select', this._boundOnSelect);
    this._provider.off('clear', this._boundOnClear);
    this._boundOnData = this._boundOnSelect = this._boundOnClear = null;
    this._provider = null;
    return this;
  }

  // ── Pane ────────────────────────────────────────────────────────────────────

  get scale() { return this._scale; }
  get paneOpen() { return this._paneOpen; }

  open() {
    this._paneOpen = true;
    if (this._onPaneToggle) this._onPaneToggle();
    if (this._lastTimelineNodes.length) this._render();
    if (this._onSettingsChange) this._onSettingsChange();
  }

  close() {
    this._paneOpen = false;
    this.collapse();
    if (this._onPaneToggle) this._onPaneToggle();
    if (this._onSettingsChange) this._onSettingsChange();
  }

  toggle() { if (this._paneOpen) this.close(); else this.open(); }

  collapse() { this._applyPaneHeight(0); }

  get minPaneHeight() { return MIN_PANE_H; }
  get maxPaneHeight() { return Math.round(window.innerHeight * MAX_PANE_WINDOW); }
  get chosenHeight() { return this._chosenHeight; }

  /** px chosen by the user; 0: fit the trace. */
  setPaneHeight(px) {
    this._chosenHeight = px > 0 ? px : 0;
    if (this._paneOpen && this._model) this._applyPaneHeight(this._paneHeightFor(this._model));
  }

  _paneHeightFor(model) {
    if (this._chosenHeight > 0) return clamp(this._chosenHeight, MIN_PANE_H, this.maxPaneHeight);
    // a timeline of many ranks is taller than this and scrolls
    return clamp(model.naturalHeight, MIN_PANE_H, Math.max(MAX_PANE_FLOOR, Math.round(window.innerHeight * MAX_PANE_FRACTION)));
  }

  _applyPaneHeight(px) {
    if (this._paneEl) this._paneEl.style.setProperty('--gantt-height', px + 'px');
    if (this._resizerEl) this._resizerEl.hidden = !(px > 0);
  }

  // ── Zoom ────────────────────────────────────────────────────────────────────

  /** Zoom 1 fits the trace in the pane; the middle of the view stays. */
  setScale(val) {
    const scale = clamp(parseFloat(val), MIN_SCALE, MAX_SCALE);
    if (!(scale > 0) || scale === this._scale) return;
    this._zoomTo(scale, this._centerAnchor());
  }

  /** `anchor` { t, vx }: the time t (µs after the start of the trace) stays at x = vx of the viewport. */
  _zoomTo(scale, anchor) {
    this._scale = clamp(scale, MIN_SCALE, MAX_SCALE);
    if (this._zoomBar) this._zoomBar.setValue(this._scale);
    if (this._paneOpen && this._model) this._layout(anchor);
  }

  _userZoomed() {
    if (this._onScaleChange) this._onScaleChange(this._scale);
    if (this._onSettingsChange) this._onSettingsChange();
  }

  _zoomFromBar(scale) {
    this._zoomTo(scale, this._centerAnchor());
    this._userZoomed();
  }

  _plotCenterVx() { return M.labelW + (this._containerEl.clientWidth - M.labelW) / 2; }

  _centerAnchor() {
    if (!this._view) return null;
    const vx = this._plotCenterVx();
    return { t: this._view.timeAt(this._containerEl.scrollLeft + vx), vx };
  }

  /** Safari goes back or forward on a horizontal scroll past the end: this stops it. */
  _holdAtEdge(ev) {
    const c = this._containerEl;
    if (Math.abs(ev.deltaX) <= Math.abs(ev.deltaY)) return;
    if ((ev.deltaX < 0 && c.scrollLeft <= 0) || (ev.deltaX > 0 && c.scrollLeft >= c.scrollWidth - c.clientWidth - 1)) ev.preventDefault();
  }

  /** Ctrl+wheel (trackpad pinch) and touch pinch zoom at the pointer. Call once. */
  initPinchZoom() {
    const container = this._containerEl;
    const anchorAt = clientX => {
      const vx = clientX - container.getBoundingClientRect().left;
      return this._view ? { t: this._view.timeAt(container.scrollLeft + vx), vx } : null;
    };
    const zoom = (scale, anchor) => {
      this._zoomTo(scale, anchor);
      this._userZoomed();
    };

    // the deltas of a frame are added: one redraw per frame
    let accum = 0, anchor = null, raf = null;
    container.addEventListener('wheel', ev => {
      this._pin(ev.target);
      this._stopFling();
      if (!ev.ctrlKey) { this._holdAtEdge(ev); return; }
      ev.preventDefault();
      accum += ev.deltaY * (ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? 400 : 1);
      if (raf !== null) return;
      anchor = anchorAt(ev.clientX);
      raf = requestAnimationFrame(() => {
        raf = null;
        if (anchor) zoom(this._scale * Math.exp(-accum * PINCH_SENSITIVITY * PINCH_DAMPING), anchor);
        accum = 0;
      });
    }, { passive: false });

    let pinch = null;
    const dist = t => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
    const mid = t => (t[0].clientX + t[1].clientX) / 2;
    container.addEventListener('touchstart', ev => {
      this._pin(ev.target);
      if (ev.touches.length !== 2) { pinch = null; return; }
      ev.preventDefault();
      pinch = { dist: dist(ev.touches), scale: this._scale, anchor: anchorAt(mid(ev.touches)) };
    }, { passive: false });
    container.addEventListener('touchmove', ev => {
      if (ev.touches.length !== 2 || !pinch || !pinch.anchor) return;
      ev.preventDefault();
      zoom(pinch.scale * Math.pow(dist(ev.touches) / pinch.dist, PINCH_DAMPING), { t: pinch.anchor.t, vx: mid(ev.touches) - container.getBoundingClientRect().left });
    }, { passive: false });
    container.addEventListener('touchend', ev => { this._pin(this._pinned); if (ev.touches.length < 2) pinch = null; });
    container.addEventListener('touchcancel', () => { pinch = null; });
  }

  // ── Selection ───────────────────────────────────────────────────────────────

  /** Marks the bars of a node; `scroll` (default true) moves the view to the first. */
  highlight(nodeId, scroll) {
    this._selectedNodeId = nodeId;
    this._markSelected();
    if (scroll !== false && this._model) this._scrollToItem(itemOfNode(this._model, nodeId, 0));
  }

  /** Marks the bars of a node; scrolls to the given iteration (the first bar if it has none). */
  scrollToInterval(nodeId, iterIdx) {
    this._selectedNodeId = nodeId;
    this._markSelected();
    if (this._model) this._scrollToItem(itemOfNode(this._model, nodeId, iterIdx));
  }

  clearSelection() {
    this._selectedNodeId = null;
    this._markSelected();
  }

  _markSelected() {
    if (!this._plot || !this._model) return;
    for (const e of this._plot.querySelectorAll('[data-i]'))
      e.classList.toggle('gantt-selected', this._model.items[+e.dataset.i].id === this._selectedNodeId);
  }

  _scrollToItem(item) {
    if (!item || !this._view) return;
    const container = this._containerEl;
    this._stopFling();
    const centre = (this._view.x(item.t0) + this._view.x(item.t1)) / 2;
    const left = Math.max(0, centre - this._plotCenterVx());
    const near = Math.abs(left - container.scrollLeft) < FAR_JUMP_VIEWPORTS * container.clientWidth;
    const animate = this._animScrollMs > 0 && !this._getSimDragging() && near;
    container.scrollTo({ left, behavior: animate ? 'smooth' : 'auto' });
  }

  // ── Redraw ──────────────────────────────────────────────────────────────────

  /** Rebuilds the model (colours, palette or shown rows may have changed); the middle of the view stays. */
  reRenderPreservingScroll() {
    if (!this._lastTimelineNodes.length) return;
    this._render(this._centerAnchor());
  }

  _render(anchor) {
    const container = this._containerEl;
    const provider = this._provider;
    const nodes = timelineNodesToShow(this._lastTimelineNodes, !!provider && provider.settings.get('show_regions'));
    this._model = buildTimeline(nodes, { colorOf: d => this._colorOf(d), commExpanded: this._commExpanded });
    if (!this._model) {
      container.textContent = '';
      this._body = null;
      const msg = document.createElement('p');
      msg.id = 'timeline-no-events-msg';
      msg.textContent = 'No timed events to display.';
      container.appendChild(msg);
      return;
    }
    this._rowH = M.rowH;
    this._ensureDom();
    if (this._countEl) this._countEl.textContent = this._model.count;
    this._layout(anchor);
  }

  _colorOf(d) {
    const p = this._provider;
    if (!p) return '#778';
    const mode = p.settings.get('node-color-call');
    if (d.type === 'skeleton_call') {
      if (mode === 'backend') return p.backendColor(d.backend) || '#778';
      if (mode === 'pattern') return p.patternColor(d.pattern) || '#778';
    }
    return p.spectrumColor(p.normFromData('node-color-call', d));
  }

  _ensureDom() {
    if (this._body && this._body.isConnected) return;
    const container = this._containerEl;
    container.textContent = '';
    const body = document.createElement('div');
    body.className = 'tl-body';
    this._labels = svgEl('tl-labels');
    this._rows = document.createElement('div');
    this._rows.className = 'tl-rows';
    this._sticky = document.createElement('div');
    this._sticky.className = 'tl-sticky';
    this._beyond = document.createElement('div');
    this._beyond.className = 'tl-beyond';
    this._plot = svgEl('tl-plot');
    const axis = document.createElement('div');
    axis.className = 'tl-axis';
    this._corner = document.createElement('div');
    this._corner.className = 'tl-corner';
    this._axisSvg = svgEl('tl-axis-svg');
    this._axisBg = document.createElement('div');
    this._axisBg.className = 'tl-axis-bg';
    axis.append(this._axisBg, this._axisSvg, this._corner);
    this._rangeEl = document.createElement('div');
    this._rangeEl.className = 'tl-range';
    body.append(this._rows, this._labels, this._sticky, this._beyond, this._plot, this._rangeEl, axis);
    container.appendChild(body);
    this._body = body;
    this._axisBox = axis;
    for (const svg of [this._plot, this._axisSvg]) {
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute('class', 'tl-items');
      svg.appendChild(g);
    }
    this._plotItems = this._plot.firstChild;
    this._axisItems = this._axisSvg.firstChild;
    this._cursors = new Cursors(this._plot, this._axisSvg, () => ({ min: M.labelW, max: this._view.contentW }));
  }

  /** Sizes the content from the zoom and the pane, then draws the window. `anchor`: see _zoomTo. */
  _layout(anchor) {
    this._stopFling();
    const container = this._containerEl, model = this._model;
    this._fitRows();
    const contentW = Math.max(container.clientWidth || 400, MIN_CONTENT_W) * this._scale;
    this._view = makeView(model, contentW);
    this._pal = palette(isDark());

    const body = this._body;
    body.style.width = contentW + 'px';
    body.style.setProperty('--tl-h', model.height + 'px');
    this._labels.setAttribute('width', M.labelW);
    this._labels.setAttribute('height', model.rowsHeight);
    this._labels.innerHTML = drawLabels(model, this._pal);
    this._layoutBackground();
    this._plot.setAttribute('height', model.rowsHeight);
    this._axisSvg.setAttribute('height', M.axisH);
    this._cursors.setPalette(this._pal);
    this._cursors.perUs = this._view.pxPerUs;

    if (anchor) container.scrollLeft = Math.max(0, this._view.x(model.range.minT + anchor.t) - anchor.vx);
    this._hideHover();
    this._win = null;
    this._drawWindow();
    this._placeLock();

    this._applyPaneHeight(this._paneHeightFor(model));
  }

  /** Fits the flexible rows to the pane; returns whether their height changed. */
  _fitRows() {
    const rowH = fitRowHeight(this._model, this._containerEl.clientHeight - M.axisH);
    if (rowH === this._rowH) return false;
    this._rowH = rowH;
    stackRows(this._model, rowH);
    return true;
  }

  /** The layers that do not move with the window: row stripes and axis background (fixed in the viewport), the end of the trace. */
  _layoutBackground() {
    const { _model: model, _view: view, _pal: pal } = this;
    this._rows.replaceChildren(...model.rows.map(row => {
      const e = document.createElement('div');
      Object.assign(e.style, { top: row.y + 'px', height: row.h + 'px', background: pal[row.shade] });
      return e;
    }));
    Object.assign(this._rows.style, { width: this._containerEl.clientWidth + 'px', height: model.rowsHeight + 'px' });
    this._labels.style.marginTop = -model.rowsHeight + 'px';
    Object.assign(this._sticky.style, { width: this._containerEl.clientWidth + 'px', height: model.rowsHeight + 'px', marginTop: -model.rowsHeight + 'px' });
    Object.assign(this._beyond.style, { left: view.endX + 'px', width: (view.contentW - view.endX) + 'px', height: model.rowsHeight + 'px',
      background: pal.beyond, borderLeft: '1px solid ' + pal.end, boxSizing: 'border-box' });
    Object.assign(this._axisBg.style, { width: this._containerEl.clientWidth + 'px', height: M.axisH + 'px', background: pal.rowEven, borderTop: '1px solid ' + pal.axis });
    Object.assign(this._axisBox.style, { width: view.contentW + 'px', height: M.axisH + 'px' });
    Object.assign(this._corner.style, { width: M.labelW + 'px', background: pal.rowEven });
  }

  /** Draws a line at every tick as the background of a window svg; `lineH` px tall, null: the full height. */
  _tickBackground(svg, color, win, lineH) {
    const step = tickStepPx(this._view);
    Object.assign(svg.style, {
      backgroundImage: `linear-gradient(90deg, ${color} 0 1px, transparent 1px)`,
      backgroundSize: `${step}px ${lineH == null ? '100%' : lineH + 'px'}`,
      backgroundRepeat: 'repeat-x',
      backgroundPosition: `${M.labelW - win.x0}px 0`,
    });
  }

  // ── Window ──────────────────────────────────────────────────────────────────

  /** The viewport plus OVERSPILL viewports at each side. */
  _wantedWindow() {
    const c = this._containerEl, w = this._view.contentW, vw = c.clientWidth;
    return { x0: Math.max(0, Math.floor(c.scrollLeft - OVERSPILL * vw)), x1: Math.min(w, Math.ceil(c.scrollLeft + vw + OVERSPILL * vw)) };
  }

  _drawWindow() {
    const win = this._wantedWindow(), view = this._view, model = this._model;
    this._win = win;
    const width = win.x1 - win.x0;
    for (const svg of [this._plot, this._axisSvg]) {
      svg.style.left = win.x0 + 'px';
      svg.setAttribute('width', width);
    }
    this._tickBackground(this._plot, this._pal.grid, win, null);
    this._tickBackground(this._axisSvg, this._pal.tickMark, win, M.tickLen);
    this._replaceItems(drawPlot(model, view, win, this._pal, this._selectedNodeId));
    this._labelEls = null;
    this._sticky.replaceChildren();
    this._stickyLabels = new Map();
    this._followLabels();
    this._axisItems.innerHTML = drawAxis(model, view, win, this._pal);
    this._cursors.origin = win.x0;
    this._cursors.update(model.rowsHeight);
  }

  /**
   * A wheel or touch gesture is delivered to the element it started on, and to nothing if that element is removed: a zoom that starts on a bar
   * would stop at the first redraw. The element stays, hidden, until PIN_MS after the last event.
   */
  _pin(el) {
    if (el && el.parentNode === this._plotItems) this._pinned = el;
    if (!this._pinned) return;
    clearTimeout(this._pinTimer);
    this._pinTimer = setTimeout(() => { this._pinned?.remove(); this._pinned = null; }, PIN_MS);
  }

  _replaceItems(markup) {
    const g = this._plotItems, keep = this._pinned && this._pinned.parentNode === g ? this._pinned : null;
    for (let c = g.firstChild; c;) { const next = c.nextSibling; if (c !== keep) g.removeChild(c); c = next; }
    if (keep) {
      keep.removeAttribute('data-i');
      keep.setAttribute('visibility', 'hidden');
      keep.setAttribute('class', 'pointer-events-none');
    }
    g.insertAdjacentHTML('beforeend', markup);
  }

  /**
   * A bar that starts before the left edge of the plot has its label in the sticky layer at the edge, so that it does not move with the scroll (the
   * label in the window is hidden). A scroll only updates which bars these are and where each label ends.
   */
  _followLabels() {
    const { _model: model, _view: view, _win: win, _containerEl: c } = this;
    if (!model || !view || !win || !this._plot) return;
    const edge = c.scrollLeft + M.labelW;
    const t = view.minT + view.timeAt(edge);
    // the last drawn (innermost region) of each row: the labels under it would overlap
    const crossing = new Map();
    model.rows.forEach((row, r) => {
      let top = null;
      forEachItemIn(model, r, t, t, it => {
        if (it.shape === 'bar' && it.label && view.x(it.t0) < edge && view.x(it.t1) - view.x(it.t0) > M.minLabelW) top = it;
      });
      if (top) crossing.set(top.idx, top);
    });
    for (const [idx, pinned] of this._stickyLabels)
      if (!crossing.has(idx)) { pinned.div.remove(); pinned.svg?.firstChild.removeAttribute('visibility'); this._stickyLabels.delete(idx); }
    if (crossing.size && !this._labelEls)
      this._labelEls = new Map([...this._plot.querySelectorAll('svg[data-li]')].map(e => [+e.dataset.li, e]));
    for (const [idx, it] of crossing) {
      let pinned = this._stickyLabels.get(idx);
      if (!pinned) {
        const div = document.createElement('div');
        const row = model.rows[it.row];
        div.textContent = it.label;
        Object.assign(div.style, { top: row.y + M.barPad + 'px', height: row.h - 2 * M.barPad + 'px', left: M.labelW + 'px', paddingLeft: M.labelPadX + 'px',
          color: labelColor(it, this._pal), opacity: it.hidden ? 0.3 : '' });
        this._sticky.appendChild(div);
        const svg = this._labelEls && this._labelEls.get(idx);
        if (svg) svg.firstChild.setAttribute('visibility', 'hidden');
        pinned = { svg, div };
        this._stickyLabels.set(idx, pinned);
      }
      // the label ends where the bar does
      pinned.div.style.width = Math.max(0, Math.min(view.x(it.t1) - 2 - edge, c.clientWidth - M.labelW)) + 'px';
    }
  }

  _onScroll() {
    this._followLabels();
    if (!this._win || !this._model) return;
    const c = this._containerEl, vw = c.clientWidth, left = c.scrollLeft, w = this._view.contentW;
    const inside = (from, to) => this._win.x0 <= Math.max(0, from) && this._win.x1 >= Math.min(w, to);
    if (!inside(left, left + vw)) { this._redrawNow(); return; }
    if (!inside(left - REDRAW_WITHIN * vw, left + vw + REDRAW_WITHIN * vw) && this._idleTimer === null)
      this._idleTimer = setTimeout(() => this._redrawNow(), IDLE_REDRAW_MS);
  }

  _redrawNow() {
    clearTimeout(this._idleTimer);
    this._idleTimer = null;
    this._drawWindow();
  }

  // ── Cursors and clicks ──────────────────────────────────────────────────────

  _xOf(ev) { return ev.clientX - this._containerEl.getBoundingClientRect().left; }   // in the viewport

  _timeLabel(x) { return '+' + formatDuration(this._view.timeAt(x)); }

  _inPlot(vx) { return vx >= M.labelW && vx <= this._containerEl.clientWidth && this._containerEl.scrollLeft + vx <= this._view.endX; }

  _hideHover() {
    this._hoverTime = null;
    this._containerEl.classList.remove('tl-over-lock');
    if (this._cursors) this._cursors.hideHover();
  }

  _onMouseMove(ev) {
    if (!this._view || !this._cursors || (this._drag && this._drag.active)) return;
    const vx = this._xOf(ev);
    if (!this._inPlot(vx)) { this._hideHover(); return; }
    const x = this._containerEl.scrollLeft + vx;
    this._containerEl.classList.toggle('tl-over-lock', this._cursors.hitsLock(x, !!ev.target.closest('.tl-axis')));
    this._hoverTime = this._model.range.minT + this._view.timeAt(x);
    this._pendingX = x;
    if (this._moveFrame) return;
    this._moveFrame = requestAnimationFrame(() => {
      this._moveFrame = 0;
      if (this._hoverTime === null) return;
      this._cursors.show('hover', this._pendingX, this._timeLabel(this._pendingX));
    });
  }

  _placeLock() {
    if (!this._cursors) return;
    if (this._lockedTime === null) { this._cursors.unlock(); return; }
    const x = this._view.x(this._lockedTime);
    if (x < M.labelW || x > this._view.endX) { this._cursors.unlock(); return; }
    this._cursors.show('lock', x, this._timeLabel(x));
  }

  // ── Drag: pan; with Shift, zoom to the range ────────────────────────────────

  /** A press in the plot never starts a text selection: it would extend over the page once the pointer leaves the timeline. */
  _onMouseDown(ev) {
    const c = this._containerEl, r = c.getBoundingClientRect();
    if (ev.button !== 0 || ev.clientX - r.left >= c.clientWidth || ev.clientY - r.top >= c.clientHeight) return;   // not a scrollbar
    ev.preventDefault();
    window.getSelection()?.removeAllRanges();
  }

  _noSelect(ev) { ev.preventDefault(); }

  _onPointerDown(ev) {
    this._stopFling();
    if (ev.button !== 0 || ev.pointerType === 'touch' || !this._model) return;   // a touch screen scrolls by itself
    const c = this._containerEl, r = c.getBoundingClientRect();
    if (ev.clientX - r.left >= c.clientWidth || ev.clientY - r.top >= c.clientHeight) return;   // a scrollbar
    if (ev.target.closest && ev.target.closest('[data-comm-toggle]')) return;
    document.addEventListener('selectstart', this._noSelect);
    this._drag = { id: ev.pointerId, x: ev.clientX, y: ev.clientY, left: c.scrollLeft, top: c.scrollTop, range: ev.shiftKey, active: false, a: 0, b: 0 };
  }

  _onPointerMove(ev) {
    const d = this._drag, c = this._containerEl;
    if (!d || ev.pointerId !== d.id) return;
    const dx = ev.clientX - d.x, dy = ev.clientY - d.y;
    if (!d.active) {
      if (Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
      d.active = true;
      c.setPointerCapture(d.id);
      c.classList.add(d.range ? 'tl-ranging' : 'tl-panning');
      this._hideHover();
      if (d.range) {
        this._dragKey = e => { if (e.key === 'Escape') this._endDrag(false); };
        document.addEventListener('keydown', this._dragKey);
      }
    }
    if (!d.range) {
      c.scrollLeft = d.left - dx;
      c.scrollTop = d.top - dy;
      this._samples.push({ t: ev.timeStamp, x: c.scrollLeft, y: c.scrollTop });
      if (this._samples.length > FLING_SAMPLES) this._samples.shift();
      return;
    }
    const left = c.getBoundingClientRect().left, inPlot = x => clamp(x, M.labelW, this._view.endX);
    d.a = inPlot(d.x - left + d.left);
    d.b = inPlot(ev.clientX - left + c.scrollLeft);
    Object.assign(this._rangeEl.style, { display: 'block', left: Math.min(d.a, d.b) + 'px', width: Math.abs(d.b - d.a) + 'px', height: this._model.rowsHeight + 'px' });
  }

  _endDrag(commit, ev) {
    const d = this._drag;
    if (!d) return;
    this._drag = null;
    document.removeEventListener('selectstart', this._noSelect);
    const c = this._containerEl;
    if (this._dragKey) { document.removeEventListener('keydown', this._dragKey); this._dragKey = null; }
    c.classList.remove('tl-panning', 'tl-ranging');
    this._rangeEl.style.display = 'none';
    if (!d.active) return;
    // the click that follows the release belongs to the drag
    this._suppressClick = true;
    setTimeout(() => { this._suppressClick = false; }, 0);
    try { c.releasePointerCapture(d.id); } catch (e) { /* released already */ }
    if (commit && d.range && Math.abs(d.b - d.a) >= MIN_RANGE_DRAG) this._zoomToRange(d.a, d.b);
    if (commit && !d.range && ev) this._fling(releaseVelocity(this._samples, ev.timeStamp));
    this._samples = [];
  }

  /** Coasts after the release; any new input stops it. `v`: px/ms of the scroll position. */
  _fling(v) {
    if (!isMoving(v)) return;
    const c = this._containerEl;
    let x = c.scrollLeft, y = c.scrollTop, last = performance.now();
    const step = now => {
      const dt = Math.min(now - last, FLING_MAX_DT);
      last = now;
      x += v.x * dt;
      y += v.y * dt;
      c.scrollLeft = x;
      c.scrollTop = y;
      // the end of the scroll range stops that direction
      if (Math.abs(c.scrollLeft - x) > FLING_BOUND_PX) { x = c.scrollLeft; v = { x: 0, y: v.y }; }
      if (Math.abs(c.scrollTop - y) > FLING_BOUND_PX) { y = c.scrollTop; v = { x: v.x, y: 0 }; }
      v = decayVelocity(v, dt);
      this._flingFrame = isMoving(v) ? requestAnimationFrame(step) : 0;
    };
    this._flingFrame = requestAnimationFrame(step);
  }

  _stopFling() {
    if (this._flingFrame) cancelAnimationFrame(this._flingFrame);
    this._flingFrame = 0;
  }

  /** Zooms so that the range between the content x a and b fills the plot, centred. */
  _zoomToRange(a, b) {
    const view = this._view, dt = Math.abs(view.timeAt(b) - view.timeAt(a));
    const mid = (view.timeAt(a) + view.timeAt(b)) / 2;
    this._zoomTo(scaleToFit(this._model, this._containerEl.clientWidth, dt), { t: mid, vx: this._plotCenterVx() });
    this._userZoomed();
  }

  _onClick(ev) {
    if (this._suppressClick) { this._suppressClick = false; return; }
    if (!this._model || !ev.target.closest) return;
    if (ev.target.closest('[data-comm-toggle]')) {
      this._commExpanded = !this._commExpanded;
      this._render(this._centerAnchor());
      return;
    }
    const barEl = ev.target.closest('[data-i]');
    if (barEl) {
      const item = this._model.items[+barEl.dataset.i];
      if (typeof this._onSimBarClick === 'function' && this._onSimBarClick(item.id, item.iter)) return;
      if (this._provider) this._provider._emit('select', { nodeId: item.id, source: 'gantt', iterIdx: item.iter, rank: item.rank });
      return;
    }
    if (!ev.target.closest('.tl-body')) return;
    const vx = this._xOf(ev);
    if (!this._inPlot(vx)) return;
    const x = this._containerEl.scrollLeft + vx;
    // a click on the marker removes it, so a double click anywhere clears it; any other click is a background click, as in the graph
    if (this._cursors.hitsLock(x, !!ev.target.closest('.tl-axis'))) {
      this._lockedTime = null;
    } else {
      this._lockedTime = this._model.range.minT + this._view.timeAt(x);
      if (this._provider) this._provider._emit('clear', { background: true });
    }
    this._placeLock();
  }
}
