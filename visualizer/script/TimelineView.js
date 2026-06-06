// TimelineView.js — Gantt / timeline chart view class.
//
// Renders per-iteration timing data as a horizontal timeline SVG inside a
// scroll container element.
//
// ── Usage ─────────────────────────────────────────────────────────────────────
//
//   const timeline = new TimelineView(containerEl, {
//       paneEl:           document.getElementById('gantt-pane'),
//       scaleSliderEl:    document.getElementById('gantt-scale'),
//       scaleLabelEl:     document.getElementById('gantt-scale-label'),
//       countEl:          document.getElementById('gantt-count'),
//       initialScale:     1,
//       initialOpen:      true,
//       onPaneToggle:     () => syncExternalMenuIcons(),
//       onSettingsChange: () => saveSettings(),
//       onSimBarClick:    (nodeId, iterIdx) => simEngine.ganttClick(nodeId, iterIdx),
//       getSimDragging:   () => simEngine.isDragging,
//   });
//
//   timeline.attachProvider(provider);   // registers 'data' listener; auto-renders
//
//   // Via provider 'select' / 'clear' events (or directly from simulation.js):
//   timeline.highlight('node-uuid', /* scroll */ true);
//   timeline.clearSelection();
//
//   // From SourceCodeView badge / InfoView duration-plot clicks:
//   timeline.scrollToInterval('node-uuid', iterIdx);

'use strict';

// ── Module-local layout constants ────────────────────────────────────────────
// These are implementation details of the timeline chart; callers have no reason
// to inspect or override them.  animScrollMs is the exception — it is passed
// via options.animScrollMs so it can match the scroll duration used by other
// views in the same app (set in graph.js from state.js's ANIM_SCROLL_MS).

const _LW               = 72;    // label column width (px)
const _PAD_RIGHT        = 90;    // right-side padding (room for "end of trace" label)
const _AXIS_H           = 36;    // bottom axis strip height; also the regions row height
const _ROW_H            = 28;    // height of each backend / transfer / event row
const _ROW_PAD          = 3;     // top/bottom padding inside each row bar
const _MIN_LABEL_W      = 24;    // minimum bar width before drawing an inline label
const _TICK_PX          = 90;    // target pixel spacing between time-axis ticks
const _PINCH_SENSITIVITY = 0.08; // ctrl+wheel / touch-pinch scale speed (exp factor per px)

// ── Helpers ───────────────────────────────────────────────────────────────────

function _isDark() {
  return document.documentElement.classList.contains('dark-mode');
}

function _ganttRegionFillDark(depth) {
  var r = Math.min(185, 80 + depth * 25);
  var g = Math.min(140, 60 + depth * 18);
  var b = Math.min(70,  28 + depth * 10);
  return 'rgba(' + r + ',' + g + ',' + b + ',0.60)';
}

// ── TimelineView ──────────────────────────────────────────────────────────────

export class TimelineView {

  // ── Constructor ─────────────────────────────────────────────────────────────

  /**
   * @param {Element}  containerEl        The scrollable SVG host (e.g. #gantt-svg-container).
   * @param {object}   [options]
   * @param {Element}  [options.paneEl]          Flex pane whose height is set to fit the SVG.
   * @param {Element}  [options.scaleSliderEl]   Range input for x-axis scale.
   * @param {Element}  [options.scaleLabelEl]    Text label showing current scale value.
   * @param {Element}  [options.countEl]         Element whose textContent is set to bar count.
   * @param {number}   [options.initialScale=1]
   * @param {boolean}  [options.initialOpen=true]
   * @param {Function} [options.onPaneToggle]    Called after open/close/toggle.
   * @param {Function} [options.onSettingsChange] Called when scale changes (for saveSettings).
   * @param {Function} [options.onSimBarClick]   (nodeId, iterIdx) → boolean; return true to suppress normal select.
   * @param {Function} [options.getSimDragging]  () → boolean; when true scroll is instant.
   * @param {number}   [options.animScrollMs=400] Scroll animation duration in ms.
   *                   Pass the same value as ANIM_SCROLL_MS from state.js so all
   *                   views share a consistent feel.
   */
  constructor(containerEl, options) {
    options = options || {};
    if (!containerEl) throw new Error('TimelineView: containerEl is required');

    this._containerEl    = containerEl;
    this._paneEl         = options.paneEl         || null;
    this._scaleSliderEl  = options.scaleSliderEl  || null;
    this._scaleLabelEl   = options.scaleLabelEl   || null;
    this._countEl        = options.countEl        || null;

    this._onPaneToggle     = options.onPaneToggle     || null;
    this._onSettingsChange = options.onSettingsChange || null;
    this._onSimBarClick    = options.onSimBarClick    || null;
    this._getSimDragging   = options.getSimDragging   || function() { return false; };
    this._animScrollMs     = (options.animScrollMs != null) ? options.animScrollMs : 400;

    this._scale    = (options.initialScale != null) ? options.initialScale : 1;
    this._paneOpen = (options.initialOpen  != null) ? options.initialOpen  : true;

    // Per-render state (survives re-renders; cleared when a new trace is loaded).
    this._selectedNodeId    = null;
    this._lockedTime        = null;
    this._hoverTime         = null;
    this._timeRange         = null;   // { minT, maxT, range } from last render
    this._lastTimelineNodes = [];

    // Provider binding.
    this._provider       = null;
    this._boundOnData    = null;
    this._boundOnSelect  = null;
    this._boundOnClear   = null;
  }

  // ── Provider attachment ──────────────────────────────────────────────────────

  /**
   * Register on the provider's 'data' and 'select' events.
   * 'data'   — re-renders the chart whenever new trace data arrives.
   * 'select' — highlights bars and scrolls when another view selects a node.
   *            Ignores events with source === 'gantt' (our own bar clicks).
   */
  attachProvider(provider) {
    if (this._provider) this.detachProvider();
    this._provider = provider;

    var self = this;
    this._boundOnData = function(rawData /*, minmax */) {
      self._lastTimelineNodes = rawData.timeline_nodes || [];
      if (self._paneOpen && self._lastTimelineNodes.length)
        self._render(self._lastTimelineNodes);
    };

    this._boundOnSelect = function(ev) {
      if (ev.source === 'gantt') {
        // Our own bar click — apply the visual selection class without scrolling
        // (the user's pointer is already at the right position).
        self.highlight(ev.nodeId, false);
      } else {
        // Selection originated in another view — scroll to the relevant bar.
        var targetId = ev.focusId || ev.nodeId;
        if (ev.iterIdx != null && ev.iterIdx > 0) {
          self.scrollToInterval(targetId, ev.iterIdx);
        } else {
          self.highlight(targetId, true);
        }
      }
    };

    this._boundOnClear = function() {
      self.clearSelection();
    };

    provider.on('data',   this._boundOnData);
    provider.on('select', this._boundOnSelect);
    provider.on('clear',  this._boundOnClear);
    return this;
  }

  detachProvider() {
    if (!this._provider) return this;
    if (this._boundOnData) {
      this._provider.off('data',   this._boundOnData);
      this._boundOnData = null;
    }
    if (this._boundOnSelect) {
      this._provider.off('select', this._boundOnSelect);
      this._boundOnSelect = null;
    }
    if (this._boundOnClear) {
      this._provider.off('clear',  this._boundOnClear);
      this._boundOnClear = null;
    }
    this._provider = null;
    return this;
  }

  // ── Public API — pane lifecycle ──────────────────────────────────────────────

  get scale()    { return this._scale;    }
  get paneOpen() { return this._paneOpen; }

  open() {
    this._paneOpen = true;
    if (this._onPaneToggle) this._onPaneToggle();
    if (this._lastTimelineNodes.length) this._render(this._lastTimelineNodes);
    if (this._onSettingsChange) this._onSettingsChange();
  }

  close() {
    this._paneOpen = false;
    if (this._paneEl) this._paneEl.style.flex = '0 0 0';
    if (this._onPaneToggle) this._onPaneToggle();
    if (this._onSettingsChange) this._onSettingsChange();
  }

  toggle() { if (this._paneOpen) this.close(); else this.open(); }

  setScale(val) {
    this._scale = parseFloat(val);
    if (this._scaleSliderEl) this._scaleSliderEl.value = this._scale;
    if (this._scaleLabelEl)  this._scaleLabelEl.textContent = this._scale.toFixed(1) + '×';
    if (this._onSettingsChange) this._onSettingsChange();
    if (this._paneOpen && this._lastTimelineNodes.length) this._render(this._lastTimelineNodes);
  }

  // ── Public API — selection ───────────────────────────────────────────────────

  /** Highlight all bars for nodeId; optionally animate-scroll to the first one. */
  highlight(nodeId, scroll) {
    this._selectedNodeId = nodeId;
    var firstEl = null;
    var container = this._containerEl;
    container.querySelectorAll('.gantt-bar').forEach(function(b) {
      var sel = b.dataset.nodeId === nodeId;
      b.classList.toggle('gantt-selected', sel);
      if (sel && !firstEl) firstEl = b;
    });
    if (scroll !== false && firstEl) {
      var cx = firstEl.dataset.centerX != null
        ? parseFloat(firstEl.dataset.centerX)
        : parseFloat(firstEl.getAttribute('x') || 0) + parseFloat(firstEl.getAttribute('width') || 0) / 2;
      var dur = this._getSimDragging() ? 0 : this._animScrollMs;
      $(container).stop(true).animate({ scrollLeft: Math.max(0, cx - container.clientWidth / 2) }, dur);
    }
  }

  /** Highlight all bars for nodeId and scroll to the specific iteration bar. */
  scrollToInterval(nodeId, iterIdx) {
    this._selectedNodeId = nodeId;
    var target = null;
    var container = this._containerEl;
    container.querySelectorAll('.gantt-bar').forEach(function(b) {
      var sel = b.dataset.nodeId === nodeId;
      b.classList.toggle('gantt-selected', sel);
      if (sel && target === null && parseInt(b.dataset.intervalIdx || 0) === iterIdx)
        target = b;
    });
    // Fall back to the first bar for this node if the exact iteration was not found.
    if (!target) {
      container.querySelectorAll('.gantt-bar').forEach(function(b) {
        if (!target && b.dataset.nodeId === nodeId) target = b;
      });
    }
    if (!target) return;
    var cx = target.dataset.centerX != null
      ? parseFloat(target.dataset.centerX)
      : parseFloat(target.getAttribute('x') || 0) + parseFloat(target.getAttribute('width') || 0) / 2;
    var dur = this._getSimDragging() ? 0 : this._animScrollMs;
    $(container).stop(true).animate({ scrollLeft: Math.max(0, cx - container.clientWidth / 2) }, dur);
  }

  clearSelection() {
    this._selectedNodeId = null;
    this._containerEl.querySelectorAll('.gantt-bar').forEach(function(b) {
      b.classList.remove('gantt-selected');
    });
  }

  // ── Public API — resize / zoom ───────────────────────────────────────────────

  /** Re-render while keeping the currently visible centre of the bar area stable. */
  reRenderPreservingScroll() {
    if (!this._lastTimelineNodes.length) return;
    var container = this._containerEl;
    var svg    = container.querySelector('svg');
    var svgW   = svg ? parseFloat(svg.getAttribute('width'))  || 0 : 0;
    var barW   = svgW - _LW - _PAD_RIGHT;
    var cFrac  = barW > 0
      ? (container.scrollLeft + container.clientWidth / 2 - _LW) / barW
      : 0.5;
    cFrac = Math.max(0, Math.min(1, cFrac));

    this._render(this._lastTimelineNodes);

    var newSvg  = container.querySelector('svg');
    var newSvgW = newSvg ? parseFloat(newSvg.getAttribute('width')) || 0 : 0;
    var newBarW = newSvgW - _LW - _PAD_RIGHT;
    container.scrollLeft = Math.max(0, _LW + cFrac * newBarW - container.clientWidth / 2);
  }

  /**
   * Zoom the x axis so that the measured interval (locked cursor ↔ hover cursor)
   * fills ~90 % of the visible bar area, then scroll to centre it.
   * Triggered by the Z key in ui.js.
   */
  zoomToMeasuredInterval() {
    if (this._lockedTime === null || this._hoverTime === null) return;
    if (!this._paneOpen || !this._lastTimelineNodes.length || !this._timeRange) return;

    var dt = Math.abs(this._hoverTime - this._lockedTime);
    if (dt < 1) return;

    var container    = this._containerEl;
    var clientW      = container.clientWidth;
    var fracInterval = dt / this._timeRange.range;
    var targetBW     = (clientW - _LW) / (fracInterval * 0.9);
    var newScale     = Math.max(1, (targetBW + _LW + _PAD_RIGHT) / clientW);

    this._scale = newScale;
    if (this._scaleSliderEl) this._scaleSliderEl.value = newScale;
    if (this._scaleLabelEl)  this._scaleLabelEl.textContent = newScale.toFixed(1) + '×';
    if (this._onSettingsChange) this._onSettingsChange();

    this._render(this._lastTimelineNodes);

    // Scroll so the midpoint of the measured interval is centred.
    var tMid  = (this._lockedTime + this._hoverTime) / 2;
    var newBW = clientW * this._scale - _LW - _PAD_RIGHT;
    var midX  = _LW + (tMid - this._timeRange.minT) / this._timeRange.range * newBW;
    container.scrollLeft = Math.max(0, midX - clientW / 2);
  }

  // ── Public API — pinch / ctrl+wheel zoom ─────────────────────────────────────

  /**
   * Attach ctrl+wheel (trackpad pinch) and touchscreen two-finger-pinch zoom
   * listeners to the container element.  Call once after construction.
   */
  initPinchZoom() {
    var self      = this;
    var container = this._containerEl;

    function _applyZoom(newScale, anchorT, clientX) {
      newScale = Math.max(1, Math.min(20, newScale));
      self._scale = newScale;
      if (self._scaleSliderEl) self._scaleSliderEl.value = newScale;
      if (self._scaleLabelEl)  self._scaleLabelEl.textContent = newScale.toFixed(1) + '×';

      self._render(self._lastTimelineNodes);

      if (self._timeRange && anchorT !== null) {
        var tr   = self._timeRange;
        var newBW = Math.max(1, container.clientWidth * newScale - _LW - _PAD_RIGHT);
        var newX  = _LW + (anchorT - tr.minT) / tr.range * newBW;
        container.scrollLeft = Math.max(0, newX - clientX);
      }
      if (self._onSettingsChange) self._onSettingsChange();
    }

    function _timeAtClientX(clientX) {
      var tr = self._timeRange;
      if (!tr) return null;
      var absX = clientX + container.scrollLeft;
      var bw   = Math.max(1, container.clientWidth * self._scale - _LW - _PAD_RIGHT);
      return tr.minT + Math.max(0, absX - _LW) / bw * tr.range;
    }

    // ── Trackpad pinch (ctrl+wheel) ──────────────────────────────────────────
    // Deltas are accumulated per animation frame to avoid rebuilding the SVG
    // dozens of times per second during fast pinch gestures.
    var _wheelAccum   = 0;
    var _wheelAnchorT = null;
    var _wheelClientX = null;
    var _wheelRafId   = null;

    container.addEventListener('wheel', function(e) {
      if (!e.ctrlKey) return;
      e.preventDefault();
      _wheelAccum += e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
      if (_wheelRafId === null) {
        var rect = container.getBoundingClientRect();
        _wheelClientX = e.clientX - rect.left;
        _wheelAnchorT = _timeAtClientX(_wheelClientX);
        _wheelRafId = requestAnimationFrame(function() {
          _wheelRafId = null;
          _applyZoom(
            self._scale * Math.exp(-_wheelAccum * _PINCH_SENSITIVITY),
            _wheelAnchorT, _wheelClientX
          );
          _wheelAccum = 0;
        });
      }
    }, { passive: false });

    // ── Touchscreen two-finger pinch ─────────────────────────────────────────
    var _pinch = null;

    function _pinchDist(touches) {
      var dx = touches[0].clientX - touches[1].clientX;
      var dy = touches[0].clientY - touches[1].clientY;
      return Math.sqrt(dx * dx + dy * dy);
    }

    container.addEventListener('touchstart', function(e) {
      if (e.touches.length !== 2) { _pinch = null; return; }
      e.preventDefault();
      var rect = container.getBoundingClientRect();
      var midX = (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left;
      _pinch = { startDist: _pinchDist(e.touches), startScale: self._scale,
                 anchorT: _timeAtClientX(midX), midX: midX };
    }, { passive: false });

    container.addEventListener('touchmove', function(e) {
      if (e.touches.length !== 2 || !_pinch) return;
      e.preventDefault();
      var rect  = container.getBoundingClientRect();
      var midX  = (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left;
      var ratio = _pinchDist(e.touches) / _pinch.startDist;
      _applyZoom(_pinch.startScale * ratio, _pinch.anchorT, midX);
    }, { passive: false });

    container.addEventListener('touchend',    function(e) { if (e.touches.length < 2) _pinch = null; });
    container.addEventListener('touchcancel', function()  { _pinch = null; });
  }

  // ── Internal render ──────────────────────────────────────────────────────────

  _render(cyNodes) {
    var self      = this;
    var container = this._containerEl;
    var provider  = this._provider;

    // ── Categorise nodes ──────────────────────────────────────────────────────
    var timedBars = cyNodes.filter(function(n) {
      return n.data.start != null && n.data.end != null &&
        (n.data.type === 'skeleton_call' || n.data.type === 'external');
    });
    var timedTransfers = cyNodes.filter(function(n) {
      return n.data.type === 'transfer' &&
        n.data.start != null && n.data.end != null;
    });
    var timedMarkers = cyNodes.filter(function(n) {
      return n.data.timestamp != null &&
        (n.data.type === 'allocation' || n.data.type === 'deallocation');
    });
    var timedRegions = cyNodes.filter(function(n) {
      return n.data.type === 'region' &&
        n.data.region_start != null && n.data.region_end != null;
    });

    container.innerHTML = '';
    if (!timedBars.length && !timedMarkers.length && !timedRegions.length) {
      var msg = document.createElement('p');
      msg.style.cssText = 'padding:0.5em;color:#888;font-family:monospace;font-size:0.8em';
      msg.textContent = 'No timed events to display.';
      container.appendChild(msg);
      return;
    }

    // ── Time range ────────────────────────────────────────────────────────────
    var minT = Infinity, maxT = -Infinity;
    timedBars.forEach(function(n) {
      var ivs = n.data.intervals || [[n.data.start, n.data.end]];
      ivs.forEach(function(iv) { minT = Math.min(minT, iv[0]); maxT = Math.max(maxT, iv[1]); });
    });
    timedTransfers.forEach(function(n) {
      var ivs = n.data.intervals || [[n.data.start, n.data.end]];
      ivs.forEach(function(iv) { minT = Math.min(minT, iv[0]); maxT = Math.max(maxT, iv[1]); });
    });
    timedMarkers.forEach(function(n) {
      minT = Math.min(minT, n.data.timestamp); maxT = Math.max(maxT, n.data.timestamp);
    });
    timedRegions.forEach(function(n) {
      var ivs = (n.data.region_intervals && n.data.region_intervals.length)
        ? n.data.region_intervals : [[n.data.region_start, n.data.region_end]];
      ivs.forEach(function(iv) { minT = Math.min(minT, iv[0]); maxT = Math.max(maxT, iv[1]); });
    });
    var range = Math.max(maxT - minT, 1);
    this._timeRange = { minT: minT, maxT: maxT, range: range };

    // ── Backends → one row each ───────────────────────────────────────────────
    var bSet = {};
    timedBars.forEach(function(n) { bSet[n.data.backend || 'CPU'] = true; });
    var backends = Object.keys(bSet).sort();
    var bRow = {};
    backends.forEach(function(b, i) { bRow[b] = i; });

    // ── Layout ────────────────────────────────────────────────────────────────
    var LW = _LW, AH = _AXIS_H, RH = _ROW_H, RP = _ROW_PAD, PAD_RIGHT = _PAD_RIGHT;
    var REGION_RH   = timedRegions.length   > 0 ? AH : 0;
    var TRANSFER_RH = timedTransfers.length > 0 ? RH : 0;
    var EVENT_RH    = timedMarkers.length   > 0 ? RH : 0;
    var SVG_W = Math.max(container.clientWidth || 400, 200) * this._scale;
    var H     = REGION_RH + backends.length * RH + TRANSFER_RH + EVENT_RH + AH;
    var BW    = Math.max(1, SVG_W - LW - PAD_RIGHT);

    // Dark-mode palette
    var dark = _isDark();
    var _rowBgA    = dark ? '#1e1c2e' : '#f0f7ff';
    var _rowBgEven = dark ? '#1a1824' : '#fff';
    var _rowBgOdd  = dark ? '#201e2c' : '#f8f8f8';
    var _rowBgTr   = dark ? '#221e2e' : '#f8f4ff';
    var _rowBgEv   = dark ? '#1e1c28' : '#fff8f5';
    var _labelFill = dark ? '#a09ab8' : '#555';
    var _tickFill  = dark ? '#706880' : '#777';
    var _gridLine  = dark ? '#2a2838' : '#efefef';
    var _sepLine   = dark ? '#3a3848' : '#ccc';
    var _axisLine  = dark ? '#3a3848' : '#bbb';
    var _tickMark  = dark ? '#504860' : '#aaa';

    var selectedId = this._selectedNodeId;

    // ── Helpers ───────────────────────────────────────────────────────────────
    function tx(t) { return LW + (t - minT) / range * BW; }
    function fmtDt(dt) {
      if (dt >= 1e6) return (dt / 1e6).toFixed(2) + 's';
      if (dt >= 1e3) return (dt / 1e3).toFixed(1) + 'ms';
      return dt.toFixed(1) + 'µs';
    }
    function escTxt(s) {
      return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
    function escAttr(s) {
      return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
    }
    // Resolve bar colour from provider (spectrum / backend / pattern modes).
    function _callColor(d) {
      if (!provider) return '#778';
      var mode = provider.settings['node-color-call'];
      if (d.type === 'skeleton_call') {
        if (mode === 'backend') {
          var bc = { CPU: 'yellow', OpenMP: 'red', OpenCL: 'blue', CUDA: 'green' };
          return bc[d.backend] || '#778';
        }
        if (mode === 'pattern') {
          var pc = { Map: 'yellow', Reduce: 'red', MapReduce: 'orange', MapOverlap: 'green' };
          return pc[d.pattern] || '#778';
        }
      }
      return provider.spectrumColor(provider.normFromData('node-color-call', d));
    }

    // ── SVG string buffers ────────────────────────────────────────────────────
    var defs = [];
    var body = [];

    // ── Row backgrounds + labels ──────────────────────────────────────────────
    if (REGION_RH > 0) {
      body.push('<rect x="0" y="0" width="' + SVG_W + '" height="' + REGION_RH + '" fill="' + _rowBgA + '"/>');
      body.push('<text x="' + (LW - 6) + '" y="' + (REGION_RH / 2 + 4) + '" text-anchor="end" fill="' + _labelFill + '">Regions</text>');
    }
    backends.forEach(function(b, i) {
      var y = REGION_RH + i * RH;
      body.push('<rect x="0" y="' + y + '" width="' + SVG_W + '" height="' + RH + '" fill="' + (i % 2 ? _rowBgOdd : _rowBgEven) + '"/>');
      body.push('<text x="' + (LW - 6) + '" y="' + (y + RH / 2 + 4) + '" text-anchor="end" fill="' + _labelFill + '">' + escTxt(b) + '</text>');
    });
    if (TRANSFER_RH > 0) {
      var trRowY = REGION_RH + backends.length * RH;
      body.push('<rect x="0" y="' + trRowY + '" width="' + SVG_W + '" height="' + TRANSFER_RH + '" fill="' + _rowBgTr + '"/>');
      body.push('<text x="' + (LW - 6) + '" y="' + (trRowY + TRANSFER_RH / 2 + 4) + '" text-anchor="end" fill="' + _labelFill + '">Transfers</text>');
    }
    if (EVENT_RH > 0) {
      var evRowY = REGION_RH + backends.length * RH + TRANSFER_RH;
      body.push('<rect x="0" y="' + evRowY + '" width="' + SVG_W + '" height="' + EVENT_RH + '" fill="' + _rowBgEv + '"/>');
      body.push('<text x="' + (LW - 6) + '" y="' + (evRowY + EVENT_RH / 2 + 4) + '" text-anchor="end" fill="' + _labelFill + '">Events</text>');
    }

    // Left separator + axis baseline
    body.push('<line x1="' + LW + '" y1="0" x2="' + LW + '" y2="' + (H - AH) + '" stroke="' + _sepLine + '" stroke-width="1"/>');
    body.push('<line x1="' + LW + '" y1="' + (H - AH) + '" x2="' + SVG_W + '" y2="' + (H - AH) + '" stroke="' + _axisLine + '" stroke-width="1"/>');

    // Time grid + tick labels
    var numTicks = Math.max(2, Math.floor(BW / _TICK_PX));
    for (var ti = 0; ti <= numTicks; ti++) {
      var frac = ti / numTicks;
      var tv = minT + frac * range;
      var xv = tx(tv);
      var anchor = ti === 0 ? 'start' : 'middle';
      body.push('<line x1="' + xv + '" y1="0" x2="' + xv + '" y2="' + (H - AH) + '" stroke="' + _gridLine + '" stroke-width="1"/>');
      body.push('<line x1="' + xv + '" y1="' + (H - AH) + '" x2="' + xv + '" y2="' + (H - AH + 4) + '" stroke="' + _tickMark + '" stroke-width="1"/>');
      body.push('<text x="' + xv + '" y="' + (H - AH + 14) + '" text-anchor="' + anchor + '" fill="' + _tickFill + '">+' + escTxt(fmtDt(tv - minT)) + '</text>');
    }

    // ── Region bars ───────────────────────────────────────────────────────────
    if (REGION_RH > 0) {
      var regionBarPad  = 3;
      var regionBarH    = REGION_RH - regionBarPad * 2;
      var sortedRegions = timedRegions.slice().sort(function(a, b) {
        return (a.data.nesting_level || 0) - (b.data.nesting_level || 0);
      });
      var regionClipIdx = 0;
      sortedRegions.forEach(function(n) {
        var d     = n.data;
        var lvl   = d.nesting_level || 0;
        var depth = d.region_depth !== undefined ? d.region_depth : lvl;
        var lr    = Math.max(20,  225 - depth * 34);
        var lg    = Math.max(185, 240 - depth * 9);
        var fill      = dark ? _ganttRegionFillDark(depth) : 'rgba(' + lr + ',' + lg + ',255,0.55)';
        var barStroke = dark ? 'rgba(200,160,80,0.45)' : 'rgba(70,155,220,0.5)';
        var ivs = (d.region_intervals && d.region_intervals.length)
          ? d.region_intervals : [[d.region_start, d.region_end]];
        var opacityAttr = d.hidden ? ' opacity="0.3"' : '';

        ivs.forEach(function(iv, iidx) {
          var x1 = tx(iv[0]), x2 = tx(iv[1]);
          var bw = Math.max(2, x2 - x1);
          var centerX = x1 + bw / 2;
          var selCls = d.id === selectedId ? ' gantt-selected' : '';
          var tip = escTxt(d.label + '\nRegion [depth ' + lvl + ']\n' + fmtDt(iv[1] - iv[0]));
          body.push('<rect class="gantt-bar' + selCls + '"' + opacityAttr
            + ' data-node-id="' + escAttr(d.id) + '" data-center-x="' + centerX + '" data-interval-idx="' + iidx + '"'
            + ' x="' + x1 + '" y="' + regionBarPad + '" width="' + bw + '" height="' + regionBarH + '" rx="2"'
            + ' fill="' + fill + '" stroke="' + barStroke + '" stroke-width="0.5" style="cursor:pointer">'
            + '<title>' + tip + '</title></rect>');
          if (bw > _MIN_LABEL_W) {
            var clipId = 'gantt-rc-' + (regionClipIdx++);
            defs.push('<clipPath id="' + clipId + '"><rect x="' + (x1 + 2) + '" y="' + regionBarPad
              + '" width="' + Math.max(0, bw - 4) + '" height="' + regionBarH + '"/></clipPath>');
            body.push('<text x="' + (x1 + 4) + '" y="' + (regionBarPad + regionBarH / 2 + 4) + '"'
              + ' fill="' + (dark ? 'rgba(240,200,130,0.90)' : 'rgba(0,70,160,0.80)') + '"' + opacityAttr
              + ' clip-path="url(#' + clipId + ')" style="pointer-events:none">' + escTxt(d.label) + '</text>');
          }
        });
      });
    }

    // ── Backend bars ──────────────────────────────────────────────────────────
    var barIdx = 0;
    timedBars.forEach(function(n) {
      var d       = n.data;
      var backend = d.backend || 'CPU';
      var ri      = bRow[backend];
      var y       = REGION_RH + ri * RH + RP;
      var bh      = RH - RP * 2;
      var fill    = _callColor(d);
      var ivs     = d.intervals || [[d.start, d.end]];
      var total   = ivs.length;
      var opacityAttr = d.hidden ? ' opacity="0.3"' : '';

      ivs.forEach(function(iv, iidx) {
        var x1 = tx(iv[0]), x2 = tx(iv[1]);
        var bw = Math.max(2, x2 - x1);
        var centerX = x1 + bw / 2;
        var clipId  = 'gantt-c-' + (barIdx++);
        var selCls  = d.id === selectedId ? ' gantt-selected' : '';
        var iterLabel = total > 1 ? ' [' + (iidx + 1) + '/' + total + ']' : '';
        var tip = escTxt(d.label + (d.pattern ? ' [' + d.pattern + ']' : '') + iterLabel
          + '\n' + (d.backend || '') + '\n' + fmtDt(iv[1] - iv[0]));
        defs.push('<clipPath id="' + clipId + '"><rect x="' + (x1 + 2) + '" y="' + y
          + '" width="' + Math.max(0, bw - 4) + '" height="' + bh + '"/></clipPath>');
        body.push('<rect class="gantt-bar' + selCls + '"' + opacityAttr
          + ' data-node-id="' + escAttr(d.id) + '" data-center-x="' + centerX + '" data-interval-idx="' + iidx + '"'
          + ' x="' + x1 + '" y="' + y + '" width="' + bw + '" height="' + bh + '" rx="2"'
          + ' fill="' + fill + '" stroke="rgba(0,0,0,0.2)" stroke-width="0.5" style="cursor:pointer">'
          + '<title>' + tip + '</title></rect>');
        if (bw > _MIN_LABEL_W) {
          var iterSuffix = total > 1 ? ' ' + (iidx + 1) : '';
          body.push('<text x="' + (x1 + 4) + '" y="' + (y + bh / 2 + 4) + '"' + opacityAttr
            + ' fill="rgba(255,255,255,0.9)" clip-path="url(#' + clipId + ')" style="pointer-events:none">'
            + escTxt(d.label + iterSuffix) + '</text>');
        }
      });
    });

    // ── Transfer bars ─────────────────────────────────────────────────────────
    if (TRANSFER_RH > 0) {
      var trBarY   = REGION_RH + backends.length * RH + RP;
      var trBarH   = TRANSFER_RH - RP * 2;
      var trBarIdx = 0;
      timedTransfers.forEach(function(n) {
        var d     = n.data;
        var ivs   = d.intervals || [[d.start, d.end]];
        var total = ivs.length;
        var opacityAttr = d.hidden ? ' opacity="0.3"' : '';
        ivs.forEach(function(iv, iidx) {
          var x1 = tx(iv[0]), x2 = tx(iv[1]);
          var bw = Math.max(2, x2 - x1);
          var centerX = x1 + bw / 2;
          var clipId  = 'gantt-tr-' + (trBarIdx++);
          var selCls  = d.id === selectedId ? ' gantt-selected' : '';
          var iterLabel = total > 1 ? ' [' + (iidx + 1) + '/' + total + ']' : '';
          var tip = escTxt(d.label + iterLabel + '\n' + (d.direction || 'transfer') + '\n' + fmtDt(iv[1] - iv[0]));
          defs.push('<clipPath id="' + clipId + '"><rect x="' + (x1 + 2) + '" y="' + trBarY
            + '" width="' + Math.max(0, bw - 4) + '" height="' + trBarH + '"/></clipPath>');
          body.push('<rect class="gantt-bar' + selCls + '"' + opacityAttr
            + ' data-node-id="' + escAttr(d.id) + '" data-center-x="' + centerX + '" data-interval-idx="' + iidx + '"'
            + ' x="' + x1 + '" y="' + trBarY + '" width="' + bw + '" height="' + trBarH + '" rx="2"'
            + ' fill="purple" stroke="rgba(0,0,0,0.2)" stroke-width="0.5" style="cursor:pointer">'
            + '<title>' + tip + '</title></rect>');
          if (bw > _MIN_LABEL_W) {
            var iterSuffix = total > 1 ? ' ' + (iidx + 1) : '';
            body.push('<text x="' + (x1 + 4) + '" y="' + (trBarY + trBarH / 2 + 4) + '"' + opacityAttr
              + ' fill="rgba(255,255,255,0.9)" clip-path="url(#' + clipId + ')" style="pointer-events:none">'
              + escTxt(d.label + iterSuffix) + '</text>');
          }
        });
      });
    }

    // ── Event markers (allocation / deallocation) ─────────────────────────────
    if (EVENT_RH > 0) {
      var evRowY2 = REGION_RH + backends.length * RH + TRANSFER_RH;
      var evCY    = evRowY2 + EVENT_RH / 2;
      var HS = 7;

      function starPoints(cx, cy, outerR, innerR) {
        var pts = [];
        for (var i = 0; i < 10; i++) {
          var angle = (i * Math.PI / 5) - Math.PI / 2;
          var r = i % 2 === 0 ? outerR : innerR;
          pts.push((cx + Math.cos(angle) * r).toFixed(2) + ',' + (cy + Math.sin(angle) * r).toFixed(2));
        }
        return pts.join(' ');
      }
      function diamondPoints(cx, cy, hs) {
        return cx + ',' + (cy - hs) + ' ' + (cx + hs) + ',' + cy
          + ' ' + cx + ',' + (cy + hs) + ' ' + (cx - hs) + ',' + cy;
      }

      timedMarkers.forEach(function(n) {
        var d = n.data;
        var cx = tx(d.timestamp);
        var pts, fill, stroke, strokeW;
        if (d.type === 'allocation') {
          pts = starPoints(cx, evCY, HS, HS * 0.42); fill = 'white'; stroke = '#333'; strokeW = 1;
        } else if (d.type === 'deallocation') {
          pts = starPoints(cx, evCY, HS, HS * 0.42); fill = 'black'; stroke = '#ddd'; strokeW = 1;
        } else {
          pts = diamondPoints(cx, evCY, HS); fill = 'purple'; stroke = 'rgba(0,0,0,0.25)'; strokeW = 0.5;
        }
        var selCls = d.id === selectedId ? ' gantt-selected' : '';
        var opacityAttr = d.hidden ? ' opacity="0.3"' : '';
        var tip = escTxt(d.label + '\n' + d.type + '\n@+' + fmtDt(d.timestamp - minT));
        body.push('<polygon class="gantt-bar' + selCls + '"' + opacityAttr
          + ' data-node-id="' + escAttr(d.id) + '" data-center-x="' + cx + '"'
          + ' points="' + pts + '" fill="' + fill + '" stroke="' + stroke + '" stroke-width="' + strokeW + '"'
          + ' style="cursor:pointer"><title>' + tip + '</title></polygon>');
      });
    }

    // ── End-of-trace marker ───────────────────────────────────────────────────
    var _eotColor = dark ? '#504860' : '#bbb';
    var eotX = LW + BW;
    body.push('<line x1="' + eotX + '" y1="0" x2="' + eotX + '" y2="' + (H - AH)
      + '" stroke="' + _eotColor + '" stroke-width="1" stroke-dasharray="4,3"/>');
    body.push('<text x="' + (eotX + 6) + '" y="' + Math.floor((H - AH) / 2)
      + '" dominant-baseline="middle" fill="' + _eotColor + '" font-style="italic">end of trace</text>');

    // ── Interactive cursor elements ───────────────────────────────────────────
    // data-gantt-role attributes instead of IDs so multiple instances can coexist.
    var _hoverStroke  = dark ? '#504860' : '#ccc';
    var _hoverBgFill  = dark ? '#2a2838' : 'white';
    var _hoverLblFill = dark ? '#888090' : '#aaa';
    var _lockStroke   = dark ? '#8878a8' : '#888';
    var _lockBgFill   = dark ? '#22202a' : 'white';
    var _lockBgStroke = dark ? '#604878' : '#999';
    var _lockLblFill  = dark ? '#c0b8d8' : '#555';
    var _deltaFill    = dark ? '#8878a8' : '#888';
    var _deltaBgFill  = dark ? '#22202a' : 'white';
    var _deltaBgStr   = dark ? '#3a3848' : '#bbb';
    var _deltaLblFill = dark ? '#c0b8d8' : '#555';
    var DELTA_Y = H - AH + 22;

    body.push(
      '<g data-gantt-role="hover-g" style="display:none;pointer-events:none">'
      + '<line data-gantt-role="hover-line" y1="0" y2="' + (H - AH) + '" stroke="' + _hoverStroke + '" stroke-width="1"/>'
      + '<rect data-gantt-role="hover-bg" y="' + (H - 16) + '" height="14" rx="2"'
      +   ' fill="' + _hoverBgFill + '" stroke="' + _hoverStroke + '" stroke-width="0.5" opacity="0.92"/>'
      + '<text data-gantt-role="hover-lbl" y="' + (H - 4) + '" text-anchor="middle" fill="' + _hoverLblFill + '"></text>'
      + '</g>',

      '<g data-gantt-role="locked-g" style="display:none;pointer-events:none">'
      + '<line data-gantt-role="locked-line" y1="0" y2="' + (H - AH) + '" stroke="' + _lockStroke + '" stroke-width="1.5"/>'
      + '<rect data-gantt-role="locked-bg" y="' + (H - 16) + '" height="14" rx="2"'
      +   ' fill="' + _lockBgFill + '" stroke="' + _lockBgStroke + '" stroke-width="0.8" opacity="0.95"/>'
      + '<text data-gantt-role="locked-lbl" y="' + (H - 4) + '" text-anchor="middle" fill="' + _lockLblFill + '"></text>'
      + '</g>',

      '<g data-gantt-role="delta-g" style="display:none;pointer-events:none">'
      + '<line data-gantt-role="delta-shaft" y1="' + DELTA_Y + '" y2="' + DELTA_Y + '" stroke="' + _deltaFill + '" stroke-width="1"/>'
      + '<polygon data-gantt-role="delta-arrl" fill="' + _deltaFill + '"/>'
      + '<polygon data-gantt-role="delta-arrr" fill="' + _deltaFill + '"/>'
      + '<rect data-gantt-role="delta-bg" height="14" rx="2"'
      +   ' fill="' + _deltaBgFill + '" stroke="' + _deltaBgStr + '" stroke-width="0.6" opacity="0.93"/>'
      + '<text data-gantt-role="delta-lbl" text-anchor="middle" dominant-baseline="middle"'
      +   ' fill="' + _deltaLblFill + '" font-size="10"></text>'
      + '</g>'
    );

    // ── Inject SVG ────────────────────────────────────────────────────────────
    container.innerHTML =
      '<svg width="' + SVG_W + '" height="' + H
      + '" style="display:block;font-family:monospace;font-size:11px;cursor:default">'
      + '<defs>' + defs.join('') + '</defs>'
      + body.join('')
      + '</svg>';

    // Live references to interactive elements via data-gantt-role (instance-scoped).
    var svg        = container.querySelector('svg');
    var hoverG     = container.querySelector('[data-gantt-role="hover-g"]');
    var hoverLine  = container.querySelector('[data-gantt-role="hover-line"]');
    var hoverBg    = container.querySelector('[data-gantt-role="hover-bg"]');
    var hoverLbl   = container.querySelector('[data-gantt-role="hover-lbl"]');
    var lockedG    = container.querySelector('[data-gantt-role="locked-g"]');
    var lockedLine = container.querySelector('[data-gantt-role="locked-line"]');
    var lockedBg   = container.querySelector('[data-gantt-role="locked-bg"]');
    var lockedLbl  = container.querySelector('[data-gantt-role="locked-lbl"]');
    var deltaG     = container.querySelector('[data-gantt-role="delta-g"]');
    var deltaShaft = container.querySelector('[data-gantt-role="delta-shaft"]');
    var deltaArrL  = container.querySelector('[data-gantt-role="delta-arrl"]');
    var deltaArrR  = container.querySelector('[data-gantt-role="delta-arrr"]');
    var deltaBg    = container.querySelector('[data-gantt-role="delta-bg"]');
    var deltaLbl   = container.querySelector('[data-gantt-role="delta-lbl"]');

    // ── Cursor-line helpers ───────────────────────────────────────────────────
    function xToTimeLabel(x) { return '+' + fmtDt((x - LW) / BW * range); }

    function svgXFromEvent(ev) {
      return ev.clientX - svg.getBoundingClientRect().left;
    }

    function placeCursorLine(group, line, bg, lbl, x) {
      var text = xToTimeLabel(x);
      lbl.textContent = text;
      lbl.setAttribute('x', x);
      var tw = 0;
      try { tw = lbl.getBBox().width; } catch (e) { tw = text.length * 6.5; }
      var pad = 4;
      var bgX = Math.max(LW, Math.min(LW + BW + PAD_RIGHT - tw - pad * 2, x - tw / 2 - pad));
      lbl.setAttribute('x', bgX + pad + tw / 2);
      bg.setAttribute('x', bgX);
      bg.setAttribute('width', tw + pad * 2);
      line.setAttribute('x1', x);
      line.setAttribute('x2', x);
      group.style.display = '';
    }

    function updateDelta(hoverX) {
      if (lockedG.style.display === 'none') { deltaG.style.display = 'none'; return; }
      var lx   = parseFloat(lockedLine.getAttribute('x1'));
      var sign = hoverX >= lx ? '+' : '-';
      var dt   = Math.abs((hoverX - lx) / BW * range);
      var text = sign + fmtDt(dt);
      var midX = (lx + hoverX) / 2;
      deltaLbl.textContent = text;
      deltaLbl.setAttribute('x', midX);
      deltaLbl.setAttribute('y', DELTA_Y);
      var tw = 0;
      try { tw = deltaLbl.getBBox().width; } catch (e) { tw = text.length * 6; }
      var pad = 4;
      deltaBg.setAttribute('x', midX - tw / 2 - pad);
      deltaBg.setAttribute('y', DELTA_Y - 7);
      deltaBg.setAttribute('width', tw + pad * 2);
      var minX = Math.min(lx, hoverX), maxX = Math.max(lx, hoverX), AHS = 5;
      if (maxX - minX > AHS * 3) {
        deltaShaft.setAttribute('x1', minX + AHS); deltaShaft.setAttribute('x2', maxX - AHS);
        deltaArrL.setAttribute('points', minX + ',' + DELTA_Y + ' ' + (minX+AHS) + ',' + (DELTA_Y-AHS*0.5) + ' ' + (minX+AHS) + ',' + (DELTA_Y+AHS*0.5));
        deltaArrR.setAttribute('points', maxX + ',' + DELTA_Y + ' ' + (maxX-AHS) + ',' + (DELTA_Y-AHS*0.5) + ' ' + (maxX-AHS) + ',' + (DELTA_Y+AHS*0.5));
      } else {
        deltaShaft.setAttribute('x1', minX); deltaShaft.setAttribute('x2', maxX);
        deltaArrL.setAttribute('points', ''); deltaArrR.setAttribute('points', '');
      }
      deltaG.style.display = '';
    }

    // ── SVG event listeners ───────────────────────────────────────────────────
    var onSimBarClick = this._onSimBarClick;
    var providerRef   = this._provider;

    // Delegated click: bar → emit 'select'; empty area → set / update locked cursor.
    svg.addEventListener('click', function(ev) {
      var barEl = ev.target.closest ? ev.target.closest('[data-node-id]') : null;
      if (barEl && barEl.dataset.nodeId) {
        var nodeId  = barEl.dataset.nodeId;
        var iterIdx = parseInt(barEl.dataset.intervalIdx || 0);
        var handled = typeof onSimBarClick === 'function' && onSimBarClick(nodeId, iterIdx);
        if (handled) return;
        if (providerRef) providerRef._emit('select', { nodeId: nodeId, source: 'gantt', iterIdx: iterIdx });
      } else {
        var x = svgXFromEvent(ev);
        if (x < LW || x > LW + BW) return;
        self._lockedTime = minT + (x - LW) / BW * range;
        placeCursorLine(lockedG, lockedLine, lockedBg, lockedLbl, x);
      }
    });

    // Mousemove: batch cursor updates into one rAF per frame so getBBox() never
    // blocks more than once per paint cycle.
    var _rafPending = false;
    var _pendingMX  = 0;
    svg.addEventListener('mousemove', function(ev) {
      var x = svgXFromEvent(ev);
      if (x < LW || x > LW + BW) {
        hoverG.style.display = 'none'; deltaG.style.display = 'none';
        self._hoverTime = null; _rafPending = false; return;
      }
      self._hoverTime = minT + (x - LW) / BW * range;
      _pendingMX = x;
      if (_rafPending) return;
      _rafPending = true;
      requestAnimationFrame(function() {
        _rafPending = false;
        placeCursorLine(hoverG, hoverLine, hoverBg, hoverLbl, _pendingMX);
        updateDelta(_pendingMX);
      });
    });

    svg.addEventListener('mouseleave', function() {
      hoverG.style.display = 'none'; deltaG.style.display = 'none';
      self._hoverTime = null; _rafPending = false;
    });

    // Restore locked cursor if one was set before this re-render.
    if (this._lockedTime !== null) {
      var lx = tx(this._lockedTime);
      if (lx >= LW && lx <= LW + BW)
        placeCursorLine(lockedG, lockedLine, lockedBg, lockedLbl, lx);
    }

    // ── Size the pane ─────────────────────────────────────────────────────────
    if (this._paneEl) this._paneEl.style.flex = '0 0 ' + H + 'px';

    // ── Update gantt bar count display ────────────────────────────────────────
    if (this._countEl) {
      var ganttBarCount = 0;
      timedBars.forEach(function(n) {
        ganttBarCount += (n.data.intervals || [[n.data.start, n.data.end]]).length;
      });
      timedTransfers.forEach(function(n) {
        ganttBarCount += (n.data.intervals || [[n.data.start, n.data.end]]).length;
      });
      timedMarkers.forEach(function() { ganttBarCount += 1; });
      timedRegions.forEach(function(n) {
        ganttBarCount += (n.data.region_intervals && n.data.region_intervals.length)
          ? n.data.region_intervals.length : 1;
      });
      this._countEl.textContent = ganttBarCount;
    }
  }
}
