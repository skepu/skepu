// InfoView.js — Info pane: fetches per-node detail data and renders it.
//
// Extracted from the former fetchAndOpenInfo / _renderDurationPlot helpers.
// All data access goes through TraceDataProvider; cross-view side effects
// (graph opacity, source highlighting, Gantt scrolling) are delivered via
// constructor callbacks so InfoView has no imports from the other view modules.
//
// ── Usage ─────────────────────────────────────────────────────────────────────
//
//   const info = new InfoView({
//       provider:          myProvider,          // TraceDataProvider instance
//       onOpaqueNodes:     ids  => …,           // update graph node opacity
//       onHighlightNode:   data => …,           // highlight source/trace in active tab
//       onSelectIteration: (nodeId, iterIdx) => …, // duration-plot bar click
//       onOpen:            ()   => …,           // show / resize the info pane
//       onError:           err  => …,           // surface fetch errors
//   });
//
//   info.open(nodeId, 'graph');          // called when a graph node is clicked
//   info.open(nodeId, 'gantt');          // called from the Gantt chart
//   info.open(nodeId, 'code-listing');   // called from source-gutter badges

'use strict';

import { renderHeatmap } from './heatmap.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function _fmtDur(us) {
  if (us == null || us < 0) return null;
  if (us >= 1e6)  return (us / 1e6).toFixed(1)  + ' s';
  if (us >= 1000) return (us / 1000).toFixed(1) + ' ms';
  return us.toFixed(0) + ' μs';
}

function _durationStats(vals) {
  var n = vals.length;
  var min = Infinity, max = -Infinity, sum = 0;
  for (var i = 0; i < n; i++) {
    if (vals[i] < min) min = vals[i];
    if (vals[i] > max) max = vals[i];
    sum += vals[i];
  }
  var mean   = sum / n;
  var sorted = vals.slice().sort(function(a, b) { return a - b; });
  var median = (n % 2 === 0)
    ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    : sorted[Math.floor(n / 2)];
  var variance = 0;
  for (var i = 0; i < n; i++) variance += (vals[i] - mean) * (vals[i] - mean);
  return { n: n, min: min, max: max, mean: mean, median: median,
           stddev: Math.sqrt(variance / n) };
}

// ── Constants ─────────────────────────────────────────────────────────────────

const NODE_HEADINGS = {
  'skeleton_call'   : 'Skeleton Call',
  'allocation'      : 'Container Allocation',
  'deallocation'    : 'Container Deallocation',
  'transfer'        : 'Data Transfer',
  'container_update': 'Container Write',
  'region'          : 'Instrumented Region',
  'fusion'          : 'Suggested Skeleton Call Fusion',
  'external'        : 'External Data Access',
};

// Keys excluded from the info key/value table.
const HIDDEN_KEYS = new Set([
  'type', 'durations', 'trace_index',
  'content_snapshot', 'content_snapshots',
  'content_snapshot_labels', 'content_snapshot_virtual_label',
]);

// ── InfoView ──────────────────────────────────────────────────────────────────

export class InfoView {

  // ── Constructor ─────────────────────────────────────────────────────────────

  /**
   * @param {object}          options
   * @param {TraceDataProvider} options.provider         Required.
   * @param {Function}        [options.onOpaqueNodes]    ids → void
   * @param {Function}        [options.onHighlightNode]  data → void
   *                          Called (except for 'code-listing' source) with the
   *                          node's server-response data object so the receiver
   *                          can decide whether to highlight source or trace
   *                          based on its own active-tab state.
   *                          SourceCodeView.highlightNode() is the canonical handler.
   * @param {Function}        [options.onSelectIteration] (nodeId, iterIdx) → void
   * @param {Function}        [options.onOpen]            () → void
   * @param {Function}        [options.onError]           err → void
   */
  constructor(options) {
    options = options || {};
    if (!options.provider)
      throw new Error('InfoView: options.provider is required');

    this._provider = options.provider;

    this._onOpaqueNodes     = options.onOpaqueNodes     || null;
    this._onHighlightNode   = options.onHighlightNode   || null;
    this._onSelectIteration = options.onSelectIteration || null;
    this._onOpen            = options.onOpen            || null;
    this._onError           = options.onError           || null;
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  /**
   * Fetch detail data for `nodeId` and populate the info pane.
   *
   * @param {string}  nodeId    Graph node UUID.
   * @param {string}  [source]  Originating view: 'graph' | 'gantt' | 'code-listing'
   * @param {number}  [traceIdx] Per-iteration trace_index override (simulation mode).
   */
  open(nodeId, source, traceIdx) {
    var self = this;
    this._provider.fetchNodeData(nodeId)
      .then(function(data) { self._render(data, nodeId, source, traceIdx); })
      .catch(function(err) {
        console.error('[InfoView] fetch error:', err);
        if (self._onError) self._onError(err);
      });
  }

  /** Clear the info pane — called on background click / deselection. */
  close() {
    var headingEl = document.getElementById('info-heading');
    var bodyEl    = document.getElementById('info');
    var heatmapEl = document.getElementById('heatmap-container');
    var plotEl    = document.getElementById('plot');
    if (headingEl) headingEl.textContent = '';
    if (bodyEl)    bodyEl.innerHTML      = '';
    if (heatmapEl) heatmapEl.innerHTML   = '';
    if (plotEl)    plotEl.innerHTML      = '';
  }

  // ── Internal render ──────────────────────────────────────────────────────────

  _render(data, nodeId, source, traceIdx) {
    // ── Heading ───────────────────────────────────────────────────────────────
    var headingEl = document.getElementById('info-heading');
    if (headingEl) headingEl.textContent = NODE_HEADINGS[data.type] || data.type || '';

    // ── Key/value table ───────────────────────────────────────────────────────
    var bodyEl = document.getElementById('info');
    if (bodyEl) {
      var html = '';
      for (var key in data) {
        if (!data.hasOwnProperty(key)) continue;
        if (HIDDEN_KEYS.has(key) || /internal/.test(key)) continue;
        var value = data[key];
        if (key === 'file')     value = String(value).split('\\').pop().split('/').pop();
        if (key === 'Duration') value = _fmtDur(parseInt(value)) || value;
        html += '<div class="info-row"><span class="info-key">' + key +
                '</span><span class="info-value">' + value + '</span></div>';
      }
      // Use jQuery when available — the existing DOM uses it for .html().
      if (typeof $ === 'function') { $(bodyEl).html(html); }
      else                         { bodyEl.innerHTML = html; }
    }

    // ── Heatmap ───────────────────────────────────────────────────────────────
    var heatmapEl = document.getElementById('heatmap-container');
    if (heatmapEl) {
      renderHeatmap(
        heatmapEl,
        data.content_snapshots               || null,
        data.content_snapshot_labels         || null,
        data.content_snapshot_virtual_label  || null
      );
    }

    // ── Open the pane ─────────────────────────────────────────────────────────
    if (this._onOpen) this._onOpen();

    // ── Source / trace highlighting ───────────────────────────────────────────
    // Delegate entirely to the receiver (SourceCodeView.highlightNode) which
    // knows which of its own tabs is active.  Pass a copy of data so the
    // traceIdx override does not mutate the cached object.
    if (source !== 'code-listing' && this._onHighlightNode) {
      var nodeData = Object.assign({}, data);
      if (traceIdx != null) nodeData.trace_index = traceIdx;
      this._onHighlightNode(nodeData);
    }

    // ── Opaque-node dimming ───────────────────────────────────────────────────
    // For container_update nodes, dim everything except the live-set members.
    if (this._onOpaqueNodes) {
      var liveSet = (data.type === 'container_update' && data.internal)
        ? (data.internal.is_live || [])
        : [];
      this._onOpaqueNodes(liveSet);
    }

    // ── Duration plot ─────────────────────────────────────────────────────────
    this._renderDurationPlot(
      document.getElementById('plot'),
      data.durations,
      nodeId
    );
  }

  // ── Duration plot ────────────────────────────────────────────────────────────
  // SVG bar chart of per-iteration durations, coloured by the provider's current
  // spectrum using the trace-wide min/max range for consistency with the graph.

  _renderDurationPlot(container, durations, nodeId) {
    if (!container) return;
    container.innerHTML = '';

    if (!durations || durations.length < 2) {
      container.style.display = 'none';
      return;
    }
    container.style.display = 'block';

    var vals  = durations.map(function(d) { return d.y; });
    var st    = _durationStats(vals);
    var n     = vals.length;
    var dark  = document.documentElement.classList.contains('dark-mode');

    // ── Stats grid ────────────────────────────────────────────────────────────
    function _statCell(k, v) {
      return '<div class="stat-cell"><span class="stat-key">' + k +
             '</span><span class="stat-val">' + v + '</span></div>';
    }
    var statsHtml =
      '<div class="stat-grid">' +
      _statCell('Min',     _fmtDur(st.min))    +
      _statCell('Max',     _fmtDur(st.max))    +
      _statCell('Iters',   n)                  +
      _statCell('Mean',    _fmtDur(st.mean))   +
      _statCell('Median',  _fmtDur(st.median)) +
      _statCell('Std Dev', _fmtDur(st.stddev)) +
      '</div>';

    // ── SVG bar chart ─────────────────────────────────────────────────────────
    // Three-part layout (fixed label SVG + stretchy bar SVG + plain HTML axis)
    // so text is never distorted by horizontal scaling.
    var LABEL_W  = 40;
    var VW       = 260;
    var R_PAD    = 10;
    var BAR_AREA = VW - R_PAD;
    var BAR_MIN  = 8;
    var ROW_H    = 10;
    var BAR_H    = 7;
    var svgH     = n * ROW_H + 1;
    var localRange = st.max - st.min || 1;

    // Use the trace-wide duration range (same scale as graph/Gantt node colours)
    // so the spectrum is consistent across all views.
    var mm         = this._provider.lastMinmax && this._provider.lastMinmax['duration-total'];
    var globalMin   = mm ? mm.min : st.min;
    var globalRange = mm ? (mm.max - mm.min) || 1 : localRange;

    var textColor = dark ? '#9a9a9a' : '#888888';
    var axisColor = dark ? '#444444' : '#cccccc';

    // Label SVG (fixed width, no distortion)
    var lblLines = [
      '<svg xmlns="http://www.w3.org/2000/svg"',
      ' width="' + LABEL_W + '" height="' + svgH + '"',
      ' style="flex-shrink:0;display:block;font-family:sans-serif;font-size:8px">',
    ];
    for (var i = 0; i < n; i++) {
      var yTop = i * ROW_H + 1;
      lblLines.push(
        '<text x="' + (LABEL_W - 3) + '" y="' + (yTop + BAR_H - 1) + '"' +
        ' text-anchor="end" fill="' + textColor + '">' + (i + 1) + '</text>'
      );
    }
    lblLines.push('</svg>');

    // Bar SVG (stretchy, preserveAspectRatio=none)
    var provider = this._provider;
    var barLines = [
      '<svg xmlns="http://www.w3.org/2000/svg"',
      ' width="100%" height="' + svgH + '"',
      ' viewBox="0 0 ' + VW + ' ' + svgH + '"',
      ' preserveAspectRatio="none"',
      ' style="flex:1;min-width:0;display:block">',
    ];
    for (var i = 0; i < n; i++) {
      var yTop      = i * ROW_H + 1;
      var localNorm = (vals[i] - st.min) / localRange;
      var bw        = BAR_MIN + localNorm * (BAR_AREA - BAR_MIN);
      var colorNorm = Math.max(0, Math.min(1, (vals[i] - globalMin) / globalRange));
      var fill      = provider.spectrumColor(colorNorm);
      barLines.push(
        '<rect x="0" y="' + yTop + '" width="' + bw.toFixed(1) + '"' +
        ' height="' + BAR_H + '" fill="' + fill + '" rx="1"' +
        ' data-node-id="' + nodeId + '" data-iter-idx="' + i + '"' +
        ' style="cursor:pointer">' +
        '<title>Iteration ' + (i + 1) + ': ' + _fmtDur(vals[i]) + '</title>' +
        '</rect>'
      );
    }
    barLines.push(
      '<line x1="0" y1="' + (n * ROW_H) + '" x2="' + (VW - R_PAD) + '" y2="' + (n * ROW_H) + '"' +
      ' stroke="' + axisColor + '" stroke-width="1"/>'
    );
    barLines.push('</svg>');

    // X-axis labels (plain HTML, no SVG distortion)
    var axisRow =
      '<div style="display:flex;justify-content:space-between;' +
      'padding-left:' + LABEL_W + 'px;' +
      'font-family:sans-serif;font-size:0.75em;color:' + textColor + '">' +
      '<span>' + _fmtDur(st.min) + '</span>' +
      '<span>' + _fmtDur(st.max) + '</span>' +
      '</div>';

    // Delegated click handler — fires _onSelectIteration callback.
    var onSelectIteration = this._onSelectIteration;
    container.onclick = function(ev) {
      var bar = ev.target.closest('rect[data-iter-idx]');
      if (!bar) return;
      if (onSelectIteration)
        onSelectIteration(bar.dataset.nodeId, parseInt(bar.dataset.iterIdx, 10));
    };

    container.innerHTML =
      '<div class="plot-heading">Iteration Durations</div>' +
      statsHtml +
      '<div style="display:flex;align-items:flex-start;padding:4px 0 0">' +
      lblLines.join('') + barLines.join('') +
      '</div>' +
      axisRow;
  }
}
