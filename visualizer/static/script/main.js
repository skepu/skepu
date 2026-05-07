

var cy;
var expand_collapse;
var daglayout;

var opaqueNodes = [];
var baseZoom = 1;
var isZooming = false;

// Gantt pane state (referenced from inline script in main.html)
var _lastGraphNodes      = [];
var _ganttPaneOpen       = true;   // overridden by loadSettings if user closed it
var _sourcePaneOpen      = true;   // overridden by loadSettings if user closed it
var _ganttPaneRatio      = 0;      // unused — kept for settings compatibility only
var _ganttScale          = 1;      // x-axis scale multiplier; overridden by loadSettings
var _ganttSelectedNodeId = null;
var _lastMinmax          = null;   // min/max ranges for colour normalisation; set before renderGantt()
var _ganttLockedTime     = null;   // time value of the locked cursor line (survives re-renders)
var _ganttHoverTime      = null;   // time value of the current hover position (null when off-chart)
var _ganttTimeRange      = null;   // {minT, maxT, range} from the last renderGantt() call
var _legendOpen          = true;   // overridden by loadSettings

// Animation durations (ms) — keep consistent across all views.
var ANIM_ZOOM_MS   = 400;   // graph pan/zoom-to-node, expand/collapse
var ANIM_SCROLL_MS = 400;   // Gantt scroll, code-listing scroll

// Gantt layout constants shared between renderGantt() and scroll-preservation logic.
var GANTT_LW        = 72;   // label column width (px)
var GANTT_PAD_RIGHT = 90;   // space reserved at the right for the "end of trace" label

// ── Color spectrums ───────────────────────────────────────────────────────────
// Each entry holds RGB control points sampled from well-known scientific colormaps.
// lerpColor interpolates linearly between stops for a normalised t ∈ [0, 1].
var _colorSpectrum = 'default';

var COLOR_SPECTRUMS = {
  'default':   { stops: [[0,255,0],[255,0,0]] },
  'viridis':   { stops: [[68,1,84],[72,40,120],[62,83,160],[42,120,142],
                          [34,168,132],[122,209,81],[189,223,38],[253,231,37]] },
  'plasma':    { stops: [[13,8,135],[83,2,163],[139,10,165],[184,50,137],
                          [219,92,104],[244,136,73],[254,188,42],[240,249,33]] },
  'inferno':   { stops: [[0,0,4],[40,11,84],[101,21,110],[159,42,99],
                          [212,72,66],[245,125,21],[250,193,39],[252,255,164]] },
  'cividis':   { stops: [[0,34,78],[30,68,122],[66,97,141],[98,124,143],
                          [137,152,133],[177,177,94],[215,202,48],[253,231,37]] },
  'cool-warm': { stops: [[59,76,192],[221,221,221],[180,4,38]] },
  'grayscale': { stops: [[240,240,240],[20,20,20]] },
};

function lerpColor(stops, t) {
  var n = stops.length - 1;
  var i = Math.min(Math.floor(t * n), n - 1);
  var f = t * n - i;
  var a = stops[i], b = stops[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

// Map a normalised value t ∈ [0, 1] to a CSS rgb() string using the active spectrum.
function spectrumColor(t) {
  var stops = (COLOR_SPECTRUMS[_colorSpectrum] || COLOR_SPECTRUMS['default']).stops;
  var c = lerpColor(stops, Math.max(0, Math.min(1, t)));
  return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
}

// Same as spectrumColor but darkened by factor (default 0.6) for use as a border colour.
function spectrumColorDark(t, factor) {
  factor = (factor !== undefined) ? factor : 0.72;
  var stops = (COLOR_SPECTRUMS[_colorSpectrum] || COLOR_SPECTRUMS['default']).stops;
  var c = lerpColor(stops, Math.max(0, Math.min(1, t)));
  return 'rgb(' + Math.round(c[0]*factor) + ',' + Math.round(c[1]*factor) + ',' + Math.round(c[2]*factor) + ')';
}

// Sync checkmarks in the Color Spectrum menu and update the legend gradient bar.
function syncSpectrumMenuIcons() {
  document.querySelectorAll('.spectrum-radio').forEach(function(b) {
    b.querySelector('.menu-icon').textContent = b.dataset.spectrum === _colorSpectrum ? '✓' : '';
  });
  var bar = document.getElementById('color-bar');
  if (bar) {
    var stops = (COLOR_SPECTRUMS[_colorSpectrum] || COLOR_SPECTRUMS['default']).stops;
    bar.style.background = 'linear-gradient(to right,' + stops.map(function(c, i) {
      return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ') '
           + Math.round(i / (stops.length - 1) * 100) + '%';
    }).join(',') + ')';
  }
}

function colorSpectrumChanged(name) {
  _colorSpectrum = name;
  syncSpectrumMenuIcons();
  if (typeof saveSettings === 'function') saveSettings();
  settingChanged();
}

// Auto-render: re-draw automatically on settings change only when the graph is small.
var AUTO_RENDER_THRESHOLD = 200;   // node count below which changes trigger an immediate re-render
var _lastNodeCount        = 0;
var _settingChangedTimer  = null;

function ganttScaleChanged(val) {
  _ganttScale = parseFloat(val);
  var sl = document.getElementById('gantt-scale');
  if (sl) sl.value = _ganttScale;
  var label = document.getElementById('gantt-scale-label');
  if (label) label.textContent = _ganttScale.toFixed(1) + '×';
  saveSettings();
  if (_ganttPaneOpen && _lastGraphNodes.length) renderGantt(_lastGraphNodes);
}


// ── Gantt chart ───────────────────────────────────────────────────────────────

function openGanttPane() {
  _ganttPaneOpen = true;
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  // renderGantt sets the pane height automatically to fit the SVG content.
  if (_lastGraphNodes.length) renderGantt(_lastGraphNodes);
  saveSettings();
}

function closeGanttPane() {
  _ganttPaneOpen = false;
  document.getElementById('gantt-pane').style.flex = '0 0 0';
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  saveSettings();
}

function toggleGanttPane() {
  if (_ganttPaneOpen) closeGanttPane(); else openGanttPane();
}

function openSourcePane() {
  _sourcePaneOpen = true;
  document.getElementById('right_pane').style.display  = '';
  document.getElementById('split-gutter').style.display = '';
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  if (typeof saveSettings === 'function') saveSettings();
}

function closeSourcePane() {
  _sourcePaneOpen = false;
  document.getElementById('right_pane').style.display  = 'none';
  document.getElementById('split-gutter').style.display = 'none';
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  if (typeof saveSettings === 'function') saveSettings();
}

function toggleSourcePane() {
  if (_sourcePaneOpen) closeSourcePane(); else openSourcePane();
}

// Toggle a checkbox setting by its control ID — used by keyboard shortcuts.
// Mirrors menuToggleCheck() without closing the menu.
function toggleSetting(ctrlId) {
  var ctrl = document.getElementById(ctrlId);
  if (!ctrl) return;
  ctrl.checked = !ctrl.checked;
  ctrl.dispatchEvent(new Event('change'));
  var btn = document.querySelector('.menu-check[data-ctrl="' + ctrlId + '"]');
  if (btn) btn.querySelector('.menu-icon').textContent = ctrl.checked ? '✓' : '';
  saveSettings();
  settingChanged();
}

// Central "select a node" action: highlight in all views, open info pane.
// source: 'graph' | 'gantt' | 'code-listing' | omit
// The originating view always gets its highlight updated, but skips its own scroll/zoom.
function selectNode(nodeId, source) {
  if (cy) {
    cy.nodes().removeClass('selected_gutter');
    cy.$id(nodeId).addClass('selected_gutter');
    if (source !== 'graph')
      cy.animate({ fit: { eles: cy.$id(nodeId), padding: cy.height() / 2.3 } }, { duration: ANIM_ZOOM_MS });
  }
  ganttHighlight(nodeId, source !== 'gantt');
  fetchAndOpenInfo(nodeId, source);
}

// Per-badge cycle state: maps badge key (= ids[0]) → current cycle index.
// Reset on each graph render so stale indices don't carry over.
var _badgeCycleIndex = {};

// Called by source gutter badges.  Cycles the focused node on repeated clicks;
// all nodes sharing the badge are highlighted in both the graph and the Gantt.
function badgeClick(ids) {
  var key = ids[0];
  if (!_badgeCycleIndex.hasOwnProperty(key)) _badgeCycleIndex[key] = 0;
  else _badgeCycleIndex[key] = (_badgeCycleIndex[key] + 1) % ids.length;
  var focusId = ids[_badgeCycleIndex[key]];

  // Graph: highlight every node in the badge set, zoom only to the focused one.
  if (cy) {
    cy.nodes().removeClass('selected_gutter');
    ids.forEach(function(id) { cy.$id(id).addClass('selected_gutter'); });
    cy.animate({ fit: { eles: cy.$id(focusId), padding: cy.height() / 2.3 } }, { duration: ANIM_ZOOM_MS });
  }

  // Gantt: highlight every matching bar, scroll to the focused one.
  _ganttSelectedNodeId = focusId;
  var idSet = {};
  ids.forEach(function(id) { idSet[id] = true; });
  var scrollEl = null;
  document.querySelectorAll('.gantt-bar').forEach(function(b) {
    var sel = idSet[b.dataset.nodeId] === true;
    b.classList.toggle('gantt-selected', sel);
    if (b.dataset.nodeId === focusId && !scrollEl) scrollEl = b;
  });
  if (scrollEl) {
    var container = document.getElementById('gantt-svg-container');
    if (container) {
      var cx = scrollEl.dataset.centerX != null
        ? parseFloat(scrollEl.dataset.centerX)
        : parseFloat(scrollEl.getAttribute('x') || 0) + parseFloat(scrollEl.getAttribute('width') || 0) / 2;
      $(container).stop(true).animate({ scrollLeft: Math.max(0, cx - container.clientWidth / 2) }, ANIM_SCROLL_MS);
    }
  }

  fetchAndOpenInfo(focusId, 'code-listing');
}

// Fetch node info from the server and populate the info pane — shared between
// graph node clicks and Gantt bar/marker clicks.
function fetchAndOpenInfo(nodeId, source) {
  var headings = {
    'skeleton_call'    : 'Skeleton Call',
    'allocation'       : 'Container Allocation',
    'deallocation'     : 'Container Deallocation',
    'transfer'         : 'Data Transfer',
    'container_update' : 'Container Write',
    'region'           : 'Instrumented Region',
    'fusion'           : 'Suggested Skeleton Call Fusion',
    'external'         : 'External Data Access',
  };
  fetch('/get_data?id=' + encodeURIComponent(nodeId))
  .then(function(r) { return r.json(); })
  .then(function(data) {
    document.getElementById('info-heading').textContent = headings[data['type']] || data['type'];
    var info = '';
    for (var key in data) {
      var str = key.toString();
      if (data.hasOwnProperty(key) && !str.match('internal') && !str.match('type') && str !== 'durations') {
        var value = data[key];
        if (key === 'file') value = value.split('\\').pop().split('/').pop();
        info += '<p><strong>' + key + '</strong> ' + value + '</p>';
      }
    }
    $('#info').html(info);
    openInfoPane();
    if (source !== 'code-listing') highlightSourceLine(data['Line'], data['File']);

    if (data['type'] === 'container_update') opaqueNodes = data['internal']['is_live'];
    else opaqueNodes = [];
    if (cy) cy.style().update();

    var plot_container = document.getElementById('plot');
    plot_container.innerHTML = '';
    if (data.durations && data.durations.length > 1) {
      var items = new vis.DataSet(data.durations);
      var options = {
        style: 'bar', height: '10em',
        barChart: { width: 50, align: 'center' },
        drawPoints: false, legend: false,
      };
      new vis.Graph2d(plot_container, items, options);
      plot_container.style.display = 'block';
    } else {
      plot_container.style.display = 'none';
    }
  })
  .catch(function(e) { console.error('Error fetching node data:', e); });
}

function ganttHighlight(nodeId, scroll) {
  _ganttSelectedNodeId = nodeId;
  var firstEl = null;
  document.querySelectorAll('.gantt-bar').forEach(function(b) {
    var sel = b.dataset.nodeId === nodeId;
    b.classList.toggle('gantt-selected', sel);
    if (sel && !firstEl) firstEl = b;
  });
  // Animate the gantt container to centre the first matching element.
  // dataset.centerX is set on all bar/marker elements; fall back to rect x+width/2.
  if (scroll !== false && firstEl) {
    var container = document.getElementById('gantt-svg-container');
    if (container) {
      var cx = firstEl.dataset.centerX != null
        ? parseFloat(firstEl.dataset.centerX)
        : parseFloat(firstEl.getAttribute('x') || 0) + parseFloat(firstEl.getAttribute('width') || 0) / 2;
      var target = cx - container.clientWidth / 2;
      $(container).stop(true).animate({ scrollLeft: Math.max(0, target) }, ANIM_SCROLL_MS);
    }
  }
}

function renderGantt(cyNodes) {
  var container = document.getElementById('gantt-svg-container');
  if (!container) return;
  _lastGraphNodes = cyNodes;

  // Four categories of Gantt data
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

  // Time range across all timed elements
  var minT = Infinity, maxT = -Infinity;
  timedBars.forEach(function(n) {
    var ivs = n.data.intervals || [[n.data.start, n.data.end]];
    ivs.forEach(function(iv) { minT = Math.min(minT, iv[0]); maxT = Math.max(maxT, iv[1]); });
  });
  timedTransfers.forEach(function(n) {
    minT = Math.min(minT, n.data.start); maxT = Math.max(maxT, n.data.end);
  });
  timedMarkers.forEach(function(n) {
    minT = Math.min(minT, n.data.timestamp); maxT = Math.max(maxT, n.data.timestamp);
  });
  timedRegions.forEach(function(n) {
    minT = Math.min(minT, n.data.region_start); maxT = Math.max(maxT, n.data.region_end);
  });
  var range = Math.max(maxT - minT, 1);
  _ganttTimeRange = { minT: minT, maxT: maxT, range: range };

  // Distinct backends → one row each
  var bSet = {};
  timedBars.forEach(function(n) { bSet[n.data.backend || 'CPU'] = true; });
  var backends = Object.keys(bSet).sort();
  var bRow = {};
  backends.forEach(function(b, i) { bRow[b] = i; });

  // Layout constants.
  // SVG_W is the total SVG width: exactly container width at scale=1 (no overflow),
  // and container × scale at higher values (horizontal scroll kicks in).
  // PAD_RIGHT is carved out of SVG_W for the "end of trace" label so BW never overflows.
  var LW = GANTT_LW, AH = 36, RH = 28, RP = 3, PAD_RIGHT = GANTT_PAD_RIGHT;
  var REGION_RH   = timedRegions.length   > 0 ? 36 : 0;  // Regions row height (0 if none)
  var TRANSFER_RH = timedTransfers.length > 0 ? 28 : 0;  // Transfers row height (0 if none)
  var EVENT_RH    = timedMarkers.length   > 0 ? 28 : 0;  // Events row height (0 if none)
  var SVG_W = Math.max(container.clientWidth || 400, 200) * _ganttScale;
  var H     = REGION_RH + backends.length * RH + TRANSFER_RH + EVENT_RH + AH;
  var BW    = Math.max(1, SVG_W - LW - PAD_RIGHT);

  var NS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs) {
    var el = document.createElementNS(NS, tag);
    if (attrs) Object.keys(attrs).forEach(function(k) { el.setAttribute(k, attrs[k]); });
    return el;
  }
  function tx(t) { return LW + (t - minT) / range * BW; }
  // Timestamps from the server are in microseconds.
  function fmtDt(dt) {
    if (dt >= 1e6) return (dt / 1e6).toFixed(2) + 's';
    if (dt >= 1e3) return (dt / 1e3).toFixed(1) + 'ms';
    return dt.toFixed(1) + 'µs';
  }

  var svg = svgEl('svg', { width: SVG_W, height: H });
  svg.style.cssText = 'display:block;font-family:monospace;font-size:11px;cursor:default';
  var defs = svgEl('defs');
  svg.appendChild(defs);

  // ── Row backgrounds + labels ──────────────────────────────────────────────────

  // Regions row
  if (REGION_RH > 0) {
    svg.appendChild(svgEl('rect', { x: 0, y: 0, width: SVG_W, height: REGION_RH, fill: '#f0f7ff' }));
    var rt = svgEl('text', { x: LW - 6, y: REGION_RH / 2 + 4, 'text-anchor': 'end', fill: '#555' });
    rt.textContent = 'Regions';
    svg.appendChild(rt);
  }

  // Backend rows
  backends.forEach(function(b, i) {
    var y = REGION_RH + i * RH;
    svg.appendChild(svgEl('rect', { x: 0, y: y, width: SVG_W, height: RH, fill: i % 2 ? '#f8f8f8' : '#fff' }));
    var t = svgEl('text', { x: LW - 6, y: y + RH / 2 + 4, 'text-anchor': 'end', fill: '#555' });
    t.textContent = b;
    svg.appendChild(t);
  });

  // Transfers row
  if (TRANSFER_RH > 0) {
    var trRowY = REGION_RH + backends.length * RH;
    svg.appendChild(svgEl('rect', { x: 0, y: trRowY, width: SVG_W, height: TRANSFER_RH, fill: '#f8f4ff' }));
    var trt = svgEl('text', { x: LW - 6, y: trRowY + TRANSFER_RH / 2 + 4, 'text-anchor': 'end', fill: '#555' });
    trt.textContent = 'Transfers';
    svg.appendChild(trt);
  }

  // Events row
  if (EVENT_RH > 0) {
    var evRowY = REGION_RH + backends.length * RH + TRANSFER_RH;
    svg.appendChild(svgEl('rect', { x: 0, y: evRowY, width: SVG_W, height: EVENT_RH, fill: '#fff8f5' }));
    var et = svgEl('text', { x: LW - 6, y: evRowY + EVENT_RH / 2 + 4, 'text-anchor': 'end', fill: '#555' });
    et.textContent = 'Events';
    svg.appendChild(et);
  }

  // Left separator + axis baseline (baseline extends into the right padding area)
  svg.appendChild(svgEl('line', { x1: LW, y1: 0, x2: LW, y2: H - AH, stroke: '#ccc', 'stroke-width': 1 }));
  svg.appendChild(svgEl('line', { x1: LW, y1: H - AH, x2: SVG_W, y2: H - AH, stroke: '#bbb', 'stroke-width': 1 }));

  // Time grid + tick labels
  var numTicks = Math.max(2, Math.floor(BW / 90));
  for (var ti = 0; ti <= numTicks; ti++) {
    var frac = ti / numTicks;
    var tv = minT + frac * range;
    var xv = tx(tv);
    svg.appendChild(svgEl('line', { x1: xv, y1: 0, x2: xv, y2: H - AH, stroke: '#efefef', 'stroke-width': 1 }));
    svg.appendChild(svgEl('line', { x1: xv, y1: H - AH, x2: xv, y2: H - AH + 4, stroke: '#aaa', 'stroke-width': 1 }));
    var anchor = ti === 0 ? 'start' : 'middle';
    var tl = svgEl('text', { x: xv, y: H - AH + 14, 'text-anchor': anchor, fill: '#777' });
    tl.textContent = '+' + fmtDt(tv - minT);
    svg.appendChild(tl);
  }

  // Resolve the fill colour for a skeleton_call or external node, mirroring
  // the Cytoscape node style (backend / pattern / spectrum mode).
  function ganttCallColor(d) {
    var mode = document.getElementById('node-color-call').value;
    if (mode === 'backend') {
      var bc = { CPU: 'yellow', OpenMP: 'red', OpenCL: 'blue', CUDA: 'green' };
      return bc[d.backend] || '#778';
    }
    if (mode === 'pattern' && d.type === 'skeleton_call') {
      var pc = { Map: 'yellow', Reduce: 'red', MapReduce: 'orange', MapOverlap: 'green' };
      return pc[d.pattern] || '#778';
    }
    return spectrumColor(findPropertyNormFromData('node-color-call', d, _lastMinmax));
  }

  // ── Region bars (sorted ascending by nesting_level → deeper ones drawn on top) ──
  if (REGION_RH > 0) {
    var regionBarPad = 3;
    var regionBarH   = REGION_RH - regionBarPad * 2;
    var sortedRegions = timedRegions.slice().sort(function(a, b) {
      return (a.data.nesting_level || 0) - (b.data.nesting_level || 0);
    });
    sortedRegions.forEach(function(n, rIdx) {
      var d = n.data;
      var x1 = tx(d.region_start), x2 = tx(d.region_end);
      var bw = Math.max(2, x2 - x1);
      var lvl = d.nesting_level || 0;
      // Colour: match Cytoscape region node — rgb(255-depth*7-15, 255, 255) cyan
      var depth = d.region_depth !== undefined ? d.region_depth : lvl;
      var cv = Math.max(60, 255 - depth * 7 - 15);
      var fill = 'rgba(' + cv + ',255,255,0.55)';
      var centerX = x1 + bw / 2;

      var bar = svgEl('rect', {
        x: x1, y: regionBarPad, width: bw, height: regionBarH, rx: 2,
        fill: fill, stroke: 'rgba(100,180,200,0.5)', 'stroke-width': 0.5,
      });
      bar.classList.add('gantt-bar');
      bar.dataset.nodeId = d.id;
      bar.dataset.centerX = centerX;
      bar.style.cursor = 'pointer';
      if (d.id === _ganttSelectedNodeId) bar.classList.add('gantt-selected');

      var title = document.createElementNS(NS, 'title');
      title.textContent = d.label + '\nRegion [depth ' + lvl + ']\n'
        + fmtDt(d.region_end - d.region_start);
      bar.appendChild(title);
      bar.addEventListener('click', function(ev) {
        ev.stopPropagation();
        selectNode(d.id, 'gantt');
      });
      svg.appendChild(bar);

      if (bw > 24) {
        var clipId = 'gantt-rc-' + rIdx;
        var cp = svgEl('clipPath', { id: clipId });
        cp.appendChild(svgEl('rect', { x: x1 + 2, y: regionBarPad,
          width: Math.max(0, bw - 4), height: regionBarH }));
        defs.appendChild(cp);
        var lbl = svgEl('text', {
          x: x1 + 4, y: regionBarPad + regionBarH / 2 + 4,
          fill: 'rgba(0,80,120,0.85)', 'clip-path': 'url(#' + clipId + ')',
        });
        lbl.style.pointerEvents = 'none';
        lbl.textContent = d.label;
        svg.appendChild(lbl);
      }
    });
  }

  // ── Backend bars ──────────────────────────────────────────────────────────────
  var barIdx = 0;
  timedBars.forEach(function(n) {
    var d = n.data;
    var backend = d.backend || 'CPU';
    var ri = bRow[backend];
    var y  = REGION_RH + ri * RH + RP, bh = RH - RP * 2;
    var fill = ganttCallColor(d);
    var ivs = d.intervals || [[d.start, d.end]];
    var total = ivs.length;

    ivs.forEach(function(iv, iidx) {
      var x1 = tx(iv[0]), x2 = tx(iv[1]);
      var bw = Math.max(2, x2 - x1);
      var centerX = x1 + bw / 2;
      var clipId = 'gantt-c-' + (barIdx++);

      var cp = svgEl('clipPath', { id: clipId });
      cp.appendChild(svgEl('rect', { x: x1 + 2, y: y, width: Math.max(0, bw - 4), height: bh }));
      defs.appendChild(cp);

      var bar = svgEl('rect', {
        x: x1, y: y, width: bw, height: bh, rx: 2,
        fill: fill, stroke: 'rgba(0,0,0,0.2)', 'stroke-width': 0.5,
      });
      bar.classList.add('gantt-bar');
      bar.dataset.nodeId = d.id;
      bar.dataset.centerX = centerX;
      bar.style.cursor = 'pointer';
      if (d.id === _ganttSelectedNodeId) bar.classList.add('gantt-selected');

      var title = document.createElementNS(NS, 'title');
      var iterLabel = total > 1 ? ' [' + (iidx + 1) + '/' + total + ']' : '';
      title.textContent = d.label + (d.pattern ? ' [' + d.pattern + ']' : '') + iterLabel
        + '\n' + (d.backend || '') + '\n' + fmtDt(iv[1] - iv[0]);
      bar.appendChild(title);

      bar.addEventListener('click', function(ev) {
        ev.stopPropagation();
        selectNode(d.id, 'gantt');
      });
      svg.appendChild(bar);

      if (bw > 24) {
        var iterSuffix = total > 1 ? ' ' + (iidx + 1) : '';
        var lbl = svgEl('text', {
          x: x1 + 4, y: y + bh / 2 + 4,
          fill: 'rgba(255,255,255,0.9)', 'clip-path': 'url(#' + clipId + ')',
        });
        lbl.style.pointerEvents = 'none';
        lbl.textContent = d.label + iterSuffix;
        svg.appendChild(lbl);
      }
    });
  });

  // ── Transfer bars ─────────────────────────────────────────────────────────────
  if (TRANSFER_RH > 0) {
    var trBarY  = REGION_RH + backends.length * RH + RP;
    var trBarH  = TRANSFER_RH - RP * 2;
    var trBarIdx = 0;
    timedTransfers.forEach(function(n) {
      var d = n.data;
      var ivs = d.intervals || [[d.start, d.end]];
      var total = ivs.length;
      ivs.forEach(function(iv, iidx) {
        var x1 = tx(iv[0]), x2 = tx(iv[1]);
        var bw = Math.max(2, x2 - x1);
        var centerX = x1 + bw / 2;
        var clipId = 'gantt-tr-' + (trBarIdx++);

        var cp = svgEl('clipPath', { id: clipId });
        cp.appendChild(svgEl('rect', { x: x1 + 2, y: trBarY,
          width: Math.max(0, bw - 4), height: trBarH }));
        defs.appendChild(cp);

        var bar = svgEl('rect', {
          x: x1, y: trBarY, width: bw, height: trBarH, rx: 2,
          fill: 'purple', stroke: 'rgba(0,0,0,0.2)', 'stroke-width': 0.5,
        });
        bar.classList.add('gantt-bar');
        bar.dataset.nodeId = d.id;
        bar.dataset.centerX = centerX;
        bar.style.cursor = 'pointer';
        if (d.id === _ganttSelectedNodeId) bar.classList.add('gantt-selected');

        var iterLabel = total > 1 ? ' [' + (iidx + 1) + '/' + total + ']' : '';
        var title = document.createElementNS(NS, 'title');
        title.textContent = d.label + iterLabel
          + '\n' + (d.direction || 'transfer') + '\n' + fmtDt(iv[1] - iv[0]);
        bar.appendChild(title);
        bar.addEventListener('click', function(ev) {
          ev.stopPropagation();
          selectNode(d.id, 'gantt');
        });
        svg.appendChild(bar);

        if (bw > 24) {
          var iterSuffix = total > 1 ? ' ' + (iidx + 1) : '';
          var lbl = svgEl('text', {
            x: x1 + 4, y: trBarY + trBarH / 2 + 4,
            fill: 'rgba(255,255,255,0.9)', 'clip-path': 'url(#' + clipId + ')',
          });
          lbl.style.pointerEvents = 'none';
          lbl.textContent = d.label + iterSuffix;
          svg.appendChild(lbl);
        }
      });
    });
  }

  // ── Event markers ─────────────────────────────────────────────────────────────
  if (EVENT_RH > 0) {
    var evRowY2 = REGION_RH + backends.length * RH + TRANSFER_RH;
    var evCY    = evRowY2 + EVENT_RH / 2;
    var HS = 7; // half-size for diamond / outer radius for star

    // Build SVG polygon points for a 5-pointed star centred at (cx, cy).
    function starPoints(cx, cy, outerR, innerR) {
      var pts = [];
      for (var i = 0; i < 10; i++) {
        var angle = (i * Math.PI / 5) - Math.PI / 2;
        var r = i % 2 === 0 ? outerR : innerR;
        pts.push((cx + Math.cos(angle) * r).toFixed(2)
               + ',' + (cy + Math.sin(angle) * r).toFixed(2));
      }
      return pts.join(' ');
    }

    // Build SVG polygon points for a diamond centred at (cx, cy).
    function diamondPoints(cx, cy, hs) {
      return cx        + ',' + (cy - hs)
        + ' ' + (cx + hs) + ',' + cy
        + ' ' + cx        + ',' + (cy + hs)
        + ' ' + (cx - hs) + ',' + cy;
    }

    timedMarkers.forEach(function(n) {
      var d = n.data;
      var cx = tx(d.timestamp);

      // Shape + fill + stroke per type, matching Cytoscape node styles:
      //   allocation   → white star, black outline  (star shape in graph)
      //   deallocation → black star, black outline  (filled star in graph)
      //   transfer     → purple diamond             (diamond in graph)
      var pts, fill, stroke, strokeW;
      if (d.type === 'allocation') {
        pts    = starPoints(cx, evCY, HS, HS * 0.42);
        fill   = 'white';
        stroke = 'black';
        strokeW = 1;
      } else if (d.type === 'deallocation') {
        pts    = starPoints(cx, evCY, HS, HS * 0.42);
        fill   = 'black';
        stroke = 'black';
        strokeW = 1;
      } else {
        // transfer → purple diamond
        pts    = diamondPoints(cx, evCY, HS);
        fill   = 'purple';
        stroke = 'rgba(0,0,0,0.25)';
        strokeW = 0.5;
      }

      var marker = svgEl('polygon', {
        points: pts, fill: fill,
        stroke: stroke, 'stroke-width': strokeW,
      });
      marker.classList.add('gantt-bar');
      marker.dataset.nodeId = d.id;
      marker.dataset.centerX = cx;
      marker.style.cursor = 'pointer';
      if (d.id === _ganttSelectedNodeId) marker.classList.add('gantt-selected');

      var title = document.createElementNS(NS, 'title');
      title.textContent = d.label + '\n' + d.type + '\n@+' + fmtDt(d.timestamp - minT);
      marker.appendChild(title);
      marker.addEventListener('click', function(ev) {
        ev.stopPropagation();
        selectNode(d.id, 'gantt');
      });
      svg.appendChild(marker);
    });
  }

  // "End of trace" marker — dashed vertical line at the right edge of the bar area,
  // followed by a label that scrolls with the SVG content.
  var eotX = LW + BW; // right edge of the bar area; PAD_RIGHT of space follows before SVG_W
  svg.appendChild(svgEl('line', {
    x1: eotX, y1: 0, x2: eotX, y2: H - AH,
    stroke: '#bbb', 'stroke-width': 1, 'stroke-dasharray': '4,3',
  }));
  var eotText = svgEl('text', {
    x: eotX + 6, y: Math.floor((H - AH) / 2),
    'dominant-baseline': 'middle',
    fill: '#bbb', 'font-style': 'italic',
  });
  eotText.textContent = 'end of trace';
  svg.appendChild(eotText);

  // ── Interactive cursor lines (hover + locked) ────────────────────────────
  // Both groups sit on top of all other SVG content.  pointer-events:none so
  // they do not block tooltips / clicks on bars beneath them.

  // Convert an SVG x-coordinate to a time-offset label string.
  function xToTimeLabel(x) {
    return '+' + fmtDt((x - LW) / BW * range);
  }

  // Position a cursor-line group (line + background rect + text label) at x.
  function placeCursorLine(group, line, bg, lbl, x) {
    var text = xToTimeLabel(x);
    lbl.textContent = text;
    lbl.setAttribute('x', x);             // temporary – we need a layout pass
    // Measure after setting textContent (SVG is already in the DOM at this point).
    var tw = 0;
    try { tw = lbl.getBBox().width; } catch (e) { tw = text.length * 6.5; }
    var pad = 4;
    // Clamp the label so it stays within the SVG.
    var bgX = Math.max(LW, Math.min(LW + BW + PAD_RIGHT - tw - pad * 2,
                                    x - tw / 2 - pad));
    lbl.setAttribute('x', bgX + pad + tw / 2);
    bg.setAttribute('x', bgX);
    bg.setAttribute('width', tw + pad * 2);
    line.setAttribute('x1', x);
    line.setAttribute('x2', x);
    group.style.display = '';
  }

  // Hover line (light gray — follows mouse, disappears on leave)
  var hoverG = svgEl('g', {});
  hoverG.style.cssText = 'display:none;pointer-events:none';
  var hoverLine = svgEl('line', {
    y1: 0, y2: H - AH,
    stroke: '#ccc', 'stroke-width': 1,
  });
  var hoverBg = svgEl('rect', {
    y: H - 16, height: 14, rx: 2,
    fill: 'white', stroke: '#ccc', 'stroke-width': 0.5, opacity: 0.92,
  });
  var hoverLbl = svgEl('text', {
    y: H - 4, 'text-anchor': 'middle', fill: '#aaa',
  });
  hoverG.appendChild(hoverLine);
  hoverG.appendChild(hoverBg);
  hoverG.appendChild(hoverLbl);
  svg.appendChild(hoverG);

  // Locked line (darker gray — placed on click, stays until next click)
  var lockedG = svgEl('g', {});
  lockedG.style.cssText = 'display:none;pointer-events:none';
  var lockedLine = svgEl('line', {
    y1: 0, y2: H - AH,
    stroke: '#888', 'stroke-width': 1.5,
  });
  var lockedBg = svgEl('rect', {
    y: H - 16, height: 14, rx: 2,
    fill: 'white', stroke: '#999', 'stroke-width': 0.8, opacity: 0.95,
  });
  var lockedLbl = svgEl('text', {
    y: H - 4, 'text-anchor': 'middle', fill: '#555',
  });
  lockedG.appendChild(lockedLine);
  lockedG.appendChild(lockedBg);
  lockedG.appendChild(lockedLbl);
  svg.appendChild(lockedG);

  // Helpers to get the SVG x-coordinate from a mouse event, accounting for
  // the parent container's horizontal scroll position.
  function svgXFromEvent(ev) {
    return ev.clientX - svg.getBoundingClientRect().left;
  }

  // ── Delta distance indicator ──────────────────────────────────────────────
  // Horizontal double-ended arrow drawn in the upper portion of the label zone
  // (DELTA_Y sits 9 px below the axis baseline, centred in the upper half of
  // the 36 px AH zone).  Visible only while the hover line is active AND a
  // locked line has been placed.
  var DELTA_Y = H - AH + 22;  // arrow shaft y-coordinate (lower zone, above cursor labels)

  var deltaG = svgEl('g', {});
  deltaG.style.cssText = 'display:none;pointer-events:none';
  var deltaShaft = svgEl('line', {
    y1: DELTA_Y, y2: DELTA_Y, stroke: '#888', 'stroke-width': 1,
  });
  var deltaArrL = svgEl('polygon', { fill: '#888' });
  var deltaArrR = svgEl('polygon', { fill: '#888' });
  var deltaBg   = svgEl('rect', {
    height: 14, rx: 2, fill: 'white', stroke: '#bbb', 'stroke-width': 0.6, opacity: 0.93,
  });
  var deltaLbl  = svgEl('text', {
    'text-anchor': 'middle', 'dominant-baseline': 'middle', fill: '#555', 'font-size': 10,
  });
  deltaG.appendChild(deltaShaft);
  deltaG.appendChild(deltaArrL);
  deltaG.appendChild(deltaArrR);
  deltaG.appendChild(deltaBg);
  deltaG.appendChild(deltaLbl);
  svg.appendChild(deltaG);

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

    var minX = Math.min(lx, hoverX);
    var maxX = Math.max(lx, hoverX);
    var AHS  = 5;   // arrowhead half-length (px)
    if (maxX - minX > AHS * 3) {
      // Full double-ended arrow; shaft runs between the two arrowhead bases.
      deltaShaft.setAttribute('x1', minX + AHS);
      deltaShaft.setAttribute('x2', maxX - AHS);
      deltaArrL.setAttribute('points',
        minX + ',' + DELTA_Y + ' ' +
        (minX + AHS) + ',' + (DELTA_Y - AHS * 0.5) + ' ' +
        (minX + AHS) + ',' + (DELTA_Y + AHS * 0.5));
      deltaArrR.setAttribute('points',
        maxX + ',' + DELTA_Y + ' ' +
        (maxX - AHS) + ',' + (DELTA_Y - AHS * 0.5) + ' ' +
        (maxX - AHS) + ',' + (DELTA_Y + AHS * 0.5));
    } else {
      // Lines too close for arrowheads — just a short segment.
      deltaShaft.setAttribute('x1', minX);
      deltaShaft.setAttribute('x2', maxX);
      deltaArrL.setAttribute('points', '');
      deltaArrR.setAttribute('points', '');
    }
    deltaG.style.display = '';
  }

  svg.addEventListener('mousemove', function(ev) {
    var x = svgXFromEvent(ev);
    if (x < LW || x > LW + BW) { hoverG.style.display = 'none'; deltaG.style.display = 'none'; _ganttHoverTime = null; return; }
    _ganttHoverTime = minT + (x - LW) / BW * range;
    placeCursorLine(hoverG, hoverLine, hoverBg, hoverLbl, x);
    updateDelta(x);
  });

  svg.addEventListener('mouseleave', function() {
    hoverG.style.display = 'none';
    deltaG.style.display = 'none';
    _ganttHoverTime = null;
  });

  // SVG background clicks (bar clicks call stopPropagation, so only empty
  // space triggers this handler).
  svg.addEventListener('click', function(ev) {
    var x = svgXFromEvent(ev);
    if (x < LW || x > LW + BW) return;
    _ganttLockedTime = minT + (x - LW) / BW * range;
    placeCursorLine(lockedG, lockedLine, lockedBg, lockedLbl, x);
  });

  container.appendChild(svg);

  // Restore the locked cursor line if a time was set before this render
  // (e.g. after a zoom-to-interval or window resize).
  if (_ganttLockedTime !== null) {
    var lx = tx(_ganttLockedTime);
    if (lx >= LW && lx <= LW + BW)
      placeCursorLine(lockedG, lockedLine, lockedBg, lockedLbl, lx);
  }

  // Size the pane exactly to the SVG content so no manual resizing is needed.
  var pane = document.getElementById('gantt-pane');
  if (pane) pane.style.flex = '0 0 ' + H + 'px';
}

// Re-render the Gantt chart after a container resize, keeping the currently
// visible time range centred in the viewport.
function reRenderGanttPreservingScroll() {
  var container = document.getElementById('gantt-svg-container');
  if (!container) { renderGantt(_lastGraphNodes); return; }

  var svg = container.querySelector('svg');
  var svgW       = svg ? parseFloat(svg.getAttribute('width'))  || 0 : 0;
  var scrollLeft = container.scrollLeft;
  var clientW    = container.clientWidth;

  // Compute the fraction of the bar area that is currently centred.
  var barW = svgW - GANTT_LW - GANTT_PAD_RIGHT;
  var centerFrac = barW > 0
    ? (scrollLeft + clientW / 2 - GANTT_LW) / barW
    : 0.5;
  centerFrac = Math.max(0, Math.min(1, centerFrac));

  renderGantt(_lastGraphNodes);

  // Restore: the new SVG has a different barW — calculate the target scrollLeft.
  var newSvg  = container.querySelector('svg');
  var newSvgW = newSvg ? parseFloat(newSvg.getAttribute('width')) || 0 : 0;
  var newBarW = newSvgW - GANTT_LW - GANTT_PAD_RIGHT;
  var newScrollLeft = GANTT_LW + centerFrac * newBarW - container.clientWidth / 2;
  container.scrollLeft = Math.max(0, newScrollLeft);
}

// Zoom and pan the Gantt to frame the measured interval (locked line ↔ hover line).
// Triggered by the Z key while the measuring bar is active.
function zoomGanttToMeasuredInterval() {
  if (_ganttLockedTime === null || _ganttHoverTime === null) return;
  if (!_ganttPaneOpen || !_lastGraphNodes.length || !_ganttTimeRange) return;

  var dt = Math.abs(_ganttHoverTime - _ganttLockedTime);
  if (dt < 1) return;   // interval too small to zoom to

  var container = document.getElementById('gantt-svg-container');
  if (!container) return;
  var clientW = container.clientWidth;

  // Choose a scale so the measured interval fills ~90 % of the visible bar area.
  var fracInterval = dt / _ganttTimeRange.range;
  var targetBW     = (clientW - GANTT_LW) / (fracInterval * 0.9);
  var newScale     = Math.max(1, (targetBW + GANTT_LW + GANTT_PAD_RIGHT) / clientW);

  // Update scale state + UI without triggering an extra render.
  _ganttScale = newScale;
  var sl = document.getElementById('gantt-scale');
  if (sl) sl.value = newScale;
  var slLbl = document.getElementById('gantt-scale-label');
  if (slLbl) slLbl.textContent = newScale.toFixed(1) + '×';
  saveSettings();

  renderGantt(_lastGraphNodes);

  // Scroll so the midpoint of the interval is centred in the viewport.
  var tMid  = (_ganttLockedTime + _ganttHoverTime) / 2;
  var newBW = clientW * _ganttScale - GANTT_LW - GANTT_PAD_RIGHT;
  var midX  = GANTT_LW + (tMid - _ganttTimeRange.minT) / _ganttTimeRange.range * newBW;
  container.scrollLeft = Math.max(0, midX - clientW / 2);
}

// ── End Gantt ─────────────────────────────────────────────────────────────────

function fetchViewDataAndRender()
{
  document.getElementById('cy').style.height = "";
  document.getElementById('cy').innerHTML = "";
  let view_mode = "graph"; //document.getElementById("view-mode").value;
  if (view_mode == "graph")
  {
    document.getElementById("cy").style.display = "block";
    document.getElementById('cy').style.height = document.getElementById('cy').offsetHeight + "px";
    var tlEl = document.getElementById("timeline");
    if (tlEl) tlEl.style.display = "none";
    return fetchGraphDataAndRender();
  }
  else if (view_mode == "timeline")
  {
    document.getElementById("cy").style.display = "none";
    var tlEl = document.getElementById("timeline");
    if (tlEl) tlEl.style.display = "block";
    return fetchTimelineDataAndRender();
  }
}

function fetchTimelineDataAndRender()
{
  var container = document.getElementById('timeline');

  fetch('/timeline')
  .then(response => response.json())
  .then(data =>
  {
    // Create a DataSet (allows two way data-binding)
    var items = new vis.DataSet(data.nodes);
    var groups = data.groups;

    // Configuration for the Timeline
    var options = {
      locale: 'en',
      height: '100%'
    };

    // Create a Timeline
    var timeline = new vis.Timeline(container, items,  groups, options);
  });

}

function zoomToFit(node_id)
{
  cy.nodes().removeClass('selected_gutter');
  var item = cy.$id(node_id);
  item.addClass('selected_gutter');
  cy.animate({ fit: { eles: item, padding: 40 } }, { duration: ANIM_ZOOM_MS });
}

function zoomToFitGroup(node_ids)
{
  cy.nodes().removeClass('selected_gutter');
  var items = cy.$(node_ids.map(function(id) { return '#' + id; }).join(','));
  items.addClass('selected_gutter');
  cy.animate({ fit: { eles: items, padding: 40 } }, { duration: ANIM_ZOOM_MS });
}

function renderImage()
{
  var png64 = cy.png({ scale: 4, bg: "white" });
  document.querySelector('#png-render').setAttribute('src', png64);
  document.querySelector('#png-render').style.display = "inline";
}

function fetchGraphDataAndRender()
{
  let show_critical_path = document.getElementById("edge-opacity").value == "critical-path";
  let show_allocations = document.getElementById("container-allocations").checked;
  let show_deallocations = document.getElementById("container-deallocations").checked;
  let show_transfers = document.getElementById("container-transfers").checked;
  let show_scalars = document.getElementById("element-accesses").checked;
  let show_antideps = document.getElementById("anti-deps").checked;
  let data_as_edges = !document.getElementById("container-updates").checked;
  let show_regions = document.getElementById("show_regions").checked;
  let coalesce_region_deps = false;//document.getElementById("coalesce_region_deps").checked;
  let collapse_iteration = document.getElementById("collapse_iteration").checked;
  let fusion_analysis = document.getElementById("fusion_analysis").checked;
  let edge_labels = document.getElementById("edge-labels").checked;
  let edge_style = document.getElementById("edge-style").value;
  let dag_direction = document.getElementById("direction").value == "vertical" ? "TB" : "LR";

  fetch('/graph?container_allocations=' + show_allocations
    + "&container_deallocations=" + show_deallocations
    + "&container_transfers=" + show_transfers
    + "&show_scalars=" + show_scalars
    + "&anti_deps=" + show_antideps
    + "&data_as_edges=" + data_as_edges
    + "&show_regions=" + show_regions
    + "&coalesce_region_deps=" + coalesce_region_deps
    + "&collapse_iteration=" + collapse_iteration
    + "&fusion_analysis=" + fusion_analysis)
  .then(response => response.json())
  .then(data =>
  {
    let nodes = data["nodes"];
    let edges = data["edges"];
    let event_count = data["event_count"];
    let fusion_hints = data["fusion_hints"];

/*    console.log(fusion_hints);

    document.getElementById("optimization-hints").innerHTML = "";
    var opt = document.createElement('option');
    opt.value = "dummy";
    opt.innerHTML = "Select (" + fusion_hints.length + " hints)";
    document.getElementById("optimization-hints").appendChild(opt);
    for (var i = 0; i < fusion_hints.length; ++i)
    {
      var opt = document.createElement('option');
      opt.value = i;
      opt.innerHTML = "Skeleton fusion [" + (i+1) + "]";
      document.getElementById("optimization-hints").appendChild(opt);
    }

    document.getElementById("optimization-hints").onchange = function()
    {
      var value = parseInt(document.getElementById("optimization-hints").value);
      if (value == value)
        zoomToFitGroup(fusion_hints[value]);
    };*/

    document.getElementById("event-count").innerHTML = event_count;
    document.getElementById("node-count").innerHTML = nodes.length;
    document.getElementById("edge-count").innerHTML = edges.length;
    _lastNodeCount  = nodes.length;
    _lastGraphNodes = nodes;
    clearAlert(); // settings have just been applied; any pending alert is now stale

    // Compute min/max ranges for colour normalisation before rendering either view.
    var minmax = {
      "duration-total" : [Infinity, -Infinity],
      "elements" : [Infinity, -Infinity],
      "duration-per-element" : [Infinity, -Infinity],
      "dag-depth" : [Infinity, -Infinity],
      "total-order" : [Infinity, -Infinity],
      "critical-path" : [0, 100],
    };
    nodes.forEach(node =>
    {
      if (!(node.data.type == "skeleton_call" || node.data.type == "external" || node.data.type == "transfer")) return;

      const duration = node.data['duration'];
      const dag_depth = node.data['dag_depth'];
      const total_order = node.data['total_order'];
      minmax["duration-total"][0] = Math.min(minmax["duration-total"][0], duration);
      minmax["duration-total"][1] = Math.max(minmax["duration-total"][1], duration);
      minmax["dag-depth"][0] = Math.min(minmax["dag-depth"][0], dag_depth);
      minmax["dag-depth"][1] = Math.max(minmax["dag-depth"][1], dag_depth);
      minmax["total-order"][0] = Math.min(minmax["total-order"][0], total_order);
      minmax["total-order"][1] = Math.max(minmax["total-order"][1], total_order);

      if (!(node.data.type == "skeleton_call")) return;

      const elements = node.data['elements'].reduce((a, b) => (a * b));
      const duration_per_element = duration / elements;
      minmax["elements"][0] = Math.min(minmax["elements"][0], elements);
      minmax["elements"][1] = Math.max(minmax["elements"][1], elements);
      minmax["duration-per-element"][0] = Math.min(minmax["duration-per-element"][0], duration_per_element);
      minmax["duration-per-element"][1] = Math.max(minmax["duration-per-element"][1], duration_per_element);
    });
    _lastMinmax = minmax;   // expose to renderGantt()

    if (_ganttPaneOpen) renderGantt(nodes);

    daglayout = {
      directed: true,
      name: 'dagre',
      nodeSep: 15,
      rankSep: 20,
      rankDir: dag_direction, // TB or LR
    //  align: 'UR',
    //  ranker: 'longest-path',
      nodeDimensionsIncludeLabels: true
    };

    transfer_mapper = { "host-to-device" : "↑", "device-to-host" : "↓"}

    // Draw the cytoscape graph
    if (cy) cy.destroy();
    cy = cytoscape(
    {
      container: document.getElementById('cy'),
      elements: edges.concat(nodes),

      // How sensitive the zoom wheel is as well a how much you can zoom
      wheelSensitivity: 0.0,
      minZoom: 0.01,
      maxZoom: 5,
      userZoomingEnabled: true,

      // Choose the style of the nodes depending on the type of container
      style:
      [
        {
          selector: 'node',
          style: {
            'text-outline-width': 1,
            'text-outline-color': 'white',
            'font-size': '0.6em',
            'min-zoomed-font-size': '0.4em',
            'opacity' : function(ele) {
              if (opaqueNodes.length == 0 || opaqueNodes.includes(ele.data("id")))
                return 1;
              else return 0.2;
            }
          }
        },
        {
          selector: 'node[type="allocation"]',
          style: {
            'content': 'data(label)',
            'text-valign' : 'center',
            'shape': 'star',
            'border-width' : '1',
            'background-color' : 'white',
            'border-color' : 'black'
          }
        },
        {
          selector: 'node[type="deallocation"]',
          style: {
            'content': 'data(label)',
            'text-valign' : 'center',
            'shape': 'star',
            'border-width' : '1',
            'background-color' : 'black',
            'border-color' : 'black'
          }
        },
        {
          selector: 'node[type="external"]',
          style: {
            'content': 'data(label)',
            'text-valign' : 'center',
            'shape': 'octagon',
            'background-color' : function(ele)
            {
              return spectrumColor(findPropertyNorm("node-color-call", ele, minmax));
            },
            'border-width': 1.5,
            'border-color': function(ele)
            {
              return spectrumColorDark(findPropertyNorm("node-color-call", ele, minmax));
            }
          }
        },
        {
          selector: 'node[type="transfer"]',
          style: {
            'content': function (ele) { return ele.data("label") + transfer_mapper[ele.data("direction")]; },
            'text-valign' : 'center',
            'shape': 'diamond',
            'background-color' : 'purple',
            'border-width': 1,
            'border-color': '#6a006a'
          }
        },
        {
          selector: 'node[type="container_update"]',
          style: {
            'content': function (ele) {
              if (ele.data("iteration_count") > 1) return ele.data("label") + " x" + ele.data("iteration_count");
              return ele.data("label") + "@" + ele.data("version") + "";
            },
            'text-valign' : 'center',
            'shape': 'barrel',
            'background-color' : 'lightgray',
            'border-width': 1,
            'border-color': '#aaa'
          }
        },
        {
          selector: 'node[type="scalar"]',
          style: {
            'content': function (ele) { return ele.data("label") },
            'text-valign' : 'center',
            'shape': 'barrel',
            'background-color' : 'lightgray',
            'border-width': 1,
            'border-color': '#aaa'
          }
        },
        {
          selector: 'node[type="skeleton_call"]',
          style: {
            'content': function (ele) {
              if (ele.data("iteration_count") > 1) return ele.data("label") + " x" + ele.data("iteration_count");
              return ele.data("label");
            },
            'shape': 'ellipse',
            'text-valign' : 'center',
            'background-color': function(ele)
            {
              if (document.getElementById("node-color-call").value == "backend")
              {
                backendColors = {
                  "CPU" : "yellow",
                  "OpenMP" : "red",
                  "OpenCL" : "blue",
                  "CUDA" : "green",
                };
                return backendColors[ele.data("backend")];
              }
              else if (document.getElementById("node-color-call").value == "pattern")
              {
                patternColors = {
                  "Map" : "yellow",
                  "Reduce" : "red",
                  "MapReduce" : "orange",
                  "MapOverlap" : "green",
                };
                return patternColors[ele.data("pattern")];
              }
              return spectrumColor(findPropertyNorm("node-color-call", ele, minmax));
            },
            'border-width': 1.5,
            'border-color': function(ele)
            {
              if (document.getElementById("node-color-call").value == "backend")
              {
                var darkBackend = { "CPU": "#b3b300", "OpenMP": "#b30000", "OpenCL": "#0000b3", "CUDA": "#007a00" };
                return darkBackend[ele.data("backend")] || '#555';
              }
              else if (document.getElementById("node-color-call").value == "pattern")
              {
                var darkPattern = { "Map": "#b3b300", "Reduce": "#b30000", "MapReduce": "#b37a00", "MapOverlap": "#007a00" };
                return darkPattern[ele.data("pattern")] || '#555';
              }
              return spectrumColorDark(findPropertyNorm("node-color-call", ele, minmax));
            },
            'width': function(ele)
            {
              let norm = findPropertyNorm("node-size-call", ele, minmax);
              return 100 * Math.sqrt(norm) + 15;
            },
            'height': function(ele)
            {
              let norm = findPropertyNorm("node-size-call", ele, minmax);
              return 100 * Math.sqrt(norm) + 15;
            }
          }
        },
        {
          selector: 'node[type="region"]',
          style: {
            'content': function (ele) {
              if (ele.data("iteration_count") > 1) return ele.data("label") + " x" + ele.data("iteration_count");
              return ele.data("label");
            },
            'border-color' : '#aaa',
            'border-width': 5,
          //  'border-style' : 'dotted',
            'background-color': function(ele)
            {
              const value = 255 - ele.data('region_depth') * 7 - 15;
              return `rgb(${value},${255},${255})`
            }
          }
        },
        {
          selector: 'node[type="fusion"]',
          style: {
            'content': 'data(label)',
            'background-color': '#88f',
          }
        },
        {
          selector: 'node[type="fusion"].cy-expand-collapse-collapsed-node',
          style: {
            'content': 'data(label)',
            'text-halign' : 'center',
            'text-valign' : 'center',
            'background-color': '#88f',
          }
        },
        {
          selector: 'node[type="reduced_region"]',
          style: {
            'content': 'data(label)',
            'shape': 'round-rectangle',
            'text-valign' : 'center',
            'background-color': function(ele)
            {
              const value = ele.data('region_depth') * 15 + 5;
              return `rgb(${value},${255},${255})`
            }
          }
        },
        {
          selector: '.selected_gutter',
          style: {
            'border-width': 5,
            'border-color': 'blue',
          }
        },
        {
          selector: 'edge',
          style: {
            'content' : function(ele) { return edge_labels ? ele.data('label') : ""; },
            'text-outline-width': 1,
            'text-outline-color': 'black',
            'color' : 'white',
            'font-size': '0.5em',
            'width': function(ele) {
              if (show_critical_path && !ele.data('is_critical_path'))
                return 1;
              else if (show_critical_path && ele.data('is_critical_path'))
                return 5;
              else return 1.5;
            },
            'target-arrow-shape': 'triangle',
            'line-color': function(ele) {
              if (show_critical_path && !ele.data('is_critical_path'))
                return "lightgray";
            //  else if (show_critical_path && ele.data('is_critical_path'))
            //    return "blue";
              else return "#333";
            },
            'target-arrow-color': function(ele) {
              if (show_critical_path && !ele.data('is_critical_path'))
                return "lightgray";
            //  else if (show_critical_path && ele.data('is_critical_path'))
            //    return "blue";
              else return "#333";
            },
            "curve-style": edge_style,
            "taxi-direction": "downward",
            'opacity' : function(ele) {
              if (opaqueNodes.length == 0 || opaqueNodes.includes(ele.data("id")))
                return 1;
              else return 0.2;
            }
          }
        },
        {
          selector: 'edge[type="anti-dep"]',
          style: {
            'line-color': 'red',
            'target-arrow-color': 'red',
          }
        },
        {
          selector: 'edge[access_mode="elwise"]',
          style: {
            'line-style': 'dashed'
          }
        },
        {
          selector: 'edge[access_mode="scalar"]',
          style: {
            'line-style': 'dotted'
          }
        },
        {
        selector: ':selected',
        style: {
          'overlay-color': "#6c757d",
          'overlay-opacity': 0.3,
        }
      }
      ],
      // Layout as a Directed Acyclic Graph
      layout: daglayout
    });

/*
    cy.style()
      .selector('node')
        .style({
          'font-family' : 'cmu_serifroman',
          'background-color': 'white',
          'border-width' : 0.5,
          'border-color' : 'black'
        })
      .selector('edge')
        .style({
          'font-family' : 'cmu_serifroman',
          'background-color': 'white',
          'width' : 0.7,
          'line-color' : 'black',
          'color' : 'black',
          'text-outline-color' : 'white',
          'text-outline-width': 0.7,
        })
        .update();*/

    expand_collapse = cy.expandCollapse(
    {
      layoutBy: daglayout,
      fisheye: true,
      animate: true,
      animationDuration: ANIM_ZOOM_MS,
      undoable: false,
    //  expandCueImage: "icon-plus.png",
    //  collapseCueImage: "icon-minus.png"
    });

    // After one animation frame the browser has reflowed the gantt pane into place,
    // so cy picks up the correct container dimensions and the graph fits properly.
    requestAnimationFrame(function() { if (cy) { cy.resize(); cy.fit(); } });

    cy.on('click', function(event)
    {
      closePopup();
    });

    cy.edges().on('click', function(event)
    {
      var d = event.target.data();

      var typeHeadings = {
        'forward-dep': 'Data Dependence',
        'anti-dep':    'Anti-Dependence',
      };
      var heading = typeHeadings[d.type] || 'Edge';

      var sourceName = cy.$id(d.source).data('label') || d.source;
      var targetName = cy.$id(d.target).data('label') || d.target;

      document.getElementById('info-heading').textContent = heading;
      var info = '';
      info += '<p><strong>from</strong> '         + sourceName + '</p>';
      info += '<p><strong>to</strong> '           + targetName + '</p>';
      if (d.label)         info += '<p><strong>container</strong> '    + d.label         + '</p>';
      if (d.access_mode)   info += '<p><strong>access mode</strong> '  + d.access_mode   + '</p>';
      if (d.is_critical_path) info += '<p><strong>critical path</strong> ' + d.is_critical_path + '</p>';
      if (d.iteration_count > 1) info += '<p><strong>repeat count</strong> ' + d.iteration_count + '</p>';

      $('#info').html(info);
      openInfoPane();
      document.getElementById('plot').style.display = 'none';
    });

    document.getElementById('cy').onwheel = function(event)
    {
      event.preventDefault();
      cur = cy.pan();

      scaleX = event.deltaX * -1.5;
      scaleY = event.deltaY * -1.5;

      if (!isZooming)
        cy.panBy({ x: scaleX, y: scaleY });
    };

    document.getElementById('cy').addEventListener('gesturestart', function(e)
    {
      baseZoom = cy.zoom();
      isZooming = true;
    });

    document.getElementById('cy').addEventListener('gestureend', function(e)
    {
      isZooming = false;
    });

    document.getElementById('cy').addEventListener('gesturechange', function(e)
    {
      var rect = e.target.getBoundingClientRect();
      var x = e.clientX - rect.left;
      var y = e.clientY - rect.top;

      cy.zoom({
        level: Math.pow(e.scale, 1.2) * 1 * baseZoom,
        renderedPosition: { x: x, y: y }
      });
    }, false);

    // Display node data in the info pane when clicking on a node
    cy.nodes().on('click', function(event)
    {
      selectNode(event.target.data('id'), 'graph');
    });

    setTimeout(function()
    {
      _badgeCycleIndex = {};   // reset cycle state for the new graph
      var badges = {}

      nodes.forEach(node =>
      {
        if ((node.data.type == "skeleton_call") && node.data.line != -1)
        {
          var key = (node.data.file || '') + ':' + node.data.line;
          if (!badges.hasOwnProperty(key)) badges[key] = {"count" : 0, "ids" : [], "file" : node.data.file, "line" : node.data.line};
          badges[key].count += 1;
          badges[key].ids.push(node.data.id);
        }
      });

      for (var key in badges)
      {
        var b = badges[key];
        var basename = b.file ? b.file.split('/').pop().split('\\').pop() : '';
        var pane = document.querySelector('.cpp-tab-pane[data-filename="' + basename + '"]')
                || document.querySelector('.cpp-tab-pane');
        if (!pane) continue;
        var el = $(pane).find('.hljs-ln-n[data-line-number="' + b.line + '"]')[0];
        if (el) el.innerHTML = "<span class='badge' onclick='badgeClick(" + JSON.stringify(b.ids) + ")'>" + b.count + "</span>";
      }
    }, 200);

    let request_time = data["request_time"]; // seconds since epoch
    let response_time = data["response_time"]; // seconds since epoch
    let rendering_time = Date.now() / 1000; // milliseconds

    let modeling_duration = (response_time - request_time);
    let rendering_duration = (rendering_time - response_time);
    let total_duration = modeling_duration + rendering_duration;

    document.getElementById("modeling-time").innerHTML = Math.round(modeling_duration * 1000);
    document.getElementById("rendering-time").innerHTML = Math.round(rendering_duration * 1000);
    document.getElementById("response-time").innerHTML = Math.round(total_duration * 1000);
  });
}

function findPropertyNorm(key, ele, minmax)
{
  let mode = document.getElementById(key).value;
  var value = 0;
  if (mode == "fixed") return 0.05;
  else if (mode == "duration-total") value = ele.data('duration');
  else if (mode == "elements") value = ele.data("elements").reduce((a, b) => (a * b));
  else if (mode == "duration-per-element") value = ele.data('duration') / ele.data("elements").reduce((a, b) => (a * b));
  else if (mode == "dag-depth") value = ele.data('dag_depth');
  else if (mode == "total-order") value = ele.data('total_order');
  else if (mode == "critical-path") value = (ele.data('is_critical_path') == true ? 0 : 100);
  if (minmax[mode][1] === minmax[mode][0]) return 0.5;
  const norm = (value - minmax[mode][0]) / (minmax[mode][1] - minmax[mode][0]);
  return norm;
}

// Variant of findPropertyNorm for plain node-data objects (used by renderGantt).
function findPropertyNormFromData(key, d, mm) {
  var mode = document.getElementById(key).value;
  var value = 0;
  if (mode === 'fixed') return 0.05;
  else if (mode === 'duration-total') value = d.duration || 0;
  else if (mode === 'elements') value = (d.elements || [1]).reduce(function(a,b){return a*b;}, 1);
  else if (mode === 'duration-per-element') {
    var el = (d.elements || [1]).reduce(function(a,b){return a*b;}, 1);
    value = (d.duration || 0) / (el || 1);
  }
  else if (mode === 'dag-depth') value = d.dag_depth || 0;
  else if (mode === 'total-order') value = d.total_order || 0;
  else if (mode === 'critical-path') value = (d.is_critical_path === true ? 0 : 100);
  var mmv = mm && mm[mode];
  if (!mmv || mmv[1] === mmv[0]) return 0.5;
  return Math.max(0, Math.min(1, (value - mmv[0]) / (mmv[1] - mmv[0])));
}

function highlightSourceLine(line_nr, file_path)
{
  document.querySelectorAll('.hljs-ln-line').forEach(function(el)
  {
    el.classList.remove('line-selected');
  });

  if (line_nr == -1) return;

  // Switch to the tab matching the file, if known
  if (file_path)
  {
    var basename = file_path.split('/').pop().split('\\').pop();
    document.querySelectorAll('.cpp-tab-pane').forEach(function(pane, i)
    {
      if (pane.dataset.filename === basename)
        switchTab(document.querySelectorAll('.cpp-tab')[i], i);
    });
  }

  var activePane = document.querySelector('.cpp-tab-pane.active');
  var targets = activePane
    ? $(activePane).find('.hljs-ln-line[data-line-number="' + line_nr + '"]')
    : $('.hljs-ln-line[data-line-number="' + line_nr + '"]');

  targets.addClass('line-selected');
  if (targets[0])
    $(activePane || '#cpp_container').scrollTo(targets[0], ANIM_SCROLL_MS, {over: {top: -5}});
}

// To close popup window
function closePopup()
{
  document.getElementById('info_window').style.flex = '0 0 0em';
}

function showAlert(msg) {
  var bar = document.getElementById('alert-bar');
  if (!bar) return;
  bar.textContent = msg;
  bar.style.display = 'block';
}

function clearAlert() {
  var bar = document.getElementById('alert-bar');
  if (!bar) return;
  bar.style.display = 'none';
  bar.textContent = '';
}

// Called whenever a menu setting changes.  For small graphs it triggers an immediate
// re-render; for large graphs it shows an alert reminding the user to apply manually.
// Debounced so rapid successive calls (e.g. a checkbox dispatching its own onchange
// AND menuToggleCheck calling us) coalesce into a single action.
function settingChanged() {
  clearTimeout(_settingChangedTimer);
  _settingChangedTimer = setTimeout(function() {
    if (_lastNodeCount === 0) return; // no graph loaded yet
    if (_lastNodeCount <= AUTO_RENDER_THRESHOLD) {
      clearAlert();
      fetchViewDataAndRender();
    } else {
      showAlert('⚠ Live re-render is disabled for large graphs ('
        + _lastNodeCount + ' nodes > threshold ' + AUTO_RENDER_THRESHOLD
        + '). Use View › Apply settings to update.');
    }
  }, 50);
}

function recenterGraph()
{
  var layout = cy.layout(daglayout);
  layout.run();
}

function toggleLegend()
{
  _legendOpen = !_legendOpen;
  document.getElementById('legend').style.display = _legendOpen ? '' : 'none';
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
  saveSettings();
}




function load()
{
  // ── Custom horizontal resizer (replaces Split.js) ────────────────────────
  (function() {
    var gutter  = document.getElementById('split-gutter');
    var rightEl = document.getElementById('right_pane');
    if (!gutter || !rightEl) return;
    var _startX, _startW;

    function onMove(ev) {
      var dx   = ev.clientX - _startX;
      var cont = rightEl.parentElement;
      var minW = 120;
      var maxW = (cont ? cont.clientWidth : window.innerWidth) - 200;
      rightEl.style.flex = '0 0 ' + Math.max(minW, Math.min(_startW - dx, maxW)) + 'px';
    }
    function onUp() {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup',   onUp);
      document.body.style.cursor = '';
      if (typeof saveSettings === 'function') saveSettings();
    }
    gutter.addEventListener('mousedown', function(ev) {
      _startX = ev.clientX;
      _startW = rightEl.offsetWidth;
      document.body.style.cursor = 'col-resize';
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup',   onUp);
      ev.preventDefault();
    });
  })();

  applyStoredLayout();
  // Apply legend visibility now that #legend is in the DOM (_legendOpen was set by loadSettings).
  document.getElementById('legend').style.display = _legendOpen ? '' : 'none';
  syncSpectrumMenuIcons();   // paint the legend gradient bar with the restored spectrum
  if (!_sourcePaneOpen) closeSourcePane();
  if (_ganttPaneOpen) openGanttPane();

  hljs.highlightAll();
  hljs.initLineNumbersOnLoad();

  // hljs.initLineNumbersOnLoad() skips elements inside display:none panes.
  // Explicitly highlight and number any pane that hljs missed (e.g. the trace JSON tab).
  document.querySelectorAll('.cpp-tab-pane code').forEach(function(block) {
    if (!block.classList.contains('hljs')) {
      var fn = hljs.highlightElement || hljs.highlightBlock;
      if (fn) fn.call(hljs, block);
    }
    if (hljs.lineNumbersBlock && !block.querySelector('table')) {
      hljs.lineNumbersBlock(block);
    }
  });

  // Needed to render the graph
  fetchViewDataAndRender();


  document.getElementById("collapseRecursively").addEventListener("click", function () {
    expand_collapse.collapseRecursively(cy.$(":selected"));
  });
  document.getElementById("expandRecursively").addEventListener("click", function () {
    expand_collapse.expandRecursively(cy.$(":selected"));
  });
  document.getElementById("collapseAllRegions").addEventListener("click", function () {
    expand_collapse.collapseRecursively(cy.nodes("[type='region']"));
  });
  document.getElementById("expandAllRegions").addEventListener("click", function () {
    expand_collapse.expandRecursively(cy.nodes("[type='region']"));
  });
  document.getElementById("collapseAllFusions").addEventListener("click", function () {
    expand_collapse.collapseRecursively(cy.nodes("[type='fusion']"));
  });
  document.getElementById("expandAllFusions").addEventListener("click", function () {
    expand_collapse.expandRecursively(cy.nodes("[type='fusion']"));
  });

  var infoWindow  = document.getElementById('info_window');
  var infoResizer = document.getElementById('info-resizer');
  var _dragStartY, _dragStartH;

  infoResizer.addEventListener('mousedown', function(e) {
    _dragStartY = e.clientY;
    _dragStartH = infoWindow.offsetHeight;
    infoWindow.classList.add('resizing');
    document.addEventListener('mousemove', _onInfoDrag);
    document.addEventListener('mouseup',   _onInfoDragEnd);
    e.preventDefault();
  });

  function _onInfoDrag(e) {
    var newH = Math.max(40, _dragStartH + (_dragStartY - e.clientY));
    infoWindow.style.flex = '0 0 ' + newH + 'px';
  }

  function _onInfoDragEnd() {
    infoWindow.classList.remove('resizing');
    document.removeEventListener('mousemove', _onInfoDrag);
    document.removeEventListener('mouseup',   _onInfoDragEnd);
    var rightEl = document.getElementById('right_pane');
    if (rightEl && infoWindow.offsetHeight > 40)
      _infoPaneRatio = infoWindow.offsetHeight / rightEl.offsetHeight;
    saveSettings();
  }


  document.addEventListener("keydown", function(event) {
    // Don't steal keypresses from inputs, and ignore modifier combos.
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
    if (event.altKey || event.ctrlKey || event.metaKey) return;

    if (event.key === " ") {
      event.preventDefault();   // prevent page scroll
      fetchViewDataAndRender();
    }
    else if (event.key === "a") { toggleSetting('container-allocations'); }
    else if (event.key === "d") { toggleSetting('container-deallocations'); }
    else if (event.key === "n") { toggleSetting('anti-deps'); }
    else if (event.key === "t") { toggleSetting('container-transfers'); }
    else if (event.key === "r") { toggleSetting('show_regions'); }
    else if (event.key === "u") { toggleSetting('container-updates'); }
    else if (event.key === "l") { toggleSetting('edge-labels'); }
    else if (event.key === "e") {
      expand_collapse.expandRecursively(cy.$(":selected"));
    }
    else if (event.key === "c") {
      expand_collapse.collapseRecursively(cy.$(":selected"));
    }
    else if (event.key === "z") { zoomGanttToMeasuredInterval(); }
    else if (event.key === "g") { toggleGanttPane(); }
    else if (event.key === "s") { toggleSourcePane(); }
    else if (event.key === "Tab") {
      // Cycle forward through source-code tabs (wraps around).
      var tabs  = Array.from(document.querySelectorAll('.cpp-tab'));
      if (tabs.length > 1) {
        event.preventDefault();
        var cur  = tabs.findIndex(function(t) { return t.classList.contains('active'); });
        var next = (cur + 1) % tabs.length;
        switchTab(tabs[next], next);
      }
    }
    else if (event.key === "x" || event.key === "Escape") {
      closePopup();
    }
  });

  // Re-render Gantt on window resize (debounced) so the 1x scale is recomputed
  // while keeping the current visible time range centred.
  var _resizeTimer = null;
  window.addEventListener('resize', function() {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(function() {
      if (_ganttPaneOpen && _lastGraphNodes.length)
        reRenderGanttPreservingScroll();
    }, 60);
  });
}
