// heatmap.js — Content-snapshot heatmap renderer for the info pane.
// Renders 1–4D dense numeric datasets as colour-mapped canvas heatmaps.
// The innermost two dimensions are always displayed as a 2D heatmap (or a 1D
// strip for 1D data).  Each outer dimension gets an independent navigation bar
// above the canvas.  A separate iteration bar appears below the legend when
// multiple snapshots are present.
//
// Snapshot format (content_snapshots[] entries from /get_data):
//   Line 1:  "<dim> <ext0> [<ext1> [<ext2> [<ext3>]]]"
//   Line 2+: space-separated numbers in row-major order

// ── Viridis colormap (9 key stops, linearly interpolated) ─────────────────
var _V = [
  [0.000,  68,   1,  84],
  [0.125,  72,  40, 120],
  [0.250,  62,  74, 137],
  [0.375,  49, 104, 142],
  [0.500,  38, 130, 142],
  [0.625,  31, 158, 137],
  [0.750,  53, 183, 121],
  [0.875, 110, 206,  88],
  [1.000, 253, 231,  37],
];

function _viridis(t) {
  t = Math.max(0, Math.min(1, t));
  var i = 0;
  while (i < _V.length - 2 && _V[i + 1][0] <= t) i++;
  var lo = _V[i], hi = _V[i + 1];
  var f  = (t - lo[0]) / (hi[0] - lo[0]);
  return [
    Math.round(lo[1] + f * (hi[1] - lo[1])),
    Math.round(lo[2] + f * (hi[2] - lo[2])),
    Math.round(lo[3] + f * (hi[3] - lo[3])),
  ];
}

function _fmt(v) {
  if (!isFinite(v)) return String(v);
  if (Number.isInteger(v)) return String(v);
  return parseFloat(v.toPrecision(4)).toString();
}

// ── Module-level state ─────────────────────────────────────────────────────
var _snapshots    = [];    // raw snapshot strings
var _curIdx       = 0;    // currently displayed snapshot index
var _labels       = [];   // parallel label strings ("A · 1", …)
var _virtualLabel = null; // virtual container name, or null

// Geometry (fixed for the lifetime of the current heatmap render)
var _rows         = 0;
var _cols         = 0;
var _cellW        = 0;
var _cellH        = 0;
var _canvasW      = 0;
var _canvasH      = 0;
var _totalCount   = 0;    // product of all extents = values per snapshot
var _outerExtents = [];   // sizes of outer dims (empty for 1D/2D)
var _outerStrides = [];   // row-major stride for each outer dim
var _outerIndices = [];   // current navigation index for each outer dim

// Live DOM references — updated in-place on navigation (no DOM rebuild)
var _ctx          = null;
var _curVals      = null; // parsed values object for the current snapshot
var _counterEl    = null; // span showing iteration "3 / 10"
var _legendLabEl  = null; // div.heatmap-legend-labels
var _outerNavEls  = [];   // [{counterEl}] per outer dim

// ── Internal helpers ───────────────────────────────────────────────────────

// Parse the header line.  Returns {dim, rows, cols, outerExtents} or null.
// rows/cols are the inner two display dimensions; outerExtents are everything
// before them.  For 1D data rows=1, cols=N, outerExtents=[].
function _parseHeader(text) {
  var nl = text.indexOf('\n');
  if (nl < 0) return null;
  var hdr = text.slice(0, nl).trim().split(/\s+/).map(Number);
  var dim = hdr[0];
  if (dim < 1 || dim > 4) return null;
  var exts = hdr.slice(1);
  if (exts.length !== dim) return null;
  for (var i = 0; i < exts.length; i++) if (!(exts[i] > 0)) return null;

  var rows         = dim === 1 ? 1         : exts[dim - 2];
  var cols         = exts[dim - 1];
  var outerExtents = dim > 2  ? exts.slice(0, dim - 2) : [];
  return { dim: dim, exts: exts, rows: rows, cols: cols, outerExtents: outerExtents };
}

// Parse all values from a snapshot; returns {nums, vMin, vMax} or null.
function _parseValues(text, count) {
  var nl = text.indexOf('\n');
  if (nl < 0) return null;
  var tokens = text.slice(nl + 1).trim().split(/\s+/);
  if (tokens.length < count) return null;
  var nums = new Float64Array(count);
  var vMin = Infinity, vMax = -Infinity;
  for (var i = 0; i < count; i++) {
    var n = parseFloat(tokens[i]);
    nums[i] = n;
    if (n < vMin) vMin = n;
    if (n > vMax) vMax = n;
  }
  return { nums: nums, vMin: vMin, vMax: vMax };
}

// Compute the flat index offset for the current outer navigation indices.
function _sliceBase() {
  var base = 0;
  for (var k = 0; k < _outerIndices.length; k++)
    base += _outerIndices[k] * _outerStrides[k];
  return base;
}

// Fill the canvas using the current outer navigation indices.
function _fillCanvas(vals) {
  var imgD  = _ctx.createImageData(_canvasW, _canvasH);
  var px    = imgD.data;
  var range = vals.vMax - vals.vMin;
  var base  = _sliceBase();
  for (var r = 0; r < _rows; r++) {
    for (var c = 0; c < _cols; c++) {
      var v   = vals.nums[base + r * _cols + c];
      var t   = range > 0 ? (v - vals.vMin) / range : 0.5;
      var rgb = _viridis(t);
      for (var pr = 0; pr < _cellH; pr++) {
        var rowBase = ((r * _cellH + pr) * _canvasW + c * _cellW) * 4;
        for (var pc = 0; pc < _cellW; pc++) {
          var pidx = rowBase + pc * 4;
          px[pidx]     = rgb[0];
          px[pidx + 1] = rgb[1];
          px[pidx + 2] = rgb[2];
          px[pidx + 3] = 255;
        }
      }
    }
  }
  _ctx.putImageData(imgD, 0, 0);
}

// Update the legend min/max labels in place.
function _updateLegendLabels(vMin, vMax) {
  if (!_legendLabEl) return;
  _legendLabEl.innerHTML =
    '<span><span class="heatmap-legend-minmax-label">min</span> ' + _fmt(vMin) + '</span>' +
    '<span>' + _fmt(vMax) + ' <span class="heatmap-legend-minmax-label">max</span></span>';
}

// Update the iteration counter in place.
function _updateCounter() {
  if (!_counterEl) return;
  var label = (_labels.length > _curIdx) ? '  —  ' + _labels[_curIdx] : '';
  _counterEl.textContent = (_curIdx + 1) + ' / ' + _snapshots.length + label;
}

// Update one outer-dim counter in place.
function _updateOuterCounter(k) {
  if (!_outerNavEls[k]) return;
  _outerNavEls[k].textContent = (_outerIndices[k] + 1) + ' / ' + _outerExtents[k];
}

// ── Public API ─────────────────────────────────────────────────────────────

// Navigate the iteration axis by delta (-1 or +1).
// Returns true if navigation happened (use for preventDefault on arrow keys).
export function navigateHeatmap(delta) {
  if (_snapshots.length <= 1) return false;
  _curIdx  = (_curIdx + delta + _snapshots.length) % _snapshots.length;
  var vals = _parseValues(_snapshots[_curIdx], _totalCount);
  if (vals) {
    _curVals = vals;
    _fillCanvas(vals);
    _updateLegendLabels(vals.vMin, vals.vMax);
  }
  _updateCounter();
  console.log('[heatmap] Iteration ' + (_curIdx + 1) + ' / ' + _snapshots.length);
  return true;
}

// Main entry point.  Called by InfoView whenever a node is selected.
// container    — DOM element to render into (cleared on each call).
// snapshots    — array of raw snapshot strings, single string, or falsy.
// labels       — optional parallel label strings for each snapshot.
// virtualLabel — optional virtual container group name.
export function renderHeatmap(container, snapshots, labels, virtualLabel) {
  // Normalise to array.
  if (!snapshots || (Array.isArray(snapshots) && snapshots.length === 0)) {
    container.innerHTML = '';
    return;
  }
  if (!Array.isArray(snapshots)) snapshots = [snapshots];

  // Reset module state.
  _snapshots    = snapshots;
  _curIdx       = 0;
  _labels       = Array.isArray(labels) ? labels : [];
  _virtualLabel = virtualLabel || null;
  _ctx = _counterEl = _legendLabEl = null;
  _outerNavEls  = [];

  container.innerHTML = '';

  // ── Parse first snapshot for geometry ─────────────────────────────────
  var geo = _parseHeader(_snapshots[0]);
  if (!geo) {
    console.warn('[heatmap] Parse error: bad header in first snapshot.');
    return;
  }

  _rows         = geo.rows;
  _cols         = geo.cols;
  _outerExtents = geo.outerExtents;
  _outerIndices = new Array(_outerExtents.length).fill(0);

  // Compute row-major strides for outer dimensions and total element count.
  // For extents [D0, D1, R, C]: strides[1]=R*C, strides[0]=D1*R*C, total=D0*D1*R*C
  _outerStrides = new Array(_outerExtents.length);
  var s = _rows * _cols;
  for (var k = _outerExtents.length - 1; k >= 0; k--) {
    _outerStrides[k] = s;
    s *= _outerExtents[k];
  }
  _totalCount = s;

  var vals = _parseValues(_snapshots[0], _totalCount);
  if (!vals) {
    console.warn('[heatmap] Parse error: insufficient values in first snapshot.');
    return;
  }
  _curVals = vals;

  console.log('[heatmap] dim=' + geo.dim + ' extents=[' + geo.exts.join(',') + ']'
    + ' snapshots=' + snapshots.length);

  // ── Cell / canvas geometry ─────────────────────────────────────────────
  var MAX_CELL = 32, MIN_1D_H = 48, MAX_PX = 1024;
  var cW = container.clientWidth || 300;

  _cellW   = Math.max(1, Math.min(MAX_CELL, Math.floor(Math.min(cW, MAX_PX) / _cols)));
  _cellH   = (geo.dim === 1) ? Math.max(MIN_1D_H, _cellW) : _cellW;
  if (_rows * _cellH > MAX_PX) _cellH = Math.max(1, Math.floor(MAX_PX / _rows));
  _canvasW = _cols * _cellW;
  _canvasH = _rows * _cellH;

  // ── Dimension label ────────────────────────────────────────────────────
  var dimLabel = document.createElement('div');
  dimLabel.className = 'heatmap-dim-label';
  var shapeStr = geo.exts.join(' \xd7 ');
  var dimText  = 'Content snapshot — ' + geo.dim + 'D (' + shapeStr + ')';
  if (_virtualLabel) dimText += '  \xb7  virtual container "' + _virtualLabel + '"';
  dimLabel.textContent = dimText;
  container.appendChild(dimLabel);

  // ── Outer-dimension navigation bars (only for dim > 2) ────────────────
  for (var k = 0; k < _outerExtents.length; k++) {
    (function(dim_k) {
      var navRow = document.createElement('div');
      navRow.className = 'heatmap-nav heatmap-nav-outer';

      var lbl = document.createElement('span');
      lbl.className   = 'heatmap-nav-axis-label';
      lbl.textContent = 'axis ' + dim_k;
      navRow.appendChild(lbl);

      var btnPrev = document.createElement('button');
      btnPrev.className   = 'heatmap-nav-btn';
      btnPrev.textContent = '←';
      btnPrev.title       = 'Previous slice along axis ' + dim_k;
      btnPrev.addEventListener('click', function() {
        _outerIndices[dim_k] = (_outerIndices[dim_k] - 1 + _outerExtents[dim_k]) % _outerExtents[dim_k];
        _fillCanvas(_curVals);
        _updateOuterCounter(dim_k);
      });

      var counter = document.createElement('span');
      counter.className = 'heatmap-nav-counter';
      _outerNavEls[dim_k] = counter;
      _updateOuterCounter(dim_k);

      var btnNext = document.createElement('button');
      btnNext.className   = 'heatmap-nav-btn';
      btnNext.textContent = '→';
      btnNext.title       = 'Next slice along axis ' + dim_k;
      btnNext.addEventListener('click', function() {
        _outerIndices[dim_k] = (_outerIndices[dim_k] + 1) % _outerExtents[dim_k];
        _fillCanvas(_curVals);
        _updateOuterCounter(dim_k);
      });

      navRow.appendChild(btnPrev);
      navRow.appendChild(counter);
      navRow.appendChild(btnNext);
      container.appendChild(navRow);
    })(k);
  }

  // ── Canvas ─────────────────────────────────────────────────────────────
  var canvasWrap = document.createElement('div');
  canvasWrap.className  = 'heatmap-canvas-wrap';
  canvasWrap.style.height = Math.round(cW * _canvasH / _canvasW) + 'px';

  var canvas = document.createElement('canvas');
  canvas.width  = _canvasW;
  canvas.height = _canvasH;
  canvas.style.display        = 'block';
  canvas.style.width          = 'auto';
  canvas.style.height         = '100%';
  canvas.style.imageRendering = 'pixelated';
  canvas.className = 'heatmap-canvas';
  _ctx = canvas.getContext('2d');
  _fillCanvas(vals);
  canvasWrap.appendChild(canvas);
  container.appendChild(canvasWrap);

  // ── Legend ─────────────────────────────────────────────────────────────
  var legendDiv = document.createElement('div');
  legendDiv.className = 'heatmap-legend';

  var title = document.createElement('div');
  title.className   = 'heatmap-legend-title';
  title.textContent = 'Value range';
  legendDiv.appendChild(title);

  var barH = 12;
  var bar  = document.createElement('canvas');
  bar.width  = cW;
  bar.height = barH;
  bar.className = 'heatmap-legend-bar';
  var bctx = bar.getContext('2d');
  var bimg = bctx.createImageData(cW, barH);
  var bpx  = bimg.data;
  for (var x = 0; x < cW; x++) {
    var rgb = _viridis(x / (cW - 1 || 1));
    for (var y = 0; y < barH; y++) {
      var bidx = (y * cW + x) * 4;
      bpx[bidx]     = rgb[0];
      bpx[bidx + 1] = rgb[1];
      bpx[bidx + 2] = rgb[2];
      bpx[bidx + 3] = 255;
    }
  }
  bctx.putImageData(bimg, 0, 0);
  legendDiv.appendChild(bar);

  _legendLabEl = document.createElement('div');
  _legendLabEl.className = 'heatmap-legend-labels';
  _updateLegendLabels(vals.vMin, vals.vMax);
  legendDiv.appendChild(_legendLabEl);

  container.appendChild(legendDiv);

  // ── Iteration navigation bar (only when multiple snapshots) ───────────
  if (_snapshots.length > 1) {
    var nav = document.createElement('div');
    nav.className = 'heatmap-nav';

    var lbl2 = document.createElement('span');
    lbl2.className   = 'heatmap-nav-axis-label';
    lbl2.textContent = 'iter';
    nav.appendChild(lbl2);

    var btnPrev2 = document.createElement('button');
    btnPrev2.className   = 'heatmap-nav-btn';
    btnPrev2.textContent = '←';
    btnPrev2.title       = 'Previous iteration (←)';
    btnPrev2.addEventListener('click', function() { navigateHeatmap(-1); });

    _counterEl = document.createElement('span');
    _counterEl.className = 'heatmap-nav-counter';
    _updateCounter();

    var btnNext2 = document.createElement('button');
    btnNext2.className   = 'heatmap-nav-btn';
    btnNext2.textContent = '→';
    btnNext2.title       = 'Next iteration (→)';
    btnNext2.addEventListener('click', function() { navigateHeatmap(1); });

    nav.appendChild(btnPrev2);
    nav.appendChild(_counterEl);
    nav.appendChild(btnNext2);
    container.appendChild(nav);
  }
}
