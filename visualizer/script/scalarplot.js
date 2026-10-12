// scalarplot.js — The values of a scalar over the iterations of its node, as a line plot in the info pane.
//
// A scalar node that the iteration-coalescing made of several iterations has one value per iteration (the trace has the value of every
// scalar that a skeleton returned: error norms, sums, residuals). The x axis is the iteration. Values that are not finite ('nan', 'inf',
// '-inf' in the trace) leave a gap. The y axis is linear, or logarithmic: a convergence plot of positive numbers over many orders of
// magnitude starts logarithmic, and the log scale can be switched with the checkbox of the plot when all the values are above 0.

var VIEW_W = 280, VIEW_H = 130;
var PAD_L = 46, PAD_R = 8, PAD_T = 8, PAD_B = 20;

/** A short text of a number for the labels: 5 significant digits, or an exponent for the very small and the very large ones. */
export function formatScalar(v) {
  if (typeof v !== 'number') return String(v);
  if (!isFinite(v)) return String(v);
  if (v === 0) return '0';
  var a = Math.abs(v);
  if (a >= 1e6 || a < 1e-3) return v.toExponential(3).replace(/\.?0+e/, 'e');
  var text = v.toPrecision(5);
  return text.indexOf('.') >= 0 ? text.replace(/\.?0+$/, '') : text;
}

/** The numbers of the values with their iteration (from 0), and the range. The values that are not numbers are left out. */
export function scalarSeries(values) {
  var points = [];
  for (var i = 0; i < values.length; i++)
    if (typeof values[i] === 'number' && isFinite(values[i])) points.push({ i: i, v: values[i] });
  var min = Infinity, max = -Infinity;
  for (var k = 0; k < points.length; k++) {
    if (points[k].v < min) min = points[k].v;
    if (points[k].v > max) max = points[k].v;
  }
  return { points: points, min: min, max: max, positive: points.length > 0 && min > 0 };
}

/** Does the plot start with a log scale: positive numbers over 3 orders of magnitude or more. */
export function startsLogarithmic(series) {
  return series.positive && series.max / series.min >= 1e3;
}

/** The SVG of the plot (a string): the line, the points and the labels. Pure, so that it can be tested. */
export function scalarPlotSvg(values, options) {
  options = options || {};
  var series = scalarSeries(values);
  var n = values.length;
  if (series.points.length < 2) return '';
  var log = !!options.log && series.positive;
  var tr = log ? Math.log10 : function(x) { return x; };
  var lo = tr(series.min), hi = tr(series.max);
  if (hi === lo) { lo -= 0.5; hi += 0.5; }
  var w = VIEW_W - PAD_L - PAD_R, h = VIEW_H - PAD_T - PAD_B;
  var x = function(i) { return PAD_L + (n === 1 ? 0 : i / (n - 1)) * w; };
  var y = function(v) { return PAD_T + (1 - (tr(v) - lo) / (hi - lo)) * h; };

  var out = ['<svg class="scalar-plot-svg" xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 ' + VIEW_W + ' ' + VIEW_H + '">'];
  // axes and the labels of the ends
  out.push('<line class="scalar-plot-axis" x1="' + PAD_L + '" y1="' + PAD_T + '" x2="' + PAD_L + '" y2="' + (PAD_T + h) + '"/>');
  out.push('<line class="scalar-plot-axis" x1="' + PAD_L + '" y1="' + (PAD_T + h) + '" x2="' + (PAD_L + w) + '" y2="' + (PAD_T + h) + '"/>');
  out.push('<text class="scalar-plot-label" x="' + (PAD_L - 4) + '" y="' + (PAD_T + 3) + '" text-anchor="end">' + formatScalar(series.max) + '</text>');
  out.push('<text class="scalar-plot-label" x="' + (PAD_L - 4) + '" y="' + (PAD_T + h) + '" text-anchor="end">' + formatScalar(series.min) + '</text>');
  out.push('<text class="scalar-plot-label" x="' + PAD_L + '" y="' + (VIEW_H - 5) + '" text-anchor="start">1</text>');
  out.push('<text class="scalar-plot-label" x="' + (PAD_L + w) + '" y="' + (VIEW_H - 5) + '" text-anchor="end">' + n + '</text>');
  out.push('<text class="scalar-plot-label" x="' + (PAD_L + w / 2) + '" y="' + (VIEW_H - 5) + '" text-anchor="middle">iteration</text>');
  // the line (broken where a value is not a number)
  var path = '', pen = false;
  for (var i = 0; i < n; i++) {
    var v = values[i];
    if (typeof v === 'number' && isFinite(v)) {
      path += (pen ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(v).toFixed(1);
      pen = true;
    } else pen = false;
  }
  out.push('<path class="scalar-plot-line" d="' + path + '"/>');
  // the points, each with its value (the circles are what is hovered and clicked)
  for (var k = 0; k < series.points.length; k++) {
    var p = series.points[k];
    out.push('<circle class="scalar-plot-point" cx="' + x(p.i).toFixed(1) + '" cy="' + y(p.v).toFixed(1) + '" r="' + (n > 60 ? 1.6 : 2.4) + '"' +
             ' data-iter-idx="' + p.i + '"><title>Iteration ' + (p.i + 1) + ': ' + formatScalar(p.v) + '</title></circle>');
  }
  out.push('</svg>');
  return out.join('');
}

/** The title row (with the checkbox of the log scale, in the state of the plot) and the plot, as HTML. Pure: a redraw is made of it. */
export function scalarPlotHtml(values, log) {
  var series = scalarSeries(values);
  var title = '<div class="scalar-plot-title">Value by iteration' +
    (series.positive ? '<label class="scalar-plot-log"><input type="checkbox"' + (log ? ' checked' : '') + '> log scale</label>' : '') + '</div>';
  return title + scalarPlotSvg(values, { log: log });
}

/**
 * Draws the plot of the values into the element (nothing, and hidden, for fewer than two numbers). A click on a point selects the iteration:
 * options.onSelectIteration(nodeId, index) is called with options.nodeId.
 */
export function renderScalarPlot(container, values, options) {
  options = options || {};
  container.innerHTML = '';
  var series = values && values.length >= 2 ? scalarSeries(values) : null;
  if (!series || series.points.length < 2) { container.style.display = 'none'; return; }
  container.style.display = 'block';

  var log = startsLogarithmic(series);
  var draw = function() {
    container.innerHTML = scalarPlotHtml(values, log);
    var box = container.querySelector('input[type=checkbox]');
    if (box) box.onchange = function() { log = box.checked; draw(); };
  };
  draw();

  container.onclick = function(ev) {
    var point = ev.target.closest && ev.target.closest('circle[data-iter-idx]');
    if (point && options.onSelectIteration) options.onSelectIteration(options.nodeId, parseInt(point.dataset.iterIdx, 10));
  };
}
