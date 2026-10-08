// LegendView.js — the legend panel of a graph of SkeVU (GraphView, or the graph of the designer): the kinds of nodes and edges that the graph
// has, and what the colour of a call means (the spectrum of a measure, or the skeletons or backends that it is coloured by).
//
// The graph (`graphView`) has to say: hasElements(selector) (is a kind of node or edge in the graph), _spectrumColor(t) and _spectrumColorDark(t),
// _provider (backendColor, patternColor), and callCategories(mode) (the names of the backends or patterns, mode 'backend' or 'pattern', that the calls in the graph have).
//
// Usage (see graph.js):
//
//   const legend = new LegendView(document.getElementById('legend'), graphView, options);
//   legend.settings = sharedSettingsObject;   // the same reference as graphView.settings
//   legend.render(darkMode);                  // after each render / dark-mode change
//   legend.highlightFor(ele);                 // the entry for a node or an edge that the pointer is over (legend.clearHighlight() when it leaves)
//
// options (all optional):
//   labels      { key: text }: another name for an entry (the key is its data-legend-key: 'container_update', ...)
//   extraNodes  [{ key, label, icon(dark) -> svg markup, present() -> bool }]: kinds of nodes that only the page has
//   typeKeys    { node type: key }: the entry of such a kind of node, for highlightFor

import { COLORS, regionColor } from './graph_style.js';
import { BACKEND_ORDER, PATTERN_ORDER, basePattern } from './TraceDataProvider.js';

export class LegendView {
  // legendEl  — the #legend container element (safe to pass null; render() is a no-op)
  // graphView — the associated GraphView instance (used for hasElements + spectrum colors)
  constructor(legendEl, graphView, options) {
    this._legendEl  = legendEl;
    this._gv        = graphView;
    this._options   = options || {};
    this.settings   = {};   // set by graph.js to the shared settings reference
  }

  // Build or rebuild the full legend panel.
  render(darkMode) {
    if (!this._legendEl) return;

    var dark    = !!darkMode;
    var midClr  = this._gv._spectrumColor(0.15);
    var midDark = this._gv._spectrumColorDark(0.15);

    // ── Theme-aware colours ────────────────────────────────────────────────
    var textClr   = dark ? '#c8c0d8' : '#333333';
    var edgeClr   = COLORS.edge[dark ? 1 : 0];
    var ctFill    = COLORS.data.fill,         ctStroke = COLORS.data.stroke;            // container_update / scalar
    var allocFill = COLORS.allocation.fill,   allocStr = COLORS.allocation.stroke;
    var dealFill  = COLORS.deallocation.fill, dealStr  = COLORS.deallocation.stroke;
    // region depth-0 colours, matching regionColorLight/Dark in GraphView
    var regionF   = regionColor(0, dark);
    var regionS   = dark ? 'rgba(200,160,80,0.5)' : 'rgba(70,155,220,0.6)';
    var fusionF   = COLORS.fusion.fill;
    var fusionS   = '#6655cc';

    // ── SVG shape helpers ──────────────────────────────────────────────────
    var STAR = '10,2 11.8,7.6 17.6,7.5 12.9,10.9 14.7,16.5 10,13 5.3,16.5 7.2,10.9 2.4,7.5 8.2,7.6';
    // Octagon rotated 22.5° so edges (not vertices) face top and bottom.
    var OCTA = '17.4,6.9 17.4,13.1 13.1,17.4 6.9,17.4 2.6,13.1 2.6,6.9 6.9,2.6 13.1,2.6';

    function nodeIcon(inner) {
      return '<svg viewBox="0 0 20 20" width="18" height="18">' + inner + '</svg>';
    }

    function edgeIcon(dashArray, color) {
      color = color || edgeClr;
      var da = dashArray ? ' stroke-dasharray="' + dashArray + '"' : '';
      return '<svg viewBox="0 0 44 12" width="40" height="12">' +
        '<line x1="2" y1="6" x2="34" y2="6" stroke="' + color + '" stroke-width="1.5"' + da + '/>' +
        '<polygon points="44,6 32,2 32,10" fill="' + color + '"/>' +
        '</svg>';
    }

    // Returns true if the current graph contains at least one element matching selector.
    var gv = this._gv;
    function inGraph(selector) {
      return gv.hasElements(selector);
    }

    // ── Node entries ───────────────────────────────────────────────────────
    var nodeEntries = [];
    if (inGraph('node[type="skeleton_call"]'))
      nodeEntries.push([nodeIcon('<circle cx="10" cy="10" r="9" fill="' + midClr + '" stroke="' + midDark + '" stroke-width="1.5"/>'),
       'Skeleton', 'skeleton']);
    if (inGraph('node[type="external"]'))
      nodeEntries.push([nodeIcon('<polygon points="' + OCTA + '" fill="' + midClr + '" stroke="' + midDark + '" stroke-width="1.5"/>'),
       'External', 'external']);
    if (inGraph('node[type="scalar"]'))
      nodeEntries.push([nodeIcon('<polygon points="10,2 18,17 2,17" fill="' + ctFill + '" stroke="' + ctStroke + '" stroke-width="2"/>'),
       'Scalar', 'scalar-node']);
    if (inGraph('node[type="transfer"]'))
      nodeEntries.push([nodeIcon('<polygon points="10,1 19,10 10,19 1,10" fill="purple" stroke="#6a006a" stroke-width="1.5"/>'),
       'Transfer', 'transfer']);
    if (inGraph('node[type="container_update"]'))
      nodeEntries.push([nodeIcon('<rect x="3" y="3" width="14" height="14" fill="' + ctFill + '" stroke="' + ctStroke + '" stroke-width="2"/>'),
       'Data Update', 'container_update']);
    if (inGraph('node[type="allocation"][!is_prng]'))
      nodeEntries.push([nodeIcon('<polygon points="' + STAR + '" fill="' + allocFill + '" stroke="' + allocStr + '" stroke-width="1"/>'),
       'Allocation', 'allocation']);
    if (inGraph('node[type="deallocation"]'))
      nodeEntries.push([nodeIcon('<polygon points="' + STAR + '" fill="' + dealFill + '" stroke="' + dealStr + '" stroke-width="1"/>'),
       'Deallocation', 'deallocation']);
    if (this.settings['show_regions'] && inGraph('node[type="region"]'))
      nodeEntries.push([nodeIcon('<rect x="1" y="3" width="18" height="14" rx="2" fill="' + regionF + '" stroke="' + regionS + '" stroke-width="1.5"/>'),
       'Instrumented Region', 'region']);
    if (inGraph('node[type="fusion"]'))
      nodeEntries.push([nodeIcon('<rect x="1" y="3" width="18" height="14" rx="2" fill="' + fusionF + '" stroke="' + fusionS + '" stroke-width="1.5"/>'),
       'Fusion Suggestion', 'fusion']);

    if (inGraph('node[type="allocation"][?is_prng]'))
      nodeEntries.push([nodeIcon('<polygon points="5,3 15,3 19,10 15,17 5,17 1,10" fill="' + COLORS.prng.fill[dark ? 1 : 0] + '" stroke="' + COLORS.prng.stroke[dark ? 1 : 0] + '" stroke-width="1.5"/>'),
       'Random stream', 'prng-node']);
    (this._options.extraNodes || []).forEach(function(extra) {
      if (extra.present()) nodeEntries.push([nodeIcon(extra.icon(dark)), extra.label, extra.key]);
    });

    // ── Edge entries ───────────────────────────────────────────────────────
    var aliasClr = COLORS.alias[dark ? 1 : 0];
    var aliasIcon =
      '<svg viewBox="0 0 44 12" width="40" height="12">' +
      '<line x1="2" y1="6" x2="42" y2="6" stroke="' + aliasClr + '" stroke-width="1.5" stroke-dasharray="8 4"/>' +
      '</svg>';

    var edgeEntries = [];
    if (inGraph('edge[type="forward-dep"][access_mode="proxy"]'))
      edgeEntries.push([edgeIcon(''),      'Full',            'proxy']);
    if (inGraph('edge[access_mode="elwise"]'))
      edgeEntries.push([edgeIcon('5 3'),   'Elwise',          'elwise']);
    if (inGraph('edge[access_mode="scalar"]'))
      edgeEntries.push([edgeIcon('1.5 3'), 'Scalar',          'scalar']);
    if (inGraph('edge[type="anti-dep"]'))
      edgeEntries.push([edgeIcon('', COLORS.antiDep), 'Anti',          'anti-dep']);
    if (inGraph('edge[type="virtual-alias"]'))
      edgeEntries.push([aliasIcon,           'Virtual Dataset','virtual-alias']);
    if (inGraph('edge[type="prng-dep"]')) {
      var prngIcon =
        '<svg viewBox="0 0 44 12" width="40" height="12">' +
        '<polyline points="2,6 9,3.6 16,8.4 23,3.6 30,8.4 37,6" stroke="' + edgeClr + '" stroke-width="1.5" fill="none"/>' +
        '<polygon points="44,6 32,2 32,10" fill="' + edgeClr + '"/>' +
        '</svg>';
      edgeEntries.push([prngIcon, 'PRNG', 'prng-dep']);
    }

    var labels = this._options.labels || {};
    function item(icon, label, key) {
      label = (key && labels[key]) || label;
      var keyAttr = key ? ' data-legend-key="' + key + '"' : '';
      return '<div class="legend-item"' + keyAttr + '>' + icon +
             '<span style="color:' + textClr + '">' + label + '</span></div>';
    }

    function section(title, entries) {
      return '<div class="legend-section">' +
        '<div class="legend-section-title">' + title + '</div>' +
        '<div class="legend-items">' +
        entries.map(function(e) { return item(e[0], e[1], e[2]); }).join('') +
        '</div></div>';
    }

    // What the colour of a call means: the spectrum of a measure, or (when the calls are coloured by pattern or by backend) the patterns or
    // backends that the calls in the graph have
    var mode = this.settings['node-color-call'];
    var colourSection;
    if (mode === 'pattern' || mode === 'backend') {
      var provider = gv._provider;
      var order = mode === 'pattern' ? PATTERN_ORDER : BACKEND_ORDER;
      var names = gv.callCategories(mode).slice().sort(function(a, b) {        // (in the order of the spectrum, the others last)
        var ia = order.indexOf(a), ib = order.indexOf(b);
        return (ia < 0 ? order.length : ia) - (ib < 0 ? order.length : ib);
      });
      colourSection = section(mode === 'pattern' ? 'Skeletons' : 'Backends', names.map(function(name) {
        var fill = mode === 'pattern' ? provider.patternColor(name) : provider.backendColor(name);
        var line = mode === 'pattern' ? provider.patternColor(name, true) : provider.backendColor(name, true);
        return ['<i class="legend-swatch" style="background:' + fill + ';border-color:' + line + '"></i>', name, 'swatch:' + name];
      }));
    } else {
      colourSection =
        '<div class="legend-section legend-spectrum-section">' +
        '<div class="legend-section-title">Node color</div>' +
        '<div class="legend-spectrum-row">' +
        '<span class="legend-spec-end" style="color:' + textClr + '">min</span>' +
        '<span id="color-bar"></span>' +
        '<span class="legend-spec-end" style="color:' + textClr + '">max</span>' +
        '</div></div>';
    }

    this._legendEl.innerHTML = colourSection + section('Nodes', nodeEntries) + section('Dependences', edgeEntries);

    // Populate the freshly-inserted #color-bar gradient.
    this._updateColorBar();
  }

  /** The data-legend-key of the entry for a node or an edge of the graph (a Cytoscape element), or null. */
  keyFor(ele) {
    var d = ele.data();
    if (ele.isNode()) {
      if (d.type === 'allocation' && d.is_prng) return 'prng-node';
      return (this._options.typeKeys && this._options.typeKeys[d.type]) || ({
        'skeleton_call':    'skeleton',
        'external':         'external',
        'scalar':           'scalar-node',
        'transfer':         'transfer',
        'container_update': 'container_update',
        'allocation':       'allocation',
        'deallocation':     'deallocation',
        'region':           'region',
        'fusion':           'fusion',
      })[d.type] || null;
    }
    if (d.type === 'anti-dep')      return 'anti-dep';
    if (d.type === 'virtual-alias') return 'virtual-alias';
    if (d.type === 'prng-dep')      return 'prng-dep';
    if (d.access_mode === 'elwise') return 'elwise';
    if (d.access_mode === 'scalar') return 'scalar';
    if (d.type === 'forward-dep')   return 'proxy';
    return null;
  }

  /** Mark the entry for a node or an edge (and, for a call, the swatch of the pattern or backend that it is coloured by). */
  highlightFor(ele) {
    this.highlightKey(this.keyFor(ele));
    var mode = this.settings['node-color-call'];
    if (!this._legendEl || ele.data('type') !== 'skeleton_call' || (mode !== 'pattern' && mode !== 'backend')) return;
    var name  = mode === 'pattern' ? basePattern(ele.data('pattern')) : ele.data('backend');
    var entry = this._legendEl.querySelector('[data-legend-key="swatch:' + name + '"]');
    if (entry) entry.classList.add('legend-item-hovered');
  }

  /**
   * Highlight the legend entry whose data-legend-key equals `key`.
   * Clears any previously-highlighted entry first.  Pass null to just clear.
   */
  highlightKey(key) {
    if (!this._legendEl) return;
    this._legendEl.querySelectorAll('.legend-item.legend-item-hovered')
      .forEach(function(el) { el.classList.remove('legend-item-hovered'); });
    if (!key) return;
    var entry = this._legendEl.querySelector('[data-legend-key="' + key + '"]');
    if (entry) entry.classList.add('legend-item-hovered');
  }

  /** Remove the legend hover highlight from any currently-highlighted entry. */
  clearHighlight() {
    if (!this._legendEl) return;
    this._legendEl.querySelectorAll('.legend-item.legend-item-hovered')
      .forEach(function(el) { el.classList.remove('legend-item-hovered'); });
  }

  // Repaint only the #color-bar gradient swatch (no full legend rebuild).
  _updateColorBar() {
    var bar = this._legendEl && this._legendEl.querySelector('#color-bar');
    if (!bar) return;
    // Sample the active spectrum at 20 evenly-spaced points to produce a smooth
    // linear-gradient.  This avoids importing COLOR_SPECTRUMS directly and keeps
    // the gradient in sync with whatever spectrum the provider currently uses.
    var N    = 20;
    var pts  = [];
    for (var i = 0; i <= N; i++) {
      var t = i / N;
      pts.push(this._gv._spectrumColor(t) + ' ' + Math.round(t * 100) + '%');
    }
    bar.style.background = 'linear-gradient(to right,' + pts.join(',') + ')';
  }
}
