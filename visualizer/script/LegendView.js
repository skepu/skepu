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
//   hoverGraph       () -> cytoscape: hovering an entry marks the nodes or edges of that kind in the graph (class legend-match, graph_style.js).
//                    Omit it, or set HIGHLIGHT_GRAPH_ON_LEGEND_HOVER to false, for no mark.
//   clearHighlight   () -> (): clears the page's own highlight (a node selection); called when a click locks the mark of an entry
//   onSelect         (callback) -> (): the page calls the callback when it highlights something; that ends a locked mark
//   onSpectrumClick  (x, y) -> (): click on the colour bar; opens the spectrum menu at the click
//   onColorByClick   (x, y) -> (): right click on the colour bar; opens the colour-by menu
//   labels      { key: text }: another name for an entry (the key is its data-legend-key: 'container_update', ...)
//   extraNodes  [{ key, label, icon(dark) -> svg markup, present() -> bool }]: kinds of nodes that only the page has
//   typeKeys    { node type: key }: the entry of such a kind of node, for highlightFor

import { COLORS, regionColor } from './graph_style.js';
import { BACKEND_ORDER, PATTERN_ORDER, basePattern } from './TraceDataProvider.js';

/** Colour-by names, shorter than the entries of Nodes > Color. */
export const COLOR_BY_LABELS = {
  'duration-total': 'total time', 'elements': 'elements', 'duration-per-element': 'time/element', 'iterations': 'iterations',
  'duration-cv': 'variance', 'imbalance': 'imbalance', 'comm-bytes': 'busiest rank bytes', 'dag-depth': 'DAG depth',
  'total-order': 'sequence', 'pattern': 'pattern', 'backend': 'backend',
};

/** false disables the hover mark for every graph. */
export const HIGHLIGHT_GRAPH_ON_LEGEND_HOVER = true;

export class LegendView {
  // legendEl  — the #legend container element (safe to pass null; render() is a no-op)
  // graphView — the associated GraphView instance (used for hasElements + spectrum colors)
  constructor(legendEl, graphView, options) {
    this._legendEl  = legendEl;
    this._gv        = graphView;
    this._options   = options || {};
    this.settings   = {};   // set by graph.js to the shared settings reference
    if (legendEl && this._options.onColorByClick) {
      var colorPage = this;
      legendEl.addEventListener('contextmenu', function(event) {
        if (event.target.closest && event.target.closest('.legend-spectrum-section')) {
          event.preventDefault();
          colorPage._options.onColorByClick(event.clientX, event.clientY);
        }
      });
    }
    if (legendEl && this._options.onSpectrumClick) {
      var page = this;
      legendEl.addEventListener('click', function(event) {
        if (event.target.closest && event.target.closest('.legend-spectrum-section')) {
          event.stopPropagation();           // the click must not also close the menu
          page._options.onSpectrumClick(event.clientX, event.clientY);
        }
      });
    }
    this._hoverKey   = null;
    this._lockedKey  = null; // locked entry; a click toggles it
    this._hoverFrame = 0;
    // render() recreates the entries: one listener on the panel
    if (legendEl && this._options.hoverGraph && HIGHLIGHT_GRAPH_ON_LEGEND_HOVER) {
      var self = this;
      legendEl.classList.add('legend-interactive');      // main.css
      var keyOf = function(event) { var item = event.target.closest && event.target.closest('[data-legend-key]'); return item ? item : null; };
      legendEl.addEventListener('mouseover', function(event) {
        var item = keyOf(event);
        self._hoverGraph(item ? item.getAttribute('data-legend-key') : null);
      });
      legendEl.addEventListener('mouseleave', function() { self._hoverGraph(null); });
      legendEl.addEventListener('click', function(event) {
        var item = keyOf(event);
        if (item) self._lock(item.getAttribute('data-legend-key'));
      });
      if (this._options.onSelect) this._options.onSelect(function() { self._unlock(); });
    }
  }

  // A lock is not moved by hover
  _hoverGraph(key) {
    if (key === this._hoverKey) return;
    this._hoverKey = key;
    this._markSoon();
  }

  // Toggles the lock on an entry
  _lock(key) {
    this._lockedKey = this._lockedKey === key ? null : key;
    if (this._lockedKey && this._options.clearHighlight) this._options.clearHighlight();      // the entry mark and a selection are exclusive
    this._showLock();
    this._markSoon();
  }

  _unlock() {
    if (!this._lockedKey) return;
    this._lockedKey = null;
    this._showLock();
    this._markSoon();
  }

  // Any graph interaction except pan, zoom and hover ends the lock: it is a mark to look at while panning or zooming
  _bindRelease(cy) {
    if (this._releaseCy === cy) return;
    var self = this;
    if (this._releaseCy) this._releaseCy.off('tap cxttapstart grab boxstart', this._release);
    this._releaseCy = cy;
    this._release = function() { self._unlock(); };
    cy.on('tap cxttapstart grab boxstart', this._release);
  }

  // Classes for the locked entry and the panel (main.css)
  _showLock() {
    var self = this;
    this._legendEl.classList.toggle('legend-locked', !!this._lockedKey);
    this._legendEl.querySelectorAll('.legend-item[data-legend-key]').forEach(function(el) {
      el.classList.toggle('legend-item-locked', el.getAttribute('data-legend-key') === self._lockedKey);
    });
  }

  // Marks the elements of the locked entry, else the hovered one, on the next frame: at most one style pass per frame while the pointer crosses the legend
  _markSoon() {
    var self = this;
    if (this._hoverFrame) cancelAnimationFrame(this._hoverFrame);
    this._hoverFrame = requestAnimationFrame(function() {
      self._hoverFrame = 0;
      self._markGraph();
    });
  }

  _markGraph() {
    var self = this, key = this._lockedKey || this._hoverKey;
    var cy = this._options.hoverGraph();
    if (!cy) return;
    this._bindRelease(cy);
    var matches = key ? cy.elements().filter(function(ele) {
      return !ele.hasClass('layout-phantom') && ele.visible() && (self.keyFor(ele) === key || self.swatchKeyFor(ele) === key);
    }) : null;
    cy.batch(function() {
      cy.elements('.legend-match').removeClass('legend-match');
      if (matches) matches.addClass('legend-match');
    });
    var entry = key && this._legendEl.querySelector('[data-legend-key="' + key + '"]');
    if (entry) entry.title = matches.length + ' in the graph. Click to ' + (key === this._lockedKey ? 'let go of the mark.' : 'keep the mark.');
  }

  /** Called when the graph is rebuilt. */
  clearGraphMark() {
    this._hoverKey = null;
    this._lockedKey = null;
    if (this._hoverFrame) { cancelAnimationFrame(this._hoverFrame); this._hoverFrame = 0; }
    var cy = this._options.hoverGraph && this._options.hoverGraph();
    if (cy) cy.elements('.legend-match').removeClass('legend-match');
    if (this._legendEl) this._showLock();
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
      nodeEntries.push([nodeIcon('<polygon points="10,1 19,10 10,19 1,10" fill="' + midClr + '" stroke="' + midDark + '" stroke-width="1.5"/>'),
       'Transfer', 'transfer']);
    if (inGraph('node[type="compile"]'))
      nodeEntries.push([nodeIcon('<polygon points="5,2 15,2 19,6 19,14 15,18 5,18 1,14 1,6" fill="' + COLORS.compile.fill + '" stroke="' + COLORS.compile.stroke + '" stroke-width="1.5"/>'),
       'Kernel Build', 'compile']);
    if (inGraph('node[type="communication"]'))
      nodeEntries.push([nodeIcon('<polygon points="5,2 15,2 19,10 15,18 5,18 1,10" fill="' + COLORS.communication.fill + '" stroke="' + COLORS.communication.stroke + '" stroke-width="1.5"/>'),
       'Communication', 'communication']);
    if (inGraph('node[type="container_update"]'))
      nodeEntries.push([nodeIcon('<rect x="3" y="3" width="14" height="14" fill="' + ctFill + '" stroke="' + ctStroke + '" stroke-width="2"/>'),
       'Data Update', 'container_update']);
    if (inGraph('node[type="allocation"][!is_prng]'))
      nodeEntries.push([nodeIcon('<polygon points="' + STAR + '" fill="' + allocFill + '" stroke="' + allocStr + '" stroke-width="1"/>'),
       'Allocation', 'allocation']);
    if (inGraph('node[type="deallocation"]'))
      nodeEntries.push([nodeIcon('<polygon points="' + STAR + '" fill="' + dealFill + '" stroke="' + dealStr + '" stroke-width="1"/>'),
       'Deallocation', 'deallocation']);
    if (inGraph('node[type="snapshot"]'))
      nodeEntries.push([nodeIcon('<polygon points="' + STAR + '" fill="' + COLORS.snapshot.fill + '" stroke="' + COLORS.snapshot.stroke + '" stroke-width="1"/>'),
       'Snapshot', 'snapshot']);
    if (this.settings.get('show_regions') && inGraph('node[type="region"]'))
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
    if (inGraph('edge[type="setup-dep"]'))
      edgeEntries.push([edgeIcon('10 3 2 3', COLORS.compile.stroke), 'Setup', 'setup-dep']);
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

    // A measure: the spectrum with its ends. Pattern or backend: one block per category present, no ends (not a scale).
    var mode = this.settings.get('node-color-call');
    var classBased = mode === 'pattern' || mode === 'backend';
    var swatches = '', classColors = null;
    if (classBased) {
      var provider = gv._provider;
      var order = mode === 'pattern' ? PATTERN_ORDER : BACKEND_ORDER;
      var names = gv.callCategories(mode).slice().sort(function(a, b) {        // in the order of the spectrum, the others last
        var ia = order.indexOf(a), ib = order.indexOf(b);
        return (ia < 0 ? order.length : ia) - (ib < 0 ? order.length : ib);
      });
      classColors = names.map(function(name) { return mode === 'pattern' ? provider.patternColor(name) : provider.backendColor(name); });
      swatches = section(mode === 'pattern' ? 'Skeletons' : 'Backends', names.map(function(name, i) {
        var line = mode === 'pattern' ? provider.patternColor(name, true) : provider.backendColor(name, true);
        return ['<i class="legend-swatch" style="background:' + classColors[i] + ';border-color:' + line + '"></i>', name, 'swatch:' + name];
      }));
    }
    this._classColors = classColors && classColors.length ? classColors : null;
    var ends = function(word) { return classBased ? '' : '<span class="legend-spec-end" style="color:' + textClr + '">' + word + '</span>'; };
    var colourSection = classBased && !this._classColors ? '' :
      '<div class="legend-section legend-spectrum-section' + (this._options.onSpectrumClick ? ' legend-clickable" title="Click: another color spectrum.' + (this._options.onColorByClick ? ' Right click: color by something else.' : '') : '') + '">' +
      '<div class="legend-section-title">Color by ' + (COLOR_BY_LABELS[mode] || 'measure') + '</div>' +
      '<div class="legend-spectrum-row">' + ends('min') + '<span id="color-bar"></span>' + ends('max') + '</div></div>' + swatches;

    // The program tree mode has no dependences, so it has no Dependences section.
    var dependences = this.settings.get('graph-mode') === 'program-tree' ? '' : section('Dependences', edgeEntries);
    this._legendEl.innerHTML = colourSection + section('Nodes', nodeEntries) + dependences;

    // Populate the freshly-inserted #color-bar gradient.
    this._updateColorBar();

    // A lock survives if the graph still has its entry
    if (this._lockedKey && !this._legendEl.querySelector('[data-legend-key="' + this._lockedKey + '"]')) this._lockedKey = null;
    if (this._legendEl.classList.contains('legend-interactive')) { this._showLock(); this._markSoon(); }
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
        'compile':          'compile',
        'communication':    'communication',
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
    if (d.type === 'setup-dep')     return 'setup-dep';
    if (d.access_mode === 'elwise') return 'elwise';
    if (d.access_mode === 'scalar') return 'scalar';
    if (d.type === 'forward-dep')   return 'proxy';
    return null;
  }

  /** Mark the entry for a node or an edge (and, for a call, the swatch of the pattern or backend that it is coloured by). */
  highlightFor(ele) {
    this.highlightKey(this.keyFor(ele));
    var swatch = this.swatchKeyFor(ele);
    var entry = swatch && this._legendEl && this._legendEl.querySelector('[data-legend-key="' + swatch + '"]');
    if (entry) entry.classList.add('legend-item-hovered');
  }

  /** The data-legend-key of the swatch (the pattern or the backend) that a call is coloured by, or null. */
  swatchKeyFor(ele) {
    var mode = this.settings.get('node-color-call');
    if (ele.data('type') !== 'skeleton_call' || (mode !== 'pattern' && mode !== 'backend')) return null;
    return 'swatch:' + (mode === 'pattern' ? basePattern(ele.data('pattern')) : ele.data('backend'));
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
    if (this._classColors) {
      var n = this._classColors.length, blocks = [];
      this._classColors.forEach(function(c, k) { blocks.push(c + ' ' + (100 * k / n) + '%', c + ' ' + (100 * (k + 1) / n) + '%'); });
      bar.style.background = 'linear-gradient(to right,' + blocks.join(',') + ')';
      return;
    }
    var N    = 20;
    var pts  = [];
    for (var i = 0; i <= N; i++) {
      var t = i / N;
      pts.push(this._gv._spectrumColor(t) + ' ' + Math.round(t * 100) + '%');
    }
    bar.style.background = 'linear-gradient(to right,' + pts.join(',') + ')';
  }
}
