// graph_style.js — the Cytoscape style of the SkeVU graph, shared by GraphView and the designer.
//
// buildGraphStyle(view, minmax) returns the style array. `view` is a GraphView, or anything with the same members:
//   settings, _darkMode, _highlightColor(), _spectrumColor(t), _spectrumColorDark(t), _normProp(key, ele, minmax),
//   _opaqueNodes (ids), _groupedEdgeReps (a Set), _provider (backendColor, patternColor).

export function toSubscript(val) {
  return ('' + val).replace(/[0-9]/g, function (d) { return '₀₁₂₃₄₅₆₇₈₉'[d]; });
}

/** The colours of the nodes and edges that do not come from a spectrum: the graph and its legend (LegendView) have them from here. */
export const COLORS = {
  data:         { fill: 'lightgray', stroke: '#979797' },       // a data update, a scalar
  allocation:   { fill: 'white',     stroke: '#333' },
  deallocation: { fill: 'black',     stroke: '#ddd' },
  transfer:     { fill: 'purple',    stroke: '#6a006a' },
  fusion:       { fill: '#88f' },
  prng:         { fill: ['#e2d2f2', '#4b3a63'], stroke: ['#8e44ad', '#bb8fce'] },     // [light, dark]: the allocation of a random stream (a violet hexagon)
  edge:         ['#333', '#ccc'],                               // [light, dark]
  antiDep:      'red',
  alias:        ['#8e44ad', '#bb8fce'],
};

/** The colour of a region of the depth `depth` (0: the outermost): lighter blue in the light mode, brown in the dark. */
export function regionColor(depth, dark) {
  return dark
    ? 'rgb(' + Math.min(100, 42 + depth * 15) + ',' + Math.min(74, 31 + depth * 10) + ',' + Math.min(42, 16 + depth * 6) + ')'
    : 'rgb(' + Math.max(20, 225 - depth * 34) + ',' + Math.max(185, 240 - depth * 9) + ',255)';
}

export const NODE_SIZE_SCALE = 100;
export const NODE_SIZE_MIN   = 15;

export function buildGraphStyle(self, minmax) {
  var node_labels  = self.settings['node-labels'];
  var edge_labels  = self.settings['edge-labels'];
  var transfer_map = { 'host-to-device': '↑', 'device-to-host': '↓' };

  function dark()     { return self._darkMode; }
  function hcol()     { return self._highlightColor(); }
  function sc(t)      { return self._spectrumColor(t); }
  function scd(t)     { return self._spectrumColorDark(t); }
  function norm(k, e) { return self._normProp(k, e, minmax); }

  return [
    {
      selector: 'node',
      style: {
        'color':              function() { return dark() ? 'white' : 'black'; },
        'text-outline-color': function() { return dark() ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.5)'; },
        'text-outline-width': 0.8,
        'font-size': '0.6em',
        'min-zoomed-font-size': '0.4em',
        'opacity': function(ele) {
          if (self._opaqueNodes.length === 0) return 1;
          return self._opaqueNodes.includes(ele.data('id')) ? 1 : 0.2;
        },
      }
    },
    {
      selector: 'node[type="allocation"]',
      style: {
        'content':          function(ele) { return node_labels ? ele.data('label') : ''; },
        'text-valign':      'center',
        'shape':            'star',
        'border-width':     1,
        'background-color': COLORS.allocation.fill,
        'border-color':     COLORS.allocation.stroke,
      }
    },
    {
      selector: 'node[type="allocation"][?is_prng]',
      style: {
        'shape':            'hexagon',
        'width':            52, 'height': 26,
        'border-width':     3,
        'background-color': function() { return COLORS.prng.fill[dark() ? 1 : 0]; },
        'border-color':     function() { return COLORS.prng.stroke[dark() ? 1 : 0]; },
      }
    },
    {
      selector: 'node[type="deallocation"]',
      style: {
        'content':          function(ele) { return node_labels ? ele.data('label') : ''; },
        'text-valign':      'center',
        'shape':            'star',
        'border-width':     1,
        'background-color': COLORS.deallocation.fill,
        'border-color':     COLORS.deallocation.stroke,
      }
    },
    {
      selector: 'node[type="external"]',
      style: {
        'content':          function(ele) { return node_labels ? ele.data('label') : ''; },
        'text-valign':      'center',
        'shape':            'octagon',
        'background-color': function(ele) { return sc(norm('node-color-call', ele)); },
        'border-width':     3,
        'border-color':     function(ele) { return scd(norm('node-color-call', ele)); },
        'width':  function(ele) { return NODE_SIZE_SCALE * Math.sqrt(norm('node-size-call', ele)) + NODE_SIZE_MIN; },
        'height': function(ele) { return NODE_SIZE_SCALE * Math.sqrt(norm('node-size-call', ele)) + NODE_SIZE_MIN; },
      }
    },
    {
      selector: 'node[type="transfer"]',
      style: {
        'content':          function(ele) { return node_labels ? ele.data('label') + (transfer_map[ele.data('direction')] || '') : ''; },
        'text-valign':      'center',
        'shape':            'diamond',
        'background-color': COLORS.transfer.fill,
        'border-width':     3,
        'border-color':     COLORS.transfer.stroke,
      }
    },
    {
      selector: 'node[type="container_update"], node[type="scalar"]',
      style: {
        'content': function(ele) {
          if (!node_labels) return '';
          return ele.data('iteration_count') > 1
            ? ele.data('label') + ' x' + ele.data('iteration_count')
            : ele.data('label') + (ele.data('version_count') > 1 ? toSubscript(ele.data('version')) : '');   // (no subscript for the only version)
        },
        'text-valign':      'center',
        'shape':            'rectangle',
        'height':           20, 'width': 45,
        'background-color': COLORS.data.fill,
        'border-width':     3,
        'border-color':     COLORS.data.stroke,
      }
    },
    {
      selector: 'node[type="container_update"][has_snapshot]',
      style: {
        'background-color': function() { return dark() ? '#3a2e00' : '#fff0a0'; },
        'border-color':     function() { return dark() ? '#c8a000' : '#b8860b'; },
      }
    },
    {
      selector: 'node[type="scalar"]',
      style: {
        'content':       function(ele) { return node_labels ? ele.data('label') : ''; },
        'text-margin-y': 2,
        'shape':         'triangle',
        'height':        20, 'width': 25,
      }
    },
    {
      selector: 'node[type="skeleton_call"]',
      style: {
        'content': function(ele) {
          if (!node_labels) return '';
          return ele.data('iteration_count') > 1
            ? ele.data('label') + ' x' + ele.data('iteration_count')
            : ele.data('label');
        },
        'shape':      'ellipse',
        'text-valign': 'center',
        'background-color': function(ele) {
          var mode = self.settings['node-color-call'];
          if (mode === 'backend') {
            return self._provider.backendColor(ele.data('backend')) || sc(norm('node-color-call', ele));
          }
          if (mode === 'pattern') {
            return self._provider.patternColor(ele.data('pattern')) || sc(norm('node-color-call', ele));
          }
          return sc(norm('node-color-call', ele));
        },
        'border-width': 3,
        'border-color': function(ele) {
          var mode = self.settings['node-color-call'];
          if (mode === 'backend') {
            return self._provider.backendColor(ele.data('backend'), true) || '#555';
          }
          if (mode === 'pattern') {
            return self._provider.patternColor(ele.data('pattern'), true) || '#555';
          }
          return scd(norm('node-color-call', ele));
        },
        'width':  function(ele) { return NODE_SIZE_SCALE * Math.sqrt(norm('node-size-call', ele)) + NODE_SIZE_MIN; },
        'height': function(ele) { return NODE_SIZE_SCALE * Math.sqrt(norm('node-size-call', ele)) + NODE_SIZE_MIN; },
      }
    },
    {
      selector: 'node[type="region"]',
      style: {
        'content': function(ele) {
          if (!node_labels) return '';
          return ele.data('iteration_count') > 1
            ? ele.data('label') + ' x' + ele.data('iteration_count')
            : ele.data('label');
        },
        'border-color':     function() { return dark() ? '#170D00' : '#006770'; },
        'border-width':     0,
        'background-color': function(ele) {
          var d = ele.data('region_depth') || 0;
          return regionColor(d, dark());
        },
      }
    },
    {
      selector: 'node[type="fusion"]',
      style: {
        'content':          function(ele) { return node_labels ? ele.data('label') : ''; },
        'background-color': COLORS.fusion.fill,
      }
    },
    {
      selector: 'node[type="fusion"][collapsed]',
      style: {
        'content':          function(ele) { return node_labels ? ele.data('label') : ''; },
        'text-halign':      'center', 'text-valign': 'center',
        'background-color': COLORS.fusion.fill,
      }
    },
    {
      selector: 'node[type="region"][collapsed], node[type="tree-region"]',
      style: {
        'content': function(ele) {
          if (!node_labels) return '';
          return ele.data('iteration_count') > 1
            ? ele.data('label') + ' x' + ele.data('iteration_count')
            : ele.data('label');
        },
        'shape':            'round-rectangle',
        'text-valign':      'center',
        'border-width':     3,
        'height':           40, 'width': 64,
        'border-color':     function() { return dark() ? '#733103' : '#006770'; },
        'background-color': function() { return dark() ? '#3B1A02' : 'skyblue'; },
      }
    },
    {
      selector: 'node.layout-phantom',
      style: {
        'background-opacity': 0, 'border-width': 0,
        'label': '', 'events': 'no', 'padding': 0,
      }
    },
    {
      selector: '.selected_gutter',
      style: { 'border-width': 5, 'border-color': function() { return hcol(); } }
    },
    {
      selector: 'edge.grouped-hidden',
      style: { 'display': 'none' }
    },
    {
      selector: 'edge',
      style: {
        'content': function(ele) {
          if (self._groupedEdgeReps.has(ele.id())) return ele.data('label') || '';
          return edge_labels ? (ele.data('label') || '') : '';
        },
        'text-outline-width': 0.8,
        'text-outline-color': function() { return dark() ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.5)'; },
        'color':  function() { return dark() ? 'white' : 'black'; },
        'font-size': '0.5em',
        'width': function(ele) {
          var m = self.settings['edge-width'];
          if (m === 'critical-path' || m === 'cross-backend') {
            return (m === 'critical-path' ? ele.data('is_critical_path') : ele.data('cross_backend')) ? 5 : 1;
          }
          if (m === 'iterations') {
            var mm2 = minmax['iterations'];
            if (!mm2 || mm2.max === mm2.min) return 1.5;
            return 1 + Math.max(0, Math.min(1, (ele.data('iteration_count') - mm2.min) / (mm2.max - mm2.min))) * 5;
          }
          if (m === 'order-span') {
            var mm3 = minmax['order-span'];
            if (!mm3 || mm3.max === mm3.min) return 1.5;
            return 1 + Math.max(0, Math.min(1, (ele.data('order_span') - mm3.min) / (mm3.max - mm3.min))) * 5;
          }
          if (m === 'elements') {
            var wels = ele.source().data('elements');
            if (!wels) return 1.5;
            var mm4 = minmax['elements'];
            if (!mm4 || mm4.max === mm4.min) return 1.5;
            return 1 + Math.max(0, Math.min(1, (wels.reduce(function(a,b){return a*b;},1) - mm4.min) / (mm4.max - mm4.min))) * 5;
          }
          return 1.5;
        },
        'target-arrow-shape': 'triangle',
        'line-color':         function() { return COLORS.edge[dark() ? 1 : 0]; },
        'target-arrow-color': function() { return COLORS.edge[dark() ? 1 : 0]; },
        'curve-style':  'bezier',
        'taxi-direction': 'downward',
        'opacity': function(ele) {
          // opaqueNodes dimming takes priority over all settings-based modes.
          if (self._opaqueNodes.length > 0 &&
              !self._opaqueNodes.includes(ele.data('id'))) return 0.2;
          var m = self.settings['edge-opacity'];
          if (m === 'critical-path' || m === 'cross-backend') {
            return (m === 'critical-path' ? ele.data('is_critical_path') : ele.data('cross_backend')) ? 1 : 0.15;
          }
          if (m === 'iterations') {
            var mm2 = minmax['iterations'];
            if (!mm2 || mm2.max === mm2.min) return 1;
            return 0.15 + Math.max(0, Math.min(1, (ele.data('iteration_count') - mm2.min) / (mm2.max - mm2.min))) * 0.85;
          }
          if (m === 'order-span') {
            var mm3 = minmax['order-span'];
            if (!mm3 || mm3.max === mm3.min) return 1;
            return 0.15 + Math.max(0, Math.min(1, (ele.data('order_span') - mm3.min) / (mm3.max - mm3.min))) * 0.85;
          }
          if (m === 'elements') {
            var oels = ele.source().data('elements');
            if (!oels) return 1;
            var mm4 = minmax['elements'];
            if (!mm4 || mm4.max === mm4.min) return 1;
            return 0.15 + Math.max(0, Math.min(1, (oels.reduce(function(a,b){return a*b;},1) - mm4.min) / (mm4.max - mm4.min))) * 0.85;
          }
          return 1;
        },
      }
    },
    {
      selector: 'edge[type="anti-dep"]',
      style: { 'line-color': COLORS.antiDep, 'target-arrow-color': COLORS.antiDep }
    },
    {
      selector: 'edge[access_mode="elwise"]',
      style: { 'line-style': 'dashed' }
    },
    {
      selector: 'edge[access_mode="scalar"]',
      style: { 'line-style': 'dotted' }
    },
    {
      selector: 'edge[type="virtual-alias"]',
      style: {
        'line-color':         function() { return COLORS.alias[dark() ? 1 : 0]; },
        'line-style':         'dashed',
        'line-dash-pattern':  [8, 4],
        'target-arrow-shape': 'none',
        'source-arrow-shape': 'none',
        'width':              1.5,
        'opacity':            0.65,
      }
    },
    {
      selector: 'edge[type="prng-dep"]',
      style: {
        'curve-style':       'segments',
        'segment-weights':   '0.2 0.4 0.6 0.8',
        'segment-distances': '3 -3 3 -3',
        'opacity':           0.65,
      }
    },
    {
      selector: ':selected, :active',
      style: { 'overlay-opacity': 0 }
    },
    {
      selector: 'edge.edge-highlighted',
      style: {
        'line-color':         function() { return hcol(); },
        'target-arrow-color': function() { return hcol(); },
        'width': 3,
      }
    },
    {
      selector: 'edge.edge-bidir',
      style: {
        'source-arrow-shape': 'triangle',
        'source-arrow-color': function() { return COLORS.edge[dark() ? 1 : 0]; },
      }
    },
    {
      selector: 'edge.edge-bidir[type="anti-dep"]',
      style: { 'source-arrow-color': 'red' }
    },
    {
      selector: 'edge.edge-highlighted[type="anti-dep"]',
      style: { 'line-color': 'red', 'target-arrow-color': 'red', 'width': 3 }
    },
    {
      selector: 'node.cy-hovered',
      style: { 'border-width': 6 }
    },
    {
      selector: 'edge.cy-hovered',
      style: { 'width': 4 }
    },
  ];
}
