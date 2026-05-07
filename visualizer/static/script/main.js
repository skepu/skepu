

var cy;
var expand_collapse;
var daglayout;
var splitInstance;

var previous_selected_item = undefined;
var opaqueNodes = [];
var baseZoom = 1;
var isZooming = false;

// Gantt pane state (referenced from inline script in main.html)
var _lastGraphNodes      = [];
var _ganttPaneOpen       = true;   // overridden by loadSettings if user closed it
var _ganttPaneRatio      = 0.25;   // fraction of window.innerHeight; overridden by loadSettings
var _ganttScale          = 1;      // x-axis scale multiplier; overridden by loadSettings
var _ganttSelectedNodeId = null;
var _legendOpen          = true;   // overridden by loadSettings

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
  var h = _ganttPaneRatio > 0
    ? Math.round(_ganttPaneRatio * window.innerHeight)
    : Math.round(window.innerHeight * 0.25);
  document.getElementById('gantt-pane').style.flex = '0 0 ' + h + 'px';
  _ganttPaneOpen = true;
  if (typeof syncExternalMenuIcons === 'function') syncExternalMenuIcons();
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

function ganttHighlight(nodeId) {
  _ganttSelectedNodeId = nodeId;
  var firstBar = null;
  document.querySelectorAll('.gantt-bar').forEach(function(b) {
    var sel = b.dataset.nodeId === nodeId;
    b.classList.toggle('gantt-selected', sel);
    if (sel && !firstBar) firstBar = b;
  });
  // Animate the gantt container to centre the first matching bar, like the code view does.
  if (firstBar) {
    var container = document.getElementById('gantt-svg-container');
    if (container) {
      var bx     = parseFloat(firstBar.getAttribute('x') || 0);
      var bw     = parseFloat(firstBar.getAttribute('width') || 0);
      var target = bx + bw / 2 - container.clientWidth / 2;
      $(container).stop(true).animate({ scrollLeft: Math.max(0, target) }, 500);
    }
  }
}

function renderGantt(cyNodes) {
  var container = document.getElementById('gantt-svg-container');
  if (!container) return;
  _lastGraphNodes = cyNodes;

  var timed = cyNodes.filter(function(n) {
    return n.data.start != null && n.data.end != null &&
      (n.data.type === 'skeleton_call' || n.data.type === 'external');
  });

  container.innerHTML = '';
  if (!timed.length) {
    var msg = document.createElement('p');
    msg.style.cssText = 'padding:0.5em;color:#888;font-family:monospace;font-size:0.8em';
    msg.textContent = 'No timed events to display.';
    container.appendChild(msg);
    return;
  }

  var minT = Infinity, maxT = -Infinity;
  timed.forEach(function(n) {
    var ivs = n.data.intervals || [[n.data.start, n.data.end]];
    ivs.forEach(function(iv) {
      minT = Math.min(minT, iv[0]);
      maxT = Math.max(maxT, iv[1]);
    });
  });
  var range = Math.max(maxT - minT, 1);

  // Collect distinct backends → one row each
  var bSet = {};
  timed.forEach(function(n) { bSet[n.data.backend || 'CPU'] = true; });
  var backends = Object.keys(bSet).sort();
  var bRow = {};
  backends.forEach(function(b, i) { bRow[b] = i; });

  // Layout constants.
  // SVG_W is the total SVG width: exactly the container width at scale=1 (no overflow),
  // and container * scale at higher values (horizontal scroll kicks in).
  // PAD_RIGHT is carved out of SVG_W for the "end of trace" label — the bar area BW
  // shrinks by that amount so nothing overflows at scale=1.
  var LW = 72, AH = 20, RH = 28, RP = 3, PAD_RIGHT = 90;
  var SVG_W = Math.max(container.clientWidth || 400, 200) * _ganttScale;
  var H     = backends.length * RH + AH;
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

  // Row backgrounds + labels
  backends.forEach(function(b, i) {
    var y = i * RH;
    svg.appendChild(svgEl('rect', { x: 0, y: y, width: SVG_W, height: RH, fill: i % 2 ? '#f8f8f8' : '#fff' }));
    var t = svgEl('text', { x: LW - 6, y: y + RH / 2 + 4, 'text-anchor': 'end', fill: '#555' });
    t.textContent = b;
    svg.appendChild(t);
  });

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
    var tl = svgEl('text', { x: xv, y: H - 4, 'text-anchor': anchor, fill: '#777' });
    tl.textContent = '+' + fmtDt(tv - minT);
    svg.appendChild(tl);
  }

  // Pattern colours (match graph node colours where possible)
  var patColors = {
    Map: '#3a8', Reduce: '#c44', MapReduce: '#d73',
    MapOverlap: '#55b', Scan: '#a4c', MapPool: '#36b',
  };

  var barIdx = 0;
  timed.forEach(function(n) {
    var d = n.data;
    var backend = d.backend || 'CPU';
    var ri = bRow[backend];
    var y  = ri * RH + RP, bh = RH - RP * 2;
    var fill = d.type === 'skeleton_call' ? (patColors[d.pattern] || '#778') : '#777';
    var ivs = d.intervals || [[d.start, d.end]];
    var total = ivs.length;

    ivs.forEach(function(iv, iidx) {
      var x1 = tx(iv[0]), x2 = tx(iv[1]);
      var bw = Math.max(2, x2 - x1);
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
      bar.style.cursor = 'pointer';
      if (d.id === _ganttSelectedNodeId) bar.classList.add('gantt-selected');

      var title = document.createElementNS(NS, 'title');
      var iterLabel = total > 1 ? ' [' + (iidx + 1) + '/' + total + ']' : '';
      title.textContent = d.label + (d.pattern ? ' [' + d.pattern + ']' : '') + iterLabel
        + '\n' + (d.backend || '') + '\n' + fmtDt(iv[1] - iv[0]);
      bar.appendChild(title);

      bar.addEventListener('click', function(ev) {
        ev.stopPropagation();
        ganttHighlight(d.id);
        if (cy) zoomToFit(d.id);
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

  container.appendChild(svg);
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
  if (previous_selected_item) previous_selected_item.removeClass("selected_gutter");
  var item = cy.$id(node_id);
  previous_selected_item = item;
  item.addClass("selected_gutter");
//  cy.fit(item);

  cy.animate({
    fit: {
      eles: item,
      padding: 40
    }
  }, { duration: 700 });
}

function zoomToFitGroup(node_ids)
{
  if (previous_selected_item) previous_selected_item.removeClass("selected_gutter");
  var pattern = "";
  for (node in node_ids)
  {
    pattern += "#" + node_ids[node];
    if (node < node_ids.length - 1) pattern += ",";
  }
  var items = cy.$(pattern);
  previous_selected_item = items;
  items.addClass("selected_gutter");
  cy.animate({
    fit: {
      eles: items,
      padding: 40
    }
  }, { duration: 700 });
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
    if (_ganttPaneOpen) renderGantt(nodes);

    // First find max and min value for time for correct colors
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
      if (!(node.data.type == "skeleton_call" || node.data.type == "external")) return; // transfers

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
              if (opaqueNodes.length == 0 || opaqueNodes.includes(ele.data("id")))
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
              var norm = findPropertyNorm("node-color-call", ele, minmax)
              const green = Math.floor(255 * (1 - norm));
              const red = Math.floor(255 * norm);
              return `rgb(${red},${green},0)`;
            }
          }
        },
        {
          selector: 'node[type="transfer"]',
          style: {
            'content': function (ele) { return ele.data("label") + transfer_mapper[ele.data("direction")]; },
            'text-valign' : 'center',
            'shape': 'diamond',
            'background-color' : 'purple'
          }
        },
        {
          selector: 'node[type="container_update"]',
          style: {
            'content': function (ele) { return ele.data("label") + " [" + ele.data("version") + "]"; },
            'text-valign' : 'center',
            'shape': 'barrel',
            'background-color' : 'lightgray'
          }
        },
        {
          selector: 'node[type="scalar"]',
          style: {
            'content': function (ele) { return ele.data("label") },
            'text-valign' : 'center',
            'shape': 'barrel',
            'background-color' : 'lightgray'
          }
        },
        {
          selector: 'node[type="skeleton_call"]',
          style: {
            'content': 'data(label)',
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
                console.log(ele.data("backend"));
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


              var norm = findPropertyNorm("node-color-call", ele, minmax)
              const green = Math.floor(255 * (1 - norm));
              const red = Math.floor(255 * norm);
              return `rgb(${red},${green},0)`;
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
            'content': 'data(label)',
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
              if (opaqueNodes.length == 0 || opaqueNodes.includes(ele.data("id")))
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
      animationDuration: 500,
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

    // Display the desiered data when clicking on a node
    cy.nodes().on('click', function(event)
    {
      var node = event.target;

      console.log(node);

      var label = node.data('label');
      var version = node.data('version');
      var id = node.data('id');

      var headings = {
        "skeleton_call" : "Skeleton Call",
        "allocation" : "Container Allocation",
        "deallocation" : "Container Deallocation",
        "transfer" : "Data Transfer",
        "container_update" : "Container Write",
        "region" : "Instrumented Region",
        "fusion" : "Suggested Skeleton Call Fusion",
        "external" : "External Data Access",
      }

      fetch(`/get_data?id=${encodeURIComponent(id)}`)
      .then(response => response.json())
      .then(data =>
      {
        document.getElementById('info-heading').textContent = headings[data["type"]] || data["type"];
        var info = "";
        for (var key in data)
        {
          var str = key.toString();
          if (data.hasOwnProperty(key) && !str.match("internal") && !str.match("type") && str !== "durations")
          {
            var value = data[key];
            if (key == "file") value = value.split('\\').pop().split('/').pop();
            info += '<p><strong>' + key + '</strong> ' + value + '</p>';
          }
        }
        $('#info').html(info);
        openInfoPane();
        ganttHighlight(id);

        // Update the opaque/transparent emphasis in the graph
        if (data["type"] == "container_update")
          opaqueNodes = data["internal"]["is_live"];
        else opaqueNodes = [];
        cy.style().update();

        highlightSourceLine(data["Line"], data["File"]);

        var plot_container = document.getElementById('plot');
        plot_container.innerHTML = "";
        if (data.durations && data.durations.length > 1)
        {
          var items = new vis.DataSet(data.durations);
          var options = {
            style: 'bar',
            height: '10em',
            barChart: { width: 50, align: 'center' },
            drawPoints: false,
            legend: false,
          };
          var timeline = new vis.Graph2d(plot_container, items, options);
          plot_container.style.display = "block";
        }
        else
        {
          plot_container.style.display = "none";
        }

      })
      .catch(error => console.error("Error fetching data:", error));
    });

    setTimeout(function()
    {
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
        if (el) el.innerHTML = "<span class='badge' onclick='zoomToFit(\"" + b.ids[0] + "\");ganttHighlight(\"" + b.ids[0] + "\")'>" + b.count + "</span>";
      }
    }, 200);

    let request_time = data["request_time"]; // seconds since epoch
    let response_time = data["response_time"]; // seconds since epoch
    let rendering_time = Date.now() / 1000; // milliseconds

  //  console.log(request_time, response_time, rendering_time);

    let modeling_duration = (response_time - request_time);
    let rendering_duration = (rendering_time - response_time);

    console.log(modeling_duration, rendering_duration);

    let total_duration = modeling_duration + rendering_duration;

    document.getElementById("load-time").innerHTML = Math.round(total_duration * 1000);
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
  const norm = (value - minmax[mode][0]) / (minmax[mode][1] - minmax[mode][0]);
  return norm;
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
    $(activePane || '#cpp_container').scrollTo(targets[0], 500, {over: {top: -5}});
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
  splitInstance = Split(['#left_pane', '#right_pane'], {
    sizes: [65, 35],
    onDragEnd: saveSettings,
  });
  applyStoredLayout();
  // Apply legend visibility now that #legend is in the DOM (_legendOpen was set by loadSettings).
  document.getElementById('legend').style.display = _legendOpen ? '' : 'none';
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

  // Gantt pane resizer (drag upward to grow)
  var ganttPane    = document.getElementById('gantt-pane');
  var ganttResizer = document.getElementById('gantt-resizer');
  var _gDragStartY, _gDragStartH;

  ganttResizer.addEventListener('mousedown', function(e) {
    _gDragStartY = e.clientY;
    _gDragStartH = ganttPane.offsetHeight;
    ganttPane.classList.add('resizing');
    document.addEventListener('mousemove', _onGanttDrag);
    document.addEventListener('mouseup',   _onGanttDragEnd);
    e.preventDefault();
  });

  function _onGanttDrag(e) {
    var newH = Math.max(40, _gDragStartH + (_gDragStartY - e.clientY));
    ganttPane.style.flex = '0 0 ' + newH + 'px';
  }

  function _onGanttDragEnd() {
    ganttPane.classList.remove('resizing');
    document.removeEventListener('mousemove', _onGanttDrag);
    document.removeEventListener('mouseup',   _onGanttDragEnd);
    if (ganttPane.offsetHeight > 10)
      _ganttPaneRatio = ganttPane.offsetHeight / window.innerHeight;
    renderGantt(_lastGraphNodes);
    saveSettings();
  }

  document.addEventListener("keydown",  function (event)
  {
    if (event.key === "e") {
      expand_collapse.expandRecursively(cy.$(":selected"));
    }
    else if (event.key === "c") {
      expand_collapse.collapseRecursively(cy.$(":selected"));
    }
    else if (event.key === "a") {
      fetchViewDataAndRender();
    }
    else if (event.key === "x" || event.key === "Escape") {
      closePopup();
    }
  });
}
