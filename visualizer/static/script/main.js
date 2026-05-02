

var cy;
var expand_collapse;
var daglayout;
var splitInstance;

var previous_selected_item = undefined;
var opaqueNodes = [];
var baseZoom = 1;
var isZooming = false;


function fetchViewDataAndRender()
{
  document.getElementById('cy').style.height = "";
  document.getElementById('cy').innerHTML = "";
  let view_mode = "graph"; //document.getElementById("view-mode").value;
  if (view_mode == "graph")
  {
    document.getElementById("cy").style.display = "block";
    document.getElementById('cy').style.height = document.getElementById('cy').offsetHeight + "px";
    document.getElementById("timeline").style.display = "none";
    return fetchGraphDataAndRender();
  }
  else if (view_mode == "timeline")
  {
    document.getElementById("cy").style.display = "none";
    document.getElementById("timeline").style.display = "block";
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
        if (el) el.innerHTML = "<span class='badge' onclick='zoomToFit(\"" + b.ids[0] + "\")'>" + b.count + "</span>";
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

function settingChanged()
{

}

function recenterGraph()
{
  var layout = cy.layout(daglayout);
  layout.run();
}

function toggleLegend()
{
  var status = document.getElementById('legend').style.display;
  var newStatus = status != "none" ? "none" : "block";
  document.getElementById('legend').style.display = newStatus;
}




function load()
{
  splitInstance = Split(['#left_pane', '#right_pane'], { sizes: [65, 35], onDragEnd: saveSettings });
  applyStoredLayout();

  hljs.highlightAll();
  hljs.initLineNumbersOnLoad();

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
