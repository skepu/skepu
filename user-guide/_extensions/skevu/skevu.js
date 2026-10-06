/*
 * skevu.js — runtime of the trace-embed and trace-wasm shortcodes (skevu.lua).
 *
 * skevu.mount(panelId, config) turns the panel DOM emitted by the shortcode into
 * SkeVU views (graph, source, timeline) on top of the visualizer served under
 * config.vizBase.  The trace is either a fixed example of the visualizer
 * (config.example, config.backend) or the output of a WebAssembly program that
 * the reader runs (config.wasm).
 */
(function () {
'use strict';

// ── Script loading ───────────────────────────────────────────────────────────

function loadScript(src) {
  return new Promise(function (resolve, reject) {
    var s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = function () { reject(new Error('Could not load ' + src)); };
    document.head.appendChild(s);
  });
}

// Visualizer dependencies are loaded once per page, however many panels there are.
var _deps = null;

function loadDeps(vizBase) {
  if (_deps) return _deps;
  _deps = (async function () {
    var libs = [   // classic scripts, in dependency order
      'cytoscape/cytoscape.min.js', 'dagre/dagre.min.js', 'cytoscape-dagre/cytoscape-dagre.js',
      'cytoscape.js-elk-2.3.0/dist/elk.bundled.js', 'jquery/jquery-3.7.1.min.js',
      'jquery-scrollto/jquery.scrollTo.min.js', 'highlight/highlight.min.js',
      'highlight-line-numbers/highlightjs-line-numbers.min.js',
    ];
    for (var lib of libs) await loadScript(vizBase + 'static/' + lib);

    var link = document.createElement('link');
    link.id = 'te-hljs-theme';
    link.rel = 'stylesheet';
    link.href = vizBase + 'static/highlight/styles/default.css';
    document.head.appendChild(link);

    var [gv, tdp, scv, tv] = await Promise.all(
      ['GraphView', 'TraceDataProvider', 'SourceCodeView', 'TimelineView']
        .map(function (m) { return import(vizBase + 'script/' + m + '.js'); }));
    scv.SourceCodeView.BADGE_PLACEMENT_DELAY_MS = 200;
    scv.SourceCodeView.ANIM_SCROLL_MS = 400;
    return {
      GraphView: gv.GraphView, TraceDataProvider: tdp.TraceDataProvider,
      SourceCodeView: scv.SourceCodeView, TimelineView: tv.TimelineView,
    };
  })();
  return _deps;
}

// ── WorkerSession: promise-based RPC to the visualizer's trace worker ────────
// Each panel has its own, so their loaded traces are independent.

class WorkerSession {
  constructor(workerUrl) {
    this._seq = 0;
    this._pending = new Map();
    this._worker = new Worker(workerUrl, { type: 'module' });
    this._worker.addEventListener('message', (ev) => {
      var { id, type, error } = ev.data;
      var h = this._pending.get(id);
      if (!h) return;
      this._pending.delete(id);
      type === 'error' ? h.reject(new Error(error)) : h.resolve(ev.data);
    });
    this._worker.addEventListener('error', (err) => {
      for (var h of this._pending.values()) h.reject(err);
      this._pending.clear();
    });
  }
  call(type, payload) {
    return new Promise((resolve, reject) => {
      var id = String(++this._seq);
      this._pending.set(id, { resolve, reject });
      this._worker.postMessage(Object.assign({}, payload, { id, type }));
    });
  }
  fetchGraph(params) { return this.call('graph', params); }
}

// ── Small DOM helpers ────────────────────────────────────────────────────────

function $(id) { return document.getElementById(id); }

function fmtDur(us) {
  if (us == null || us < 0) return null;
  if (us >= 1e6)  return (us / 1e6).toFixed(1)  + ' s';
  if (us >= 1000) return (us / 1000).toFixed(1) + ' ms';
  return us.toFixed(0) + ' μs';
}

function fmtMs(ms) {
  if (ms == null) return '—';
  return ms < 1000 ? ms + ' ms' : (ms / 1000).toFixed(2) + ' s';
}

function setLoading(pid, active) {
  var el = $(pid + '-loading');
  if (el) el.style.display = active ? 'flex' : 'none';
}

function setError(pid, msg) {
  setLoading(pid, false);
  var el = $(pid + '-error');
  if (el) { el.textContent = '⚠ ' + msg; el.style.display = 'block'; }
}

function updateStats(pid, stats) {
  var el = $(pid + '-stats');
  if (!el) return;
  var parts = ['Nodes <b>' + stats.nodeCount + '</b>', 'Edges <b>' + stats.edgeCount + '</b>'];
  if (stats.timelineNodes && stats.timelineNodes.length)
    parts.push('Timeline <b>' + stats.timelineNodes.length + '</b>');
  parts.push('Model <b>' + fmtMs(stats.modelingMs) + '</b>', 'Layout <b>' + fmtMs(stats.renderingMs) + '</b>');
  el.innerHTML = parts.map(function (p) { return '<span>' + p + '</span>'; }).join('');
}

function showInfo(pid, data) {
  var el = $(pid + '-info');
  if (!el) return;
  var rows = [];
  if (data.label)                                  rows.push(['Name', data.label]);
  if (data.type)                                   rows.push(['Type', data.type]);
  if (data.backend)                                rows.push(['Backend', data.backend]);
  if (data.pattern)                                rows.push(['Pattern', data.pattern]);
  if (data.duration != null && data.duration >= 0) rows.push(['Time', fmtDur(data.duration) || '0 μs']);
  if (data.elements && data.elements.length)       rows.push(['Elements', data.elements.join('×')]);
  if (data.file && data.line != null && data.line !== -1)
    rows.push(['Source', data.file.split('/').pop().split('\\').pop() + ':' + data.line]);
  if (!rows.length) {
    el.className = 'te-info-box te-info-empty';
    el.textContent = 'No details available';
    return;
  }
  el.className = 'te-info-box';
  var tbl = document.createElement('table');
  rows.forEach(function (r) {
    var tr = tbl.insertRow();
    var key = tr.insertCell();
    key.className = 'te-info-key';
    key.textContent = r[0];
    tr.insertCell().textContent = r[1];
  });
  el.innerHTML = '';
  el.appendChild(tbl);
}

function clearInfo(pid) {
  var el = $(pid + '-info');
  if (el) { el.className = 'te-info-box te-info-empty'; el.textContent = 'Click a node to see details'; }
}

// ── Interactive controls ─────────────────────────────────────────────────────

var CONTROL_DEFS = {
  'regions':          { type: 'checkbox', key: 'show_regions',       label: 'Regions'       },
  'collapse-iter':    { type: 'checkbox', key: 'collapse_iteration', label: 'Collapse iter' },
  'element-accesses': { type: 'checkbox', key: 'element-accesses',   label: 'Data nodes'    },
  'anti-deps':        { type: 'checkbox', key: 'anti-deps',          label: 'Anti-deps'     },
  'direction': {
    type: 'select', key: 'direction', label: 'Layout',
    options: [{ value: 'vertical', label: '↕ Vertical' }, { value: 'horizontal', label: '↔ Horizontal' }],
  },
  'node-color': {
    type: 'select', key: 'node-color-call', label: 'Color',
    options: [
      { value: 'duration-total', label: 'Duration' }, { value: 'dag-depth', label: 'DAG depth' },
      { value: 'total-order', label: 'Order' },       { value: 'fixed', label: 'Fixed' },
    ],
  },
};

function buildControls(pid, view, controls) {
  var row = $(pid + '-controls');
  if (!row) return;
  controls.forEach(function (name) {
    var def = CONTROL_DEFS[name];
    if (!def) return;
    var label = document.createElement('label');
    if (def.type === 'checkbox') {
      var box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = !!view.settings[def.key];
      box.addEventListener('change', function () { view.settings[def.key] = box.checked; view.render(); });
      label.append(box, ' ' + def.label);
    } else {
      var sel = document.createElement('select');
      def.options.forEach(function (o) {
        var opt = new Option(o.label, o.value);
        if (o.value === view.settings[def.key]) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.addEventListener('change', function () { view.settings[def.key] = sel.value; view.render(); });
      label.append(def.label + ' ', sel);
    }
    row.appendChild(label);
  });
}

// ── WebAssembly programs ─────────────────────────────────────────────────────
// The Emscripten glue defines a global factory, createSkePUModule; one factory per URL.

var _factories = Object.create(null);

function loadFactory(jsUrl) {
  if (!_factories[jsUrl]) {
    _factories[jsUrl] = loadScript(jsUrl).then(function () {
      if (!window.createSkePUModule) throw new Error('createSkePUModule not exported by ' + jsUrl);
      return window.createSkePUModule;
    });
  }
  return _factories[jsUrl];
}

// Runs the program in a fresh instance (no state leaks between runs).  Resolves to the
// trace.json it wrote to its virtual file system, or null.
async function runWasm(wasm, argv, outputEl, wantTrace) {
  var lines = [];
  var trace = null;
  try {
    var factory = await loadFactory(wasm.jsUrl);
    var mod = await factory({
      print:    function (l) { lines.push(l); },
      printErr: function (l) { lines.push('[stderr] ' + l); },
    });
    mod.callMain(argv);
    if (wantTrace && mod.FS) {
      try { trace = mod.FS.readFile('trace.json', { encoding: 'utf8' }); }
      catch (_) { /* no trace: not compiled with -DSKEPU_TRACING */ }
    }
  } catch (e) {
    // Emscripten throws ExitStatus when main() returns; that is not an error.
    if (!String(e).includes('ExitStatus')) lines.push('⚠ ' + e);
  }
  outputEl.textContent = lines.join('\n');
  return trace;
}

// ── mount ────────────────────────────────────────────────────────────────────

async function mount(pid, config) {
  var views    = config.views || [];
  var controls = config.controls || [];
  var vizBase  = new URL(config.vizBase, document.baseURI).href;   // import() needs an absolute or ./ URL
  var wasm     = config.wasm || null;
  var hasGraph = views.indexOf('graph') >= 0, hasSource = views.indexOf('source') >= 0,
      hasTimeline = views.indexOf('timeline') >= 0;

  var session = null, provider = null, view = null, srcView = null;
  var graphNodes = [];

  // Builds the views and the trace worker (once; for a program, at its first run).
  async function setup() {
    setLoading(pid, true);
    var deps;
    try { deps = await loadDeps(vizBase); }
    catch (e) { setError(pid, 'Could not load visualizer: ' + (e.message || e)); return false; }

    session = new WorkerSession(vizBase + 'script/backend/visualizer.js');
    var settings = Object.assign({}, config.settings);
    provider = new deps.TraceDataProvider({
      fetchGraph: function (p) { return session.fetchGraph(p); },
      settings: settings,
    });

    if (hasGraph && $(pid + '-cy')) {
      view = new deps.GraphView($(pid + '-cy'), {
        provider: provider,
        settings: settings,
        dagreWorkerUrl: vizBase + 'script/workers/dagre-worker.js',
        elkWorkerUrl:   vizBase + 'static/cytoscape.js-elk-2.3.0/dist/elk.bundled.js',
        onRenderStart: function () { setLoading(pid, true); },
        onRenderEnd: function (stats) {
          setLoading(pid, false);
          updateStats(pid, stats);
          graphNodes = stats.graphNodes || [];
        },
        onError: function (err) { setError(pid, err.message || String(err)); },
      });
      // GraphView merges the settings with its defaults into a new object; the provider needs that one.
      provider.settings = view.settings;
      buildControls(pid, view, controls);
    }

    if (hasSource && $(pid + '-tab-bar') && $(pid + '-tab-panes')) {
      srcView = new deps.SourceCodeView({
        tabBarEl: $(pid + '-tab-bar'), tabPanesEl: $(pid + '-tab-panes'),
        hljsThemeLinkEl: $('te-hljs-theme'), initialTheme: 'default',
      });
      srcView.attachProvider(provider);   // keeps the source pane in sync with the graph
    }

    if (hasTimeline && $(pid + '-timeline')) {
      var timeline = new deps.TimelineView($(pid + '-timeline'), { animScrollMs: 400 });
      timeline.attachProvider(provider);
    }

    provider.on('select', function (ev) {
      var node = graphNodes.find(function (n) { return n.data.id === (ev.focusId || ev.nodeId); });
      if (node) showInfo(pid, node.data);
    });
    provider.on('clear', function () { clearInfo(pid); });
    return true;
  }

  // Asks the worker to load a trace (message 'loadExample' or 'load') and shows it.
  async function show(type, payload) {
    var result;
    try { result = await session.call(type, payload); }
    catch (e) { setError(pid, 'Failed to load trace: ' + (e.message || e)); return; }

    if (srcView)
      srcView.rebuild(result.cppFiles || {}, result.traceJsonRaw || null,
                      result.traceFilename || 'trace.json', result.traceLineRanges || []);
    if (view) { view.resetFirstLoad(); view.render(); }
    else if (hasTimeline) provider.fetch();   // timeline only: nothing else drives the fetch
  }

  if (!wasm) {
    if (views.length && await setup())
      await show('loadExample', { index: config.example, backend: config.backend });
    return;
  }

  // A program: run it on demand and visualize the trace it writes.
  var runBtn = $(pid + '-run'), argvEl = $(pid + '-argv'), outputEl = $(pid + '-output');
  var ready = null;

  async function run() {
    if (runBtn.disabled) return;
    runBtn.disabled = true;
    runBtn.textContent = '⏳ Running…';
    var trace = await runWasm(wasm, argvEl.value.trim().split(/\s+/).filter(Boolean), outputEl, views.length > 0);
    runBtn.disabled = false;
    runBtn.textContent = '▶ Run';
    if (!trace) return;   // no trace: leave a previous visualization as it is

    $(pid + '-placeholder').style.display = 'none';
    $(pid + '-trace').style.display = '';
    if (!ready) ready = setup();
    if (!(await ready)) return;
    var cppFiles = {};
    if (wasm.srcContent) cppFiles[wasm.srcFilename] = wasm.srcContent;
    await show('load', { traceJson: trace, traceFilename: 'trace.json', cppFiles: cppFiles });
  }

  runBtn.addEventListener('click', run);
  $(pid + '-clear').addEventListener('click', function () { outputEl.textContent = ''; });
  argvEl.addEventListener('keydown', function (e) { if (e.key === 'Enter') run(); });
}

window.skevu = { mount: mount };
}());
