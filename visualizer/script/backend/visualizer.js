// visualizer.js — Web Worker replacement for Flask's visualizer.py.
//
// Loaded as an ES-module worker:
//   const worker = new Worker('./backend-js/visualizer.js', { type: 'module' });
//
// ── Message protocol ──────────────────────────────────────────────────────────
//
// Main thread → Worker:  { id: <string>, type: <string>, ...payload }
// Worker → Main thread:  { id: <string>, type: 'result',  ...result  }
//                    or:  { id: <string>, type: 'error',   error: <string> }
//
// Message types (replace the corresponding Flask routes):
//
//   'load'          POST /upload     — receive trace JSON + source files
//   'loadExample'   GET  /load_example — fetch a built-in example by index
//   'loadSnapshots' POST /upload_snapshots — merge additional snapshot files
//   'graph'         GET  /graph      — build graph and return Cytoscape data
//   'getData'       GET  /get_data   — return node info + snapshot content
//
// All parameter names in 'graph' match the Flask query-string keys exactly so
// the existing frontend fetchGraphDataAndRender() can be adapted with minimal
// changes (just swap fetch() for postMessage()).

'use strict';

import { DirectedGraph, GraphSettings } from './graph.js';
import { RegionNode, UpdateNode } from './node.js';

// ── Module-level state (mirrors Flask globals) ────────────────────────────────

let _eventData       = null;   // parsed trace events array
let _cppFiles        = {};     // { basename: content }
let _traceFilename   = null;
let _traceJsonRaw    = null;   // original file text (verbatim for source pane)
let _traceLineRanges = [];     // [[startLine, endLine], …] per top-level event
let _snapshotFiles   = {};     // { basename: text } — extracted by main thread
let _graph           = null;   // DirectedGraph from the most recent 'graph' call

// Base URL for fetching examples and library files — one level above this
// script, i.e. the visualizer root.
const _BASE_URL     = new URL('..', import.meta.url).href;
const _LIB_ROOT_URL = _BASE_URL + './static/skepu-lib/';
const _LIB_DNN_URL  = _BASE_URL + './static/skepu-lib/dnn/';

// Cached index of available library headers: { root: [...], dnn: [...] }.
// null = not yet fetched; { root: [], dnn: [] } = fetched but unavailable.
let _libIndex = null;

// ── Utility: trace line-range computation ─────────────────────────────────────
// Mirrors compute_trace_line_ranges() in visualizer.py.
// Maps each top-level JSON array element to the (1-based inclusive) line span
// it occupies in the raw source text.

function _bisectRight(arr, val) {
    let lo = 0, hi = arr.length;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (arr[mid] <= val) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/**
 * Return the index one past the end of the JSON value starting at text[pos].
 * Handles objects, arrays, strings, numbers, booleans, and null.
 * Mirrors Python's JSONDecoder.raw_decode().
 */
function _jsonValueEnd(text, pos) {
    const ch = text[pos];
    if (ch === '"') {
        let i = pos + 1;
        while (i < text.length) {
            if (text[i] === '\\') { i += 2; continue; }
            if (text[i] === '"')  return i + 1;
            i++;
        }
        throw new SyntaxError('Unterminated JSON string at offset ' + pos);
    }
    if (ch === '{' || ch === '[') {
        const close = ch === '{' ? '}' : ']';
        let depth = 1, i = pos + 1;
        while (i < text.length) {
            const c = text[i];
            if (c === '"') {
                i++;
                while (i < text.length) {
                    if (text[i] === '\\') { i += 2; continue; }
                    if (text[i] === '"')  { i++; break; }
                    i++;
                }
                continue;
            }
            if (c === '{' || c === '[') depth++;
            else if ((c === '}' || c === ']') && --depth === 0) return i + 1;
            i++;
        }
        throw new SyntaxError('Unterminated JSON object/array at offset ' + pos);
    }
    // Number, boolean, null — scan to next delimiter.
    let i = pos;
    while (i < text.length && !/[\s,\]}\r\n]/.test(text[i])) i++;
    return i;
}

function _computeTraceLineRanges(jsonText) {
    // Build sorted array of character offsets where each line begins.
    const lineStarts = [0];
    for (let i = 0; i < jsonText.length; i++)
        if (jsonText[i] === '\n') lineStarts.push(i + 1);

    const charToLine = pos => _bisectRight(lineStarts, pos);

    const ranges = [];
    let pos = jsonText.indexOf('[') + 1;
    while (pos < jsonText.length) {
        while (pos < jsonText.length && ' \t\n\r,'.includes(jsonText[pos])) pos++;
        if (pos >= jsonText.length || jsonText[pos] === ']') break;
        const start  = pos;
        const end    = _jsonValueEnd(jsonText, pos);
        ranges.push([charToLine(start), charToLine(end - 1)]);
        pos = end;
    }
    return ranges;
}

// ── Utility: file path preprocessing ─────────────────────────────────────────
// Mirrors preprocess_file_paths() in visualizer.py.
// Strips the longest common directory prefix from all `file` fields so paths
// shown in the UI are relative to the source tree root.

function _preprocessFilePaths(events) {
    const paths = events.map(e => e.file).filter(f => f && f !== '');
    if (!paths.length) return;

    // Character-by-character common prefix (matches Python os.path.commonprefix).
    let prefix = paths[0];
    for (let i = 1; i < paths.length && prefix; i++) {
        const p = paths[i];
        let j = 0;
        while (j < prefix.length && j < p.length && prefix[j] === p[j]) j++;
        prefix = prefix.slice(0, j);
    }

    // Trim to directory boundary (mirrors os.path.dirname).
    const slash = Math.max(prefix.lastIndexOf('/'), prefix.lastIndexOf('\\'));
    const dir   = slash >= 0 ? prefix.slice(0, slash) : '';
    if (!dir) return;

    for (const ev of events) {
        if (ev.file && ev.file !== '' && ev.file.startsWith(dir))
            ev.file = ev.file.slice(dir.length);
    }
}

// ── Library file loading ──────────────────────────────────────────────────────
// Port of find_library_files() in visualizer.py.
//
// Reads static/skepu-lib/index.json (generated by `make skepu-lib`) to learn
// which headers are available, then fetches whichever ones are actually
// referenced in the loaded trace.  Results are merged into _cppFiles so they
// appear in the source pane alongside user-uploaded files.
//
// Silently does nothing if the index is absent (skepu-lib not bundled) or if
// a referenced file cannot be fetched.

async function _ensureLibIndex() {
    if (_libIndex !== null) return;
    try {
        const r = await fetch(_LIB_ROOT_URL + 'index.json');
        _libIndex = r.ok ? await r.json() : { root: [], dnn: [] };
    } catch {
        _libIndex = { root: [], dnn: [] };
    }
}

async function _fetchLibraryFiles() {
    await _ensureLibIndex();
    if (!_libIndex.root.length && !_libIndex.dnn.length) return;

    // Collect basenames of every file referenced in the trace.
    const referenced = new Set(
        _eventData
            .map(ev => ev.file || '')
            .filter(Boolean)
            .map(f => f.split('/').pop().split('\\').pop())
    );

    const fetches = [];

    for (const name of _libIndex.root) {
        if (referenced.has(name) && !(name in _cppFiles)) {
            fetches.push(
                fetch(_LIB_ROOT_URL + name)
                    .then(r => r.ok ? r.text() : null)
                    .then(text => { if (text !== null) _cppFiles[name] = text; })
                    .catch(() => {})
            );
        }
    }

    for (const name of _libIndex.dnn) {
        if (referenced.has(name) && !(name in _cppFiles)) {
            fetches.push(
                fetch(_LIB_DNN_URL + name)
                    .then(r => r.ok ? r.text() : null)
                    .then(text => { if (text !== null) _cppFiles[name] = text; })
                    .catch(() => {})
            );
        }
    }

    await Promise.all(fetches);
}

// ── Handler: load ─────────────────────────────────────────────────────────────
// Replaces POST /upload.
// Payload: { traceJson, traceFilename?, cppFiles?, snapshotFiles? }
//   traceJson      — raw JSON text of the trace file
//   traceFilename  — display name (default 'trace.json')
//   cppFiles       — { basename: content } source files
//   snapshotFiles  — { basename: content } already-extracted snapshot data
// Result: { traceFilename, traceLineRanges, cppFiles, eventCount }

async function _handleLoad({ traceJson, traceFilename = 'trace.json', cppFiles = {}, snapshotFiles = {} }) {
    _traceJsonRaw    = traceJson;
    _traceFilename   = traceFilename;
    _eventData       = JSON.parse(traceJson);
    _traceLineRanges = _computeTraceLineRanges(traceJson);
    _cppFiles        = { ...cppFiles };
    _snapshotFiles   = { ...snapshotFiles };
    _graph           = null;
    _preprocessFilePaths(_eventData);
    await _fetchLibraryFiles();
    return {
        traceFilename:   _traceFilename,
        traceLineRanges: _traceLineRanges,
        traceJsonRaw:    _traceJsonRaw,
        cppFiles:        _cppFiles,
        eventCount:      _eventData.length,
    };
}

// ── Handler: loadExample ──────────────────────────────────────────────────────
// Replaces GET /load_example.
// Payload: { index, backend? }
// Fetches examples/examples.json relative to the visualizer root, then fetches
// the trace and source files for the chosen example.
// Result: same as _handleLoad

async function _handleLoadExample({ index, backend }) {
    const exUrl   = new URL('../../examples/examples.json', _BASE_URL).href;
    const examples = await fetch(exUrl).then(r => { if (!r.ok) throw new Error(r.statusText); return r.json(); });

    if (index < 0 || index >= examples.length)
        throw new RangeError('Example index out of range: ' + index);

    const ex         = examples[index];
    const traceDict  = ex.trace ?? {};
    const chosen     = backend ?? Object.keys(traceDict)[0];
    if (!(chosen in traceDict))
        throw new Error('Backend not available for this example: ' + chosen);

    const traceRelPath = traceDict[chosen];
    const traceUrl     = new URL('../../examples/' + traceRelPath, _BASE_URL).href;
    const traceText    = await fetch(traceUrl).then(r => { if (!r.ok) throw new Error(r.statusText); return r.text(); });

    _traceJsonRaw    = traceText;
    _traceFilename   = traceRelPath.split('/').pop();
    _eventData       = JSON.parse(traceText);
    _traceLineRanges = _computeTraceLineRanges(traceText);
    _cppFiles        = {};
    _snapshotFiles   = {};
    _graph           = null;
    _preprocessFilePaths(_eventData);

    // Fetch accompanying source files.
    for (const relPath of (ex.cpp_files ?? [])) {
        const url = new URL('../../examples/' + relPath, _BASE_URL).href;
        try {
            const content = await fetch(url).then(r => r.ok ? r.text() : Promise.reject(r.statusText));
            _cppFiles[relPath.split('/').pop()] = content;
        } catch { /* missing file — skip silently */ }
    }

    await _fetchLibraryFiles();

    // Snapshot archives must be pre-extracted by the main thread (we have no
    // zip library here) and sent as a subsequent 'loadSnapshots' message.
    // As a fallback, note which archive the main thread should fetch+extract.
    const sa = ex.snapshot_archive;
    const archiveRel = typeof sa === 'string' ? sa
                     : (sa && typeof sa === 'object') ? (sa[chosen] ?? null)
                     : null;

    return {
        traceFilename:    _traceFilename,
        traceLineRanges:  _traceLineRanges,
        traceJsonRaw:     _traceJsonRaw,
        cppFiles:         _cppFiles,
        eventCount:       _eventData.length,
        // Tell the main thread which archive to extract + send back, if any.
        snapshotArchiveUrl: archiveRel
            ? new URL('../examples/' + archiveRel, _BASE_URL).href
            : null,
    };
}

// ── Handler: loadSnapshots ────────────────────────────────────────────────────
// Replaces POST /upload_snapshots.
// Payload: { snapshotFiles }   — { basename: text } merged into current set
// Result:  { count }           — total number of snapshot files now held

function _handleLoadSnapshots({ snapshotFiles = {} }) {
    Object.assign(_snapshotFiles, snapshotFiles);
    return { count: Object.keys(_snapshotFiles).length };
}

// ── Handler: graph ────────────────────────────────────────────────────────────
// Replaces GET /graph.
// Parameter names match the Flask query-string keys exactly so the frontend
// requires only a minimal shim (postMessage instead of fetch).
//
// Payload: {
//   container_allocations, container_deallocations, container_transfers,
//   show_scalars, anti_deps, data_as_edges,
//   show_regions, expand_all_regions, first_load,
//   collapse_iteration, fusion_analysis, show_virtual_alias,
//   expanded_regions      — comma-separated persistent_region_id list
// }
// All boolean params are actual JS booleans (not "true"/"false" strings).
//
// Result: { nodes, edges, event_count, snapshot_count, fusion_hints,
//           all_region_pids, auto_expand_regions, timeline_nodes, badge_nodes,
//           request_time, response_time }

function _handleGraph({
    mode                    = 'dependence-dag',
    container_allocations   = false,
    container_deallocations = false,
    container_transfers     = false,
    show_scalars            = false,
    anti_deps               = false,
    data_as_edges           = false,
    show_regions            = false,
    expand_all_regions      = false,
    first_load              = false,
    collapse_iteration      = false,
    fusion_analysis         = false,
    show_virtual_alias      = false,
    expanded_regions        = '',
} = {}) {
    if (!_eventData) {
        // No trace loaded yet.  Return a minimal empty-graph response so the
        // page initialises cleanly without a console error.  This only happens
        // in static/worker mode where the page loads before any trace is chosen;
        // Flask always has a trace in server state before serving the main page.
        const t = Date.now() / 1000;
        return {
            nodes: [], edges: [],
            event_count: 0, snapshot_count: 0,
            fusion_hints: [], all_region_pids: [],
            auto_expand_regions: false,
            timeline_nodes: [], badge_nodes: [],
            request_time: t, response_time: t,
        };
    }

    const t0 = Date.now() / 1000;   // seconds since epoch — matches Flask's time.time()

    const settings = new GraphSettings();
    settings.mode          = mode;
    settings.antideps      = anti_deps;
    settings.allocations   = container_allocations;
    settings.deallocations = container_deallocations;
    settings.transfers     = container_transfers;
    settings.updates       = !data_as_edges;        // data_as_edges=true → updates=false (edges only)
    settings.scalars       = show_scalars;
    settings.regions       = show_regions;
    settings.synthesizeRootRegion = (mode == 'program-tree');

    // Re-preprocess paths on every build (idempotent after first call, matches Flask).
    _preprocessFilePaths(_eventData);
    _graph = DirectedGraph.fromEvents(_eventData, settings);

    const snapshotEvents = _eventData.filter(e => e.type === 'snapshot');

    _graph.computeDepths();
    _graph.findCriticalPath();
    _graph.computeKeyPaths();

    const fusionHints = fusion_analysis ? _graph.findFusions() : [];

    // Compute region extents before coalescing so each iteration gets its own span.
    if (_graph.settings.regions) {
        for (const node of _graph.getAllNodes()) {
            if (node instanceof RegionNode && !node.region)
                node.computeExtent();
        }
    }

    if (collapse_iteration) {
        _graph.coalesceIterations();
    } else {
        _graph.numberRegionLabels();
    }

    if (_graph.settings.regions) _graph.pruneEmptyRegions();

    _graph.createVirtualAliasEdges(show_virtual_alias);

    // Small traces are pre-expanded on first load so the user sees the full
    // structure without manually opening every region.
    const REGION_AUTO_EXPAND_THRESHOLD = 200;
    const autoExpandRegions = first_load && _eventData.length < REGION_AUTO_EXPAND_THRESHOLD;
    const expandAll         = expand_all_regions || autoExpandRegions;

    const expandedPids = expanded_regions
        ? new Set(expanded_regions.split(',').filter(Boolean))
        : new Set();

    const [nodes, edges, allRegionPids]   = _graph.toCytoscape([...expandedPids], expandAll);
    const [timelineNodes, badgeNodes]     = _graph.toTimelineAndBadges([...expandedPids], expandAll);

    const t1 = Date.now() / 1000;   // seconds since epoch

    return {
        nodes,
        edges,
        event_count:         _eventData.length,
        snapshot_count:      snapshotEvents.length,
        fusion_hints:        fusionHints,
        all_region_pids:     allRegionPids,
        auto_expand_regions: autoExpandRegions,
        timeline_nodes:      timelineNodes,
        badge_nodes:         badgeNodes,
        request_time:        t0,
        response_time:       t1,
    };
}

// ── Handler: getData ──────────────────────────────────────────────────────────
// Replaces GET /get_data.
// Payload: { nodeId }  — node UUID (worker mode); Flask mode passes { id } via query string
// Result:  infoData object (see Node.infoData()), extended with optional keys:
//   content_snapshots              — array of snapshot file texts
//   content_snapshot_labels        — parallel array of display labels
//   content_snapshot_virtual_label — virtual-container group label (if applicable)

function _handleGetData({ id, nodeId }) {
    // Accept both 'nodeId' (worker-mode RPC, avoids collision with seq id) and
    // legacy 'id' (Flask query-string forwarded directly) so both modes work.
    const resolvedId = nodeId !== undefined ? nodeId : id;
    if (!_graph) throw new Error('No graph built yet — send a "graph" message first');
    const node = _graph.getNodeById(resolvedId);
    if (!node)  throw new Error('Node not found: ' + resolvedId);

    const infoData = node.infoData();
    const hasSnapshots = Object.keys(_snapshotFiles).length > 0;

    // ── Virtual-container path ────────────────────────────────────────────────
    // When iteration-coalescing is active and the node belongs to a virtual
    // container group, interleave snapshots from all sibling members ordered
    // by execution (total_order).

    const virtualId    = node.virtualId ?? null;
    const isAggregated = (node.iterationCount ?? 1) > 1;

    if (virtualId !== null && isAggregated && virtualId in _graph._virtual_containers) {
        const vcInfo   = _graph._virtual_containers[String(virtualId)];
        const vcLabel  = vcInfo.label;
        const vcObjIds = vcInfo.object_ids;

        // Collect the canonical UpdateNode for each object_id in the group.
        const oidToNode = {};
        for (const n of _graph.getAllNodes()) {
            if (n instanceof UpdateNode && String(n.virtualId ?? '') === String(virtualId)) {
                if (!(n.objectId in oidToNode)) oidToNode[n.objectId] = n;
            }
        }

        const siblings     = vcObjIds.map(oid => oidToNode[oid]).filter(Boolean);
        const siblingLabels = siblings.map(n => n.label);

        if (hasSnapshots && siblings.length > 0) {
            const fileLists = siblings.map(n => n.snapshotDataFiles ?? []);
            const maxIters  = Math.max(...fileLists.map(fl => fl.length), 0);

            const combinedFiles  = [];
            const combinedLabels = [];

            for (let j = 0; j < maxIters; j++) {
                // Gather this iteration's slot across all members, sort by total_order.
                const slot = [];
                for (let i = 0; i < fileLists.length; i++) {
                    if (j < fileLists[i].length) {
                        const [order, file] = fileLists[i][j];
                        slot.push({ order, file, label: `${siblingLabels[i]}@${j + 1}` });
                    }
                }
                slot.sort((a, b) => (a.order == null ? 1 : 0) - (b.order == null ? 1 : 0)
                                 || (a.order ?? 0) - (b.order ?? 0));
                for (const { file, label } of slot) {
                    combinedFiles.push(file);
                    combinedLabels.push(label);
                }
            }

            const snapshots = [], validLabels = [];
            for (let i = 0; i < combinedFiles.length; i++) {
                const basename = combinedFiles[i].split(/[/\\]/).pop();
                const content  = _snapshotFiles[basename];
                if (content != null) { snapshots.push(content); validLabels.push(combinedLabels[i]); }
            }
            if (snapshots.length > 0) {
                infoData.content_snapshots              = snapshots;
                infoData.content_snapshot_labels        = validLabels;
                infoData.content_snapshot_virtual_label = vcLabel;
            }
        }

    // ── Plain (non-virtual) path ──────────────────────────────────────────────
    } else {
        const dataFiles = node.snapshotDataFiles ?? [];
        if (dataFiles.length > 0 && hasSnapshots) {
            const snapshots = [];
            for (const [, file] of dataFiles) {
                const basename = file.split(/[/\\]/).pop();
                const content  = _snapshotFiles[basename];
                if (content != null) snapshots.push(content);
            }
            if (snapshots.length > 0) infoData.content_snapshots = snapshots;
        }
    }

    return infoData;
}

// ── Message dispatch ──────────────────────────────────────────────────────────

self.onmessage = async function onMessage(evt) {
    const { id, type, ...payload } = evt.data;
    try {
        let result;
        switch (type) {
            case 'load':          result = await _handleLoad(payload);        break;
            case 'loadExample':   result = await _handleLoadExample(payload); break;
            case 'loadSnapshots': result = _handleLoadSnapshots(payload);     break;
            case 'graph':         result = _handleGraph(payload);             break;
            case 'getData':       result = _handleGetData(payload);           break;
            default: throw new Error('Unknown message type: ' + JSON.stringify(type));
        }
        self.postMessage({ id, type: 'result', ...result });
    } catch (err) {
        self.postMessage({ id, type: 'error', error: err.message ?? String(err) });
    }
};

// ── Convenience: Promise-based RPC helper (importable by the main thread) ─────
// Usage (main thread):
//   import { WorkerRPC } from './backend-js/visualizer.js';
//   const rpc = new WorkerRPC(worker);
//   const data = await rpc.call('graph', { container_allocations: true, ... });

export class WorkerRPC {
    constructor(worker) {
        this._worker  = worker;
        this._pending = new Map(); // id → { resolve, reject }
        this._seq     = 0;
        worker.addEventListener('message', ({ data }) => {
            const handler = this._pending.get(data.id);
            if (!handler) return;
            this._pending.delete(data.id);
            if (data.type === 'error') handler.reject(new Error(data.error));
            else handler.resolve(data);
        });
    }

    call(type, payload = {}) {
        return new Promise((resolve, reject) => {
            const id = String(++this._seq);
            this._pending.set(id, { resolve, reject });
            this._worker.postMessage({ id, type, ...payload });
        });
    }
}
