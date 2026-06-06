// worker-loader.js — Static-mode initialisation.
//
// Loaded as a <script type="module"> in the static frontend/index.html page
// BEFORE ui.js.  Performs three jobs:
//
//  1. Spawns the visualizer Web Worker and registers it with transport.js.
//  2. Fetches examples/examples.json and builds the examples submenu.
//  3. Wires up the file-upload form so that selecting files triggers a worker
//     'load' call instead of a traditional HTTP form POST.
//
// After a successful load (file upload or example selection), the source pane
// is rebuilt dynamically and a graph render is triggered.

import { initWorkerBackend, workerCall } from '../transport.js';
import { State } from '../state.js';

// ── 1. Spawn worker ───────────────────────────────────────────────────────────

const _worker = new Worker(
    // Resolve relative to this script's location: static/script/ → backend-js/
    new URL('../backend/visualizer.js', import.meta.url),
    { type: 'module' }
);

initWorkerBackend(_worker);

// Expose the worker globally so event-handler snippets in the HTML can call
// loadExample / loadFiles without importing this module.
window._vizWorker = _worker;

// ── 2. Build examples menu + welcome panel ────────────────────────────────────
// examples.json is fetched once; both the menubar submenu and the welcome-panel
// button list are populated from the same data.

const BACKEND_NAMES = {
    cpu: 'CPU', openmp: 'OpenMP', opencl: 'OpenCL', cuda: 'CUDA',
};

async function _buildExamplesUI() {
    var resp;
    try {
        resp = await fetch(new URL('../../examples/examples.json', import.meta.url));
    } catch (e) { return; }   // no examples available (offline, missing file, etc.)
    if (!resp.ok) return;

    var examples;
    try { examples = await resp.json(); } catch (e) { return; }
    if (!examples || !examples.length) return;

    _buildExamplesMenu(examples);
    _buildWelcomeExamples(examples);
}

/** Populate the menubar Examples submenu. */
function _buildExamplesMenu(examples) {
    var placeholder = document.getElementById('examples-submenu-placeholder');
    if (!placeholder) return;

    var html = '<div class="submenu">';
    html    += '<div class="submenu-title">Examples<span class="menu-arrow">▸</span></div>';
    html    += '<div class="dropdown">';

    examples.forEach(function(ex, idx) {
        var backends = Object.keys(ex.trace || {});
        if (!backends.length) return;

        // Use single-quoted JS strings inside the double-quoted HTML attribute.
        // JSON.stringify would produce double quotes, breaking the attribute boundary.
        if (backends.length === 1) {
            var bName = BACKEND_NAMES[backends[0]] || backends[0];
            html += '<button class="menu-action"'
                 +  ' onclick="loadExample(' + idx + ',\'' + _jsStr(backends[0]) + '\');menuClose()">'
                 +  _esc(ex.title) + ' (' + _esc(bName) + ')'
                 +  '</button>';
        } else {
            html += '<div class="submenu">';
            html += '<div class="submenu-title">' + _esc(ex.title)
                 +  '<span class="menu-arrow">▸</span></div>';
            html += '<div class="dropdown">';
            backends.forEach(function(b) {
                var bName = BACKEND_NAMES[b] || b;
                html += '<button class="menu-action"'
                     +  ' onclick="loadExample(' + idx + ',\'' + _jsStr(b) + '\');menuClose()">'
                     +  _esc(bName) + '</button>';
            });
            html += '</div></div>';
        }
    });

    html += '</div></div>';
    html += '<div class="menu-sep"></div>';
    placeholder.outerHTML = html;
}

/** Populate the welcome-panel example buttons. */
function _buildWelcomeExamples(examples) {
    var list    = document.getElementById('welcome-examples-list');
    var section = document.getElementById('welcome-examples-section');
    if (!list || !section) return;

    examples.forEach(function(ex, idx) {
        var backends = Object.keys(ex.trace || {});
        backends.forEach(function(b) {
            var bName = BACKEND_NAMES[b] || b;
            var label = backends.length === 1
                ? _esc(ex.title)
                : _esc(ex.title) + ' (' + _esc(bName) + ')';
            var btn = document.createElement('button');
            btn.className   = 'welcome-example-btn';
            btn.innerHTML   = label;
            btn.onclick     = function() { window.loadExample(idx, b); };
            list.appendChild(btn);
        });
    });

    section.style.display = '';
}

/** Escape a string for safe embedding as HTML text / attribute content. */
function _esc(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/**
 * Escape a string for use as a single-quoted JS string literal inside an
 * HTML attribute.  Escapes backslashes and single quotes only — the result
 * is intended to appear inside onclick="...loadExample(0,'<result>')...".
 */
function _jsStr(s) {
    return String(s).replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}

// Run after the DOM is ready (this module is deferred like all ES modules).
_buildExamplesUI();

// ── 4. Auto-load from URL params ──────────────────────────────────────────────
// When main.html is reached via a link from index.html, e.g.
//   main.html?example=2&backend=openmp
// load the named example automatically and clean up the URL.

(function _autoLoadFromUrl() {
    var params = new URLSearchParams(window.location.search);
    var exIdx  = params.get('example');
    if (exIdx === null) return;

    // Remove query string from the address bar so bookmarking/reload stay clean.
    history.replaceState(null, '', window.location.pathname + window.location.hash);

    var backend = params.get('backend') || undefined;
    window.loadExample(parseInt(exIdx, 10), backend);
})();

// ── 3. Public loader API ──────────────────────────────────────────────────────
// These functions are called from the static page's event handlers.

/**
 * Load a built-in example by index + backend key.
 * Called from the examples menu onclick handlers.
 */
window.loadExample = async function loadExample(index, backend) {
    _setLoadingState(true);
    try {
        var result = await workerCall('loadExample', { index: index, backend: backend });
        _applyLoadResult(result);

        if (result.snapshotArchiveUrl) {
            // Fetch, extract, and forward snapshot files to the worker.
            // This runs after _applyLoadResult so the graph renders immediately
            // while the (potentially larger) zip is still downloading.
            _loadSnapshotArchive(result.snapshotArchiveUrl);
        }
    } catch (e) {
        _setLoadingState(false);
        console.error('[worker-loader] loadExample failed:', e);
        if (typeof showAlert === 'function') showAlert('Failed to load example: ' + e.message);
    }
};

/**
 * Handle the file-picker form submission.
 * Called from the static page's "Load Trace" button onclick.
 */
window.loadFiles = async function loadFiles() {
    var traceInput    = document.getElementById('trace-file-input');
    var cppInput      = document.getElementById('cpp-files-input');
    var zipInput      = document.getElementById('snapshot-zip-input');

    var traceFile = traceInput && traceInput.files[0];
    if (!traceFile) {
        if (typeof showAlert === 'function') showAlert('Please select a trace JSON file.');
        return;
    }

    _setLoadingState(true);

    try {
        var traceJson = await _readText(traceFile);
        var cppFiles  = {};

        if (cppInput && cppInput.files.length) {
            for (var i = 0; i < cppInput.files.length; i++) {
                var f = cppInput.files[i];
                cppFiles[f.name] = await _readText(f);
            }
        }

        // Pre-extract the snapshot zip (if provided) so we can send it in
        // the same load call.  The worker merges it into _snapshotFiles and
        // it is immediately available for the first getData call.
        var snapshotFiles = {};
        var zipFile = zipInput && zipInput.files[0];
        if (zipFile) {
            try {
                var buf = await _readBinary(zipFile);
                snapshotFiles = await _unzipToFiles(buf);
            } catch (ze) {
                console.warn('[worker-loader] Could not extract snapshot zip:', ze);
            }
        }

        var result = await workerCall('load', {
            traceJson:     traceJson,
            traceFilename: traceFile.name,
            cppFiles:      cppFiles,
            snapshotFiles: snapshotFiles,
        });

        _applyLoadResult(result);
    } catch (e) {
        _setLoadingState(false);
        console.error('[worker-loader] loadFiles failed:', e);
        if (typeof showAlert === 'function') showAlert('Failed to load trace: ' + e.message);
    }
};

// ── Internal helpers ──────────────────────────────────────────────────────────

function _setLoadingState(active) {
    var btn = document.getElementById('load-trace-btn');
    if (btn) btn.disabled = active;
}

/**
 * Apply a load/loadExample result: rebuild the source pane via SourceCodeView
 * and trigger a graph render.
 */
function _applyLoadResult(result) {
    // Exit welcome mode: hide the welcome panel and restore the source pane.
    // Removing 'welcome-mode' lets the user's stored preference (applied as an
    // inline style by ui.js:load()) take over without any extra JS.
    document.body.classList.remove('welcome-mode');
    var wp = document.getElementById('welcome-panel');
    if (wp) wp.classList.add('hidden');

    // Mark trace as loaded so the menubar render indicator activates.
    State._traceLoaded = true;

    // SourceCodeView.rebuild() stores traceLineRanges and keeps window.TRACE_LINE_RANGES
    // in sync, then rebuilds tabs and triggers hljs highlighting.
    if (State.sourceView)
        State.sourceView.rebuild(
            result.cppFiles    || {},
            result.traceJsonRaw  || null,
            result.traceFilename || 'trace.json',
            result.traceLineRanges || []
        );
    else
        // Fallback: keep the global in sync even when sourceView isn't ready yet.
        window.TRACE_LINE_RANGES = result.traceLineRanges || [];

    _setLoadingState(false);

    // Reset the first-load flag so the new trace gets the same auto-expand
    // treatment as the very first render (small traces open fully expanded).
    if (State.graphView) State.graphView.resetFirstLoad();

    // Trigger a graph render.
    if (State.graphView) State.graphView.render();
}

/**
 * Read a File object as text (UTF-8).
 * @param {File} file
 * @returns {Promise<string>}
 */
function _readText(file) {
    return new Promise(function(resolve, reject) {
        var reader = new FileReader();
        reader.onload  = function(e) { resolve(e.target.result); };
        reader.onerror = function()  { reject(new Error('Failed to read ' + file.name)); };
        reader.readAsText(file, 'utf-8');
    });
}

/**
 * Read a File object as an ArrayBuffer.
 * @param {File} file
 * @returns {Promise<ArrayBuffer>}
 */
function _readBinary(file) {
    return new Promise(function(resolve, reject) {
        var reader = new FileReader();
        reader.onload  = function(e) { resolve(e.target.result); };
        reader.onerror = function()  { reject(new Error('Failed to read ' + file.name)); };
        reader.readAsArrayBuffer(file);
    });
}

// ── ZIP extraction ────────────────────────────────────────────────────────────
// Minimal implementation using the ZIP central directory and the browser's
// native DecompressionStream API (no external dependency).
// Supports compression method 0 (stored) and 8 (deflate), which covers all
// snapshot archives produced by Python's zipfile module.

/**
 * Fetch a snapshot archive, extract all files, and forward them to the worker.
 * Called after loadExample() so the graph is already rendered; snapshot data
 * becomes available for the next node-click without blocking the initial render.
 *
 * @param {string} url  Absolute URL of the .zip archive.
 */
async function _loadSnapshotArchive(url) {
    try {
        var resp = await fetch(url);
        if (!resp.ok) throw new Error('HTTP ' + resp.status + ' fetching ' + url);
        var buf           = await resp.arrayBuffer();
        var snapshotFiles = await _unzipToFiles(buf);
        var count         = Object.keys(snapshotFiles).length;
        if (!count) { console.warn('[worker-loader] Snapshot zip was empty:', url); return; }
        await workerCall('loadSnapshots', { snapshotFiles: snapshotFiles });
        console.info('[worker-loader] Loaded', count, 'snapshot file(s) from', url);
    } catch (e) {
        console.error('[worker-loader] Failed to load snapshot archive:', e);
    }
}

/**
 * Extract all files from a ZIP archive buffer.
 * Returns { basename → utf8_text } for every non-directory entry.
 *
 * @param {ArrayBuffer} buf
 * @returns {Promise<Object>}
 */
async function _unzipToFiles(buf) {
    var data = new Uint8Array(buf);
    var view = new DataView(buf);
    var dec  = new TextDecoder('utf-8');

    // ── Locate the End of Central Directory record ────────────────────────────
    // It starts with signature PK\x05\x06 and sits at the very end of the file
    // (possibly followed by a comment of up to 65535 bytes).
    var EOCD_SIG = 0x06054b50;
    var eocd = -1;
    for (var i = data.length - 22; i >= Math.max(0, data.length - 65557); i--) {
        if (view.getUint32(i, true) === EOCD_SIG) { eocd = i; break; }
    }
    if (eocd < 0) throw new Error('Not a valid ZIP archive (EOCD not found)');

    var cdOffset = view.getUint32(eocd + 16, true);  // central directory start
    var cdSize   = view.getUint32(eocd + 12, true);  // central directory size

    // ── Walk the central directory ────────────────────────────────────────────
    var files  = {};
    var pos    = cdOffset;
    var cdEnd  = cdOffset + cdSize;
    var CD_SIG = 0x02014b50;   // PK\x01\x02

    while (pos + 46 <= cdEnd) {
        if (view.getUint32(pos, true) !== CD_SIG) break;

        var method      = view.getUint16(pos + 10, true);
        var compSize    = view.getUint32(pos + 20, true);
        var nameLen     = view.getUint16(pos + 28, true);
        var extraLen    = view.getUint16(pos + 30, true);
        var commentLen  = view.getUint16(pos + 32, true);
        var localOffset = view.getUint32(pos + 42, true);
        var name        = dec.decode(data.subarray(pos + 46, pos + 46 + nameLen));

        pos += 46 + nameLen + extraLen + commentLen;

        if (name.endsWith('/')) continue;   // directory entry

        // ── Read file data from the local file header ─────────────────────────
        // The local extra field length can differ from the CD extra, so read it
        // from the local header rather than using the CD value.
        var LFH_SIG = 0x04034b50;  // PK\x03\x04
        if (view.getUint32(localOffset, true) !== LFH_SIG)
            throw new Error('Bad local file header signature for ' + name);

        var localNameLen  = view.getUint16(localOffset + 26, true);
        var localExtraLen = view.getUint16(localOffset + 28, true);
        var dataStart     = localOffset + 30 + localNameLen + localExtraLen;
        var compData      = data.subarray(dataStart, dataStart + compSize);

        var text;
        if (method === 0) {
            text = dec.decode(compData);                    // stored
        } else if (method === 8) {
            text = await _inflate(compData);               // deflated
        } else {
            console.warn('[worker-loader] Unsupported ZIP method', method, 'for', name);
            continue;
        }

        var basename = name.split('/').pop();
        if (basename) files[basename] = text;
    }

    return files;
}

/**
 * Decompress a raw DEFLATE stream using the browser's native DecompressionStream.
 * @param {Uint8Array} compressedData
 * @returns {Promise<string>}  UTF-8 decoded text
 */
async function _inflate(compressedData) {
    var stream = new DecompressionStream('deflate-raw');
    var writer = stream.writable.getWriter();
    var reader = stream.readable.getReader();

    // Write and close in a separate microtask so the reader can start draining.
    writer.write(compressedData);
    writer.close();

    var chunks = [], totalLen = 0;
    for (;;) {
        var _ref = await reader.read();
        if (_ref.done) break;
        chunks.push(_ref.value);
        totalLen += _ref.value.length;
    }

    var out = new Uint8Array(totalLen), offset = 0;
    for (var i = 0; i < chunks.length; i++) {
        out.set(chunks[i], offset);
        offset += chunks[i].length;
    }
    return new TextDecoder('utf-8').decode(out);
}
