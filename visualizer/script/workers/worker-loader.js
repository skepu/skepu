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

import { workerCall } from '../transport.js';
import { State } from '../state.js';
import { bindActions } from '../actions.js';
import { holdForReveal } from '../reveal.js';
import { headerSourceFiles } from '../backend/trace.js';
import { createSessionStore, sessionId } from '../session_store.js';
import { exampleSlug, findExample, encodePermalink, decodePermalink } from '../permalink.js';
import { LANGUAGE_LABELS, exampleLanguages, examplesOfLanguage, initialLanguage, hasSnapshots } from '../examples_list.js';

// ── 2. Build examples menu + welcome panel ────────────────────────────────────
// examples.json is fetched once; both the menubar submenu and the welcome-panel
// button list are populated from the same data.

const BACKEND_NAMES = {
    cpu: 'CPU', openmp: 'OpenMP', opencl: 'OpenCL', cuda: 'CUDA',
    mixed: 'CPU, OpenMP, OpenCL', synthetic: 'Synthetic', mpi: 'MPI',
};

/** The backends of an example in the order that they are shown: the MPI backend (all of the processes at once) last. */
function _orderedBackends(ex) {
    var backends = Object.keys(ex.trace || {});
    return backends.filter(function(b) { return b !== 'mpi'; }).concat(backends.filter(function(b) { return b === 'mpi'; }));
}

async function _buildExamplesUI() {
    var resp;
    try {
        resp = await fetch(new URL('../../examples/examples.json', import.meta.url));
    } catch (e) { if (_exampleNamed) _showSplash(); return; }   // no examples available (offline, missing file, etc.)
    if (!resp.ok) { if (_exampleNamed) _showSplash(); return; }

    var examples;
    try { examples = await resp.json(); } catch (e) { if (_exampleNamed) _showSplash(); return; }
    if (!examples || !examples.length) { if (_exampleNamed) _showSplash(); return; }

    _examples = examples;
    _buildExamplesMenu(examples);
    _buildWelcomeExamples(examples);
    _openFromHash();
}

// ── The link of an example (permalink.js) ─────────────────────────────────────
// The address says which example and backend are shown (#example=horner&backend=openmp), so that it can be kept and sent. It is not the
// state of the view: the settings are the reader's own. A page opened on a link loads the example the way a click on it does.

var _examples    = null;

/** The splash is for a blank session (early.js keeps it away where a trace is on its way): when the trace does not come, it is shown. */
function _showSplash() {
    if (window.skevu?.vscode || typeof acquireVsCodeApi === 'function') return;   // (an editor has never a blank session: no splash)
    document.documentElement.classList.remove('no-splash');
    try { sessionStorage.removeItem('skevu-upload-kept'); } catch (e) { /* (no storage) */ }
}
var _initialHash = window.location.hash;
// A link named an example: if it cannot be loaded there is a splash after all (a page that is restoring files has no link)
var _exampleNamed = /[#&]example=/.test(window.location.hash) || /[?&]example=/.test(window.location.search);

/** Puts the example (null: a trace of files) in the address, without a history entry. */
function _setLink(slug, backend) {
    if (window.skevu?.vscode || typeof acquireVsCodeApi === 'function') return;   // (an editor's page has no address of its own: replaceState throws there)
    var hash = encodePermalink({ slug: slug, backend: backend });
    try { history.replaceState(null, '', window.location.pathname + window.location.search + hash); } catch (e) { /* (a page that may not change its address) */ }
}

function _openFromHash() {
    var state = decodePermalink(_initialHash);
    if (!state) return;   // (no link, or the old kind: loaded below)
    var idx = findExample(_examples, state.slug);
    if (idx < 0) { _showSplash(); return; }   // (a link to an example that is not there)
    // After the page has made itself, as a click would be.
    var open = function() { setTimeout(function() { window.loadExample(idx, state.backend); }, 0); };
    if (document.readyState === 'complete') open(); else window.addEventListener('load', open, { once: true });
}

/** Populate the menubar Examples submenu. */
function _buildExamplesMenu(examples) {
    var placeholder = document.getElementById('examples-submenu-placeholder');
    if (!placeholder) return;

    var html = '<div class="submenu">';
    html    += '<div class="submenu-title">Examples<span class="menu-arrow">▸</span></div>';
    html    += '<div class="dropdown">';

    examples.forEach(function(ex, idx) {
        var backends = _orderedBackends(ex);
        if (!backends.length) return;

        if (backends.length === 1) {
            var bName = BACKEND_NAMES[backends[0]] || backends[0];
            html += '<button class="menu-action"'
                 +  ' data-click="loadExample menuClose" data-example="' + idx + '" data-backend="' + _esc(backends[0]) + '">'
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
                     +  ' data-click="loadExample menuClose" data-example="' + idx + '" data-backend="' + _esc(b) + '">'
                     +  _esc(bName) + '</button>';
            });
            html += '</div></div>';
        }
    });

    html += '</div></div>';
    html += '<div class="menu-sep"></div>';
    placeholder.outerHTML = html;
    bindActions(document.getElementById('menubar'));
}

/**
 * Populate the welcome-panel examples: a segmented control for the language of the programs (C++ or Python, if there are examples
 * of both; the one that was chosen last is remembered), and one multi-segment pill [title | backend | backend ...] for each example of it.
 */
function _buildWelcomeExamples(examples) {
    var list    = document.getElementById('welcome-examples-list');
    var section = document.getElementById('welcome-examples-section');
    var control = document.getElementById('welcome-language');
    if (!list || !section) return;

    var remembered = null;
    try { remembered = localStorage.getItem('skevu-example-language'); } catch (e) { /* (no storage: the default) */ }
    var language  = initialLanguage(examples, remembered);
    var languages = exampleLanguages(examples);

    function showList() {
        list.innerHTML = '';
        examplesOfLanguage(examples, language).forEach(function(entry) {
            var ex = entry.example, idx = entry.index;
            var backends = _orderedBackends(ex);
            if (!backends.length) return;

            var pill = document.createElement('div');
            pill.className = 'welcome-example-pill';

            var name = document.createElement('span');
            name.className   = 'welcome-example-name';
            name.textContent = ex.title;
            pill.appendChild(name);

            backends.forEach(function(b) {
                var seg = document.createElement('button');
                seg.className   = 'welcome-example-seg';
                seg.textContent = BACKEND_NAMES[b] || b;
                // A small yellow star (a camera flash) after the name: the example has snapshots of containers for this backend
                if (hasSnapshots(ex, b)) {
                    var star = document.createElement('sup');
                    star.className   = 'welcome-snapshot-star';
                    star.textContent = '✦';
                    star.title       = 'Has snapshots of containers';
                    seg.appendChild(star);
                }
                seg.onclick     = function() { window.loadExample(idx, b); };
                pill.appendChild(seg);
            });
            list.appendChild(pill);
        });
        if (control)
            control.querySelectorAll('button').forEach(function(b) {
                var selected = b.dataset.language === language;
                b.classList.toggle('selected', selected);
                b.setAttribute('aria-selected', selected ? 'true' : 'false');
            });
    }

    if (control) {
        control.innerHTML = '';
        languages.forEach(function(l) {
            var b = document.createElement('button');
            b.className        = 'welcome-language-seg';
            b.textContent      = LANGUAGE_LABELS[l];
            b.dataset.language = l;
            b.setAttribute('role', 'tab');
            b.onclick = function() {
                language = l;
                try { localStorage.setItem('skevu-example-language', l); } catch (e) { /* (no storage) */ }
                showList();
            };
            control.appendChild(b);
        });
        control.style.display = languages.length > 1 ? '' : 'none';   // (one language: nothing to choose)
    }

    showList();
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

// ── The files of this tab (session_store.js) ──────────────────────────────────
// What was uploaded is kept in the browser for the tab, and is loaded again when the tab is reloaded, unless the address names an example.

var _sessionStore = null;
try {
    if (!window.skevu?.vscode && typeof indexedDB !== 'undefined')
        _sessionStore = createSessionStore(indexedDB, sessionId(sessionStorage, function() { return crypto.randomUUID(); }));
} catch (e) { /* (no storage: no session) */ }

// (whether files are kept is in sessionStorage too, where early.js can read it before the page is painted: IndexedDB cannot be)
function _keepUploads(files) {
    if (!_sessionStore) return;
    _sessionStore.save(files).then(function(failed) {
        if (failed === null) return;
        try { sessionStorage.setItem('skevu-upload-kept', '1'); } catch (e) { /* (no storage) */ }
    });
}
function _forgetUploads() {
    try { sessionStorage.removeItem('skevu-upload-kept'); } catch (e) { /* (no storage) */ }
    if (_sessionStore) _sessionStore.forget();
}

(async function _restoreUploads() {
    if (decodePermalink(_initialHash) || new URLSearchParams(window.location.search).has('example')) return;   // (an example is on its way)
    if (!_sessionStore) { _showSplash(); return; }
    var saved = await _sessionStore.restore();
    if (!saved || !saved.trace) { _showSplash(); return; }
    // After the page has made itself (as for a link).
    var open = function() { setTimeout(function() { window.loadFiles(null, saved); }, 0); };
    if (document.readyState === 'complete') open(); else window.addEventListener('load', open, { once: true });
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
        _forgetUploads();
        if (_examples && _examples[index]) _setLink(exampleSlug(_examples[index].title), backend);

        if (result.snapshotArchiveUrl) {
            // Fetch, extract, and forward snapshot files to the worker.
            // This runs after _applyLoadResult so the graph renders immediately
            // while the (potentially larger) zip is still downloading.
            _loadSnapshotArchive(result.snapshotArchiveUrl);
        }
    } catch (e) {
        _setLoadingState(false);
        console.error('[worker-loader] loadExample failed:', e);
        _showSplash();
        if (typeof showAlert === 'function') showAlert('Failed to load example: ' + e.message);
    }
};

/**
 * For the guided tour (tour.js) when there is no trace: loads an example (Horner, OpenMP) and waits until it is drawn. Returns whether there is a trace.
 */
window.loadTourExample = async function loadTourExample() {
    if (!_examples) return false;
    var idx = findExample(_examples, 'horner');
    if (idx < 0) idx = 0;
    var backends = Object.keys(_examples[idx].trace || {});
    var backend = backends.indexOf('openmp') >= 0 ? 'openmp' : backends[0];
    var drawn = new Promise(function(resolve) {
        var done = function() {
            if (!State._traceLoaded) return;   // (the empty render that the page makes before any trace is not it)
            State.provider.off('rendersettled', done);
            resolve();
        };
        State.provider.on('rendersettled', done);
        setTimeout(function() { State.provider.off('rendersettled', done); resolve(); }, 10000);
    });
    await window.loadExample(idx, backend);
    if (!State._traceLoaded) return false;
    await drawn;
    await new Promise(function(r) { setTimeout(r, 500); });   // (the splash fades out)
    return true;
};

/**
 * Handle the file-picker form submission.
 * Called from the static page's "Load Trace" button onclick.
 */
export async function loadFiles(data, saved) {
    var traceInput    = document.getElementById('trace-file-input');
    var cppInput      = document.getElementById('cpp-files-input');
    var zipInput      = document.getElementById('snapshot-zip-input');

    // The files of the form, or the ones that this tab had before it was reloaded (see the session store, below).
    var picked = saved ?? { trace: traceInput?.files?.[0], cpp: cppInput ? Array.from(cppInput.files) : [], zip: zipInput?.files?.[0] };
    var traceFile = data ?? picked.trace;
    if (!traceFile) {
        if (typeof showAlert === 'function') showAlert('Please select a trace JSON file.');
        return;
    }

    _setLoadingState(true);

    try {
        var traceJson = data ?? await _readText(traceFile);
        var cppFiles  = {};

        if (window.skevu?.vscode && data) {
            // (the extension has the source files, and checks their hashes against the ones of the trace)
            const parsed = JSON.parse(data);
            for (let [fileName, hash] of Object.entries(headerSourceFiles(Array.isArray(parsed) ? null : parsed.header)))
                window.skevu.vscode.postMessage({
                    type: 'source_file',
                    file_name: fileName,
                    file_content_hash: hash
                });
        }

        for (var f of picked.cpp ?? [])
            cppFiles[f.name] = await _readText(f);

        // Pre-extract the snapshot zip (if provided) so we can send it in
        // the same load call.  The worker merges it into _snapshotFiles and
        // it is immediately available for the first getData call.
        var snapshotFiles = {};
        var zipFile = picked.zip;
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
            traceFilename: traceFile.name ?? 'trace.json',
            cppFiles:      cppFiles,
            snapshotFiles: snapshotFiles,
        });

        if (window.skevu?.vscode) {
            window.skevu.vscode.postMessage({
                type: 'trace_line_ranges',
                text: result.traceLineRanges
            });
        }

        _applyLoadResult(result);
        _setLink(null);
        if (!data && !saved) _keepUploads(picked);
    } catch (e) {
        _setLoadingState(false);
        if (saved) { _forgetUploads(); _showSplash(); }   // (what is kept does not load: it is not tried again at each reload)
        console.error('[worker-loader] loadFiles failed:', e);
        if (typeof showAlert === 'function') showAlert('Failed to load trace: ' + e.message);
    }
};

window.loadFiles = loadFiles;

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
    // Exit welcome mode. The first trace: the welcome panel becomes an overlay (with its present place and size) that stays until the graph
    // is drawn, then cross-fades with the panes (reveal.js). A later trace has no welcome panel to keep.
    var wp = document.getElementById('welcome-panel');
    var held = holdForReveal(document.documentElement, wp, document.body);
    document.body.classList.remove('welcome-mode');
    if (!held && wp) wp.classList.add('hidden');

    // Mark trace as loaded so the menubar render indicator activates.
    State._traceLoaded = true;

    // SourceCodeView.rebuild() stores traceLineRanges and keeps window.skevu.traceLineRanges
    // in sync, then rebuilds tabs and triggers hljs highlighting.
    if (State.sourceView)
        State.sourceView.rebuild(
            result.cppFiles    || {},
            // (the text that the code view shows: a trace that is compact is formatted; the ranges are those of that text)
            result.traceDisplayText ?? result.traceJsonRaw ?? null,
            result.traceFilename || 'trace.json',
            result.traceDisplayRanges ?? result.traceLineRanges ?? [],
            result.rankTraces || [],
            result.mpi ? result.mpi.detail_rank : null
        );
    else
        // Fallback: keep the global in sync even when sourceView isn't ready yet.
        (window.skevu ??= {}).traceLineRanges = result.traceDisplayRanges ?? result.traceLineRanges ?? [];

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
