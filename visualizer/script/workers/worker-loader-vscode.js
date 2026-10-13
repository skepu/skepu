import './worker-loader.js'
import { initWorkerBackend } from '../transport.js';

// ── 1. Spawn worker ───────────────────────────────────────────────────────────

async function initWorker() {

    const response = await fetch(window.skevu.visualizerFile);
    const blob = await response.blob();
    const blobUrl = URL.createObjectURL(blob);
    
    const _worker = new Worker(blobUrl);
    
    _worker.postMessage({
        type: "init_vscode_extension",
        skepuLib: window.skevu.skepuLib,
        skepuDNN: window.skevu.skepuDNN
    });

    // The worker cannot fetch the files of the extension itself (see _fetchByPage in visualizer.js): the page does, for it.
    _worker.addEventListener('message', async function(evt) {
        if (evt.data.type !== 'fetch') return;
        const { fetchId, url } = evt.data;
        try {
            const r = await fetch(url);
            _worker.postMessage({ type: 'fetched', fetchId, ok: r.ok, status: r.status, statusText: r.statusText, text: r.ok ? await r.text() : '' });
        } catch (e) {
            _worker.postMessage({ type: 'fetched', fetchId, ok: false, status: 0, statusText: String(e), text: '' });
        }
    });

    
    initWorkerBackend(_worker);
    
    // Expose the worker globally so event-handler snippets in the HTML can call
    // loadExample / loadFiles without importing this module.
    window._vizWorker = _worker;
}

initWorker();