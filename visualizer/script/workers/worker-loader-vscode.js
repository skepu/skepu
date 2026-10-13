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

    window.skevu.vscode.postMessage({
        type: "status_message",
        text: "Worker initialized"
    });
    
    initWorkerBackend(_worker);
    
    // Expose the worker globally so event-handler snippets in the HTML can call
    // loadExample / loadFiles without importing this module.
    window._vizWorker = _worker;
}

initWorker();