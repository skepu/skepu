import './worker-loader.js'
import { initWorkerBackend } from '../transport.js';

// ── 1. Spawn worker ───────────────────────────────────────────────────────────

const _worker = new Worker(
    // Resolve relative to this script's location: static/script/ → backend-js/
    new URL('../backend/visualizer_web.js', import.meta.url),
    { type: 'module' }
);

initWorkerBackend(_worker);

// Expose the worker globally so event-handler snippets in the HTML can call
// loadExample / loadFiles without importing this module.
window._vizWorker = _worker;