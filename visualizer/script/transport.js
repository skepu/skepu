// transport.js — Worker RPC transport for the visualizer backend.
//
// Must be initialised once via initWorkerBackend() before fetchGraph or
// fetchGetData are called.  worker-loader.js does this on page load.
//
// ── Usage ────────────────────────────────────────────────────────────────────
//
//   import { initWorkerBackend, fetchGraph, fetchGetData, workerCall }
//       from './transport.js';
//
//   // Call once, before any render:
//   initWorkerBackend(new Worker('./backend/visualizer.js', { type: 'module' }));
//
//   // Graph data:
//   const data = await fetchGraph({ container_allocations: true, … });
//
//   // Node info:
//   const info = await fetchGetData(nodeId);
//
//   // Any other worker message:
//   const result = await workerCall('load', { traceJson, cppFiles });

'use strict';

// ── Internal RPC state ────────────────────────────────────────────────────────

let _worker  = null;          // set by initWorkerBackend()
let _pending = new Map();     // id → { resolve, reject }
let _seq     = 0;

function _call(type, payload) {
    if (!_worker) throw new Error('transport: worker not initialised — call initWorkerBackend() first');
    return new Promise(function(resolve, reject) {
        var id = String(++_seq);
        _pending.set(id, { resolve: resolve, reject: reject });
        _worker.postMessage(Object.assign({ id: id, type: type }, payload));
    });
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Initialise the transport with a spawned Web Worker instance.
 * Must be called before any other function in this module.
 * @param {Worker} worker
 */
export function initWorkerBackend(worker) {
    _worker = worker;
    worker.addEventListener('message', function(evt) {
        var data    = evt.data;
        var handler = _pending.get(data.id);
        if (!handler) return;
        _pending.delete(data.id);
        if (data.type === 'error') handler.reject(new Error(data.error));
        else                       handler.resolve(data);
    });
}

/**
 * Fetch graph data from the worker.
 *
 * @param {object}      params  Graph parameters (same keys as the old Flask
 *                              query-string: container_allocations, show_regions, …).
 * @param {AbortSignal} [_signal]  Accepted but unused — the render-sequence guard
 *                                 in GraphView already discards stale results.
 * @returns {Promise<object>}
 */
export function fetchGraph(params, _signal) {
    return _call('graph', params);
}

/**
 * Fetch node info data from the worker.
 * @param {string} nodeId  UUID of the graph node.
 * @returns {Promise<object>}
 */
export function fetchGetData(nodeId) {
    return _call('getData', { nodeId: nodeId });
}

/**
 * Send an arbitrary message to the worker and await the response.
 * @param {string} type     Message type ('load', 'loadExample', 'loadSnapshots', …).
 * @param {object} [payload]
 * @returns {Promise<object>}
 */
export function workerCall(type, payload) {
    return _call(type, payload || {});
}
