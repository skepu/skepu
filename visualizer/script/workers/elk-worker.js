/**
 * elk-worker.js — NOT USED
 *
 * elk.bundled.js is dual-mode:
 *   • Loaded as a <script> on the main page  → exports window.ELK (ELKNode class)
 *   • Spawned as a Worker via `workerUrl`    → runs as the ELK algorithm server
 *
 * Calling `new ELK()` from inside a worker would require spawning a sub-worker,
 * which is unreliable across browsers and unnecessary here.
 *
 * Instead, _startElkWorker() in main.js builds the ELK graph on the main thread
 * and uses `new ELK({ workerUrl: ELK_WORKER_URL })` where ELK_WORKER_URL points
 * directly at elk.bundled.js.  ELK manages its own internal Worker; we get a
 * Promise back and terminate the ELK instance for cancellation.
 *
 * This file can be deleted; it is kept only as a record of why the separate
 * worker approach was abandoned.
 */
