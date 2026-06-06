// simulation.js — SimulationEngine: trace playback controller.
//
// Steps through graph nodes in total_order, emitting 'select' events on the
// injected TraceDataProvider so all attached views update in sync.
//
// Usage (see graph.js):
//
//   import { SimulationEngine } from './simulation.js';
//
//   const engine = new SimulationEngine(provider, {
//     progressEl:  document.getElementById('sim-progress'),
//     fillEl:      document.getElementById('sim-progress-fill'),
//     handleEl:    document.getElementById('sim-progress-handle'),
//     ticksEl:     document.getElementById('sim-ticks'),
//     playPauseEl: document.getElementById('sim-btn-playpause'),
//   });
//
//   engine.start();   engine.pause();   engine.stop();
//   engine.toggle();  engine.step(+1);  engine.ganttClick(nodeId, iterIdx);
//   engine.isActive   engine.isDragging
//
'use strict';

// Milliseconds between automatic steps during playback.
export const SIM_STEP_DELAY_MS = 1000;

// Automatically restart from the beginning when the last node is reached.
export const SIM_WRAP = false;

// Node types to skip — synthetic container nodes with no direct trace-event
// counterpart.  Skeleton calls, transfers, etc. are all included.
export const SIM_SKIP_TYPES = new Set(['region', 'fusion']);

export class SimulationEngine {
  // provider   — TraceDataProvider instance; 'select' events are emitted through it.
  // opts.progressEl  — #sim-progress bar container element
  // opts.fillEl      — #sim-progress-fill element
  // opts.handleEl    — #sim-progress-handle element (optional)
  // opts.ticksEl     — #sim-ticks element (optional)
  // opts.playPauseEl — play/pause button element (optional)
  constructor(provider, opts) {
    opts = opts || {};
    this._provider    = provider;
    this._progressEl  = opts.progressEl  || null;
    this._fillEl      = opts.fillEl      || null;
    this._handleEl    = opts.handleEl    || null;
    this._ticksEl     = opts.ticksEl     || null;
    this._playPauseEl = opts.playPauseEl || null;

    this._running  = false;   // timer is active and stepping
    this._paused   = false;   // paused mid-sequence (index preserved)
    this._dragging = false;   // scrubber drag in progress
    this._timer    = null;
    this._entries  = [];      // flat sequence of {id, iterIdx, order, traceIdx}
    this._index    = 0;       // next entry to visit

    this._initScrubber();
  }

  // ── Public read-only state ──────────────────────────────────────────────────

  // True when playing or paused mid-sequence.
  get isActive()   { return this._running || this._paused; }

  // True while the user is dragging the scrubber handle.
  get isDragging() { return this._dragging; }

  // ── Public playback controls ────────────────────────────────────────────────

  // Start playback from the beginning, or resume after a pause.
  start() {
    if (this._running) return;
    if (this._paused) {
      this._paused  = false;
      this._running = true;
    } else {
      this._buildNodeList();
      this._index   = 0;
      this._running = true;
      this._paused  = false;
      if (!this._entries.length) { this._running = false; this._syncMenuIcons(); return; }
    }
    this._syncMenuIcons();
    this._step();
  }

  // Pause playback, preserving the current position for a later resume.
  pause() {
    if (!this._running) return;
    clearTimeout(this._timer);
    this._timer   = null;
    this._running = false;
    this._paused  = true;
    this._syncMenuIcons();
  }

  // Stop playback and reset all simulation state.
  stop() {
    clearTimeout(this._timer);
    this._timer   = null;
    this._running = false;
    this._paused  = false;
    this._entries = [];
    this._index   = 0;
    this._syncMenuIcons();
  }

  // Toggle between playing and paused/stopped.
  toggle() {
    if (this._running) this.pause(); else this.start();
  }

  // Step forward (delta = +1) or backward (delta = -1).  Pauses automatically
  // if currently playing.  No-op when the engine is not active.
  step(delta) {
    if (!this._running && !this._paused) return;
    if (this._running) this.pause();
    // _index is the *next* entry to visit; _index-1 is the currently shown one.
    var current = this._index - 1;
    var target  = Math.max(0, Math.min(this._entries.length - 1, current + delta));
    if (target === current) return;
    this._jumpToDisplay(target);
  }

  // Called by the Gantt bar click handler when the simulation is active.
  // Finds the matching entry, pauses if playing, and jumps to that point.
  // Returns true when the click was handled (caller should skip normal selection).
  ganttClick(nodeId, iterIdx) {
    if (!this.isActive) return false;
    for (var i = 0; i < this._entries.length; i++) {
      if (this._entries[i].id === nodeId && this._entries[i].iterIdx === iterIdx) {
        if (this._running) this.pause();
        this._jumpToDisplay(i);
        return true;
      }
    }
    return false;   // node not in the entry list (e.g. a collapsed region)
  }

  // ── Private machinery ───────────────────────────────────────────────────────

  // Build a flat, total_order-sorted sequence from the provider's current graph.
  // Aggregated nodes (total_orders array) are expanded into one entry per
  // iteration so they interleave correctly with other nodes.
  _buildNodeList() {
    var nodes = this._provider.lastGraphNodes;
    if (!nodes || !nodes.length) { this._entries = []; return; }
    var flat = [];
    nodes.forEach(function(n) {
      if (SIM_SKIP_TYPES.has(n.data.type)) return;
      var orders   = n.data.total_orders  || [n.data.total_order];
      var tIndices = n.data.trace_indices || [n.data.trace_index];
      orders.forEach(function(order, idx) {
        if (order != null)
          flat.push({ id: n.data.id, iterIdx: idx, order: order, traceIdx: tIndices[idx] });
      });
    });
    flat.sort(function(a, b) { return a.order - b.order; });
    this._entries = flat;
  }

  // Navigate to the entry at displayIdx and update all views via the provider.
  _jumpToDisplay(displayIdx) {
    displayIdx = Math.max(0, Math.min(this._entries.length - 1, displayIdx));
    var entry = this._entries[displayIdx];
    this._provider._emit('select', {
      nodeId:   entry.id,
      source:   'simulation',
      focusId:  entry.id,
      allIds:   [entry.id],
      iterIdx:  entry.iterIdx,   // 0-based; TimelineView uses this to scrollToInterval
      traceIdx: entry.traceIdx,  // raw trace-file index; InfoView uses this for line highlight
    });
    this._index = displayIdx + 1;
    this._updateProgress();
  }

  _step() {
    if (!this._running) return;
    if (this._index >= this._entries.length) {
      if (SIM_WRAP) {
        this._index = 0;
      } else {
        this.stop();
        return;
      }
    }
    this._jumpToDisplay(this._index);
    var self = this;
    this._timer = setTimeout(function() { self._step(); }, SIM_STEP_DELAY_MS);
  }

  _updateProgress() {
    var bar    = this._progressEl;
    var fill   = this._fillEl;
    var handle = this._handleEl;
    if (!bar || !fill) return;
    var active = this._running || this._paused;
    bar.style.display = active ? 'block' : 'none';
    var pct = (active && this._entries.length > 0) ? (this._index / this._entries.length * 100) : 0;
    fill.style.width = pct + '%';
    if (handle) handle.style.left = pct + '%';
  }

  // Render one tick mark per entry into the ticks element.
  // Limited to ≤ 500 DOM nodes by subsampling when the list is large.
  _renderTicks() {
    var container = this._ticksEl;
    if (!container || this._entries.length < 2) return;
    container.innerHTML = '';
    var n    = this._entries.length;
    var step = Math.max(1, Math.ceil(n / 500));
    var frag = document.createDocumentFragment();
    for (var i = 0; i < n; i += step) {
      var tick = document.createElement('div');
      tick.className = 'sim-tick';
      tick.style.left = (i / (n - 1) * 100) + '%';
      frag.appendChild(tick);
    }
    container.appendChild(frag);
  }

  _syncMenuIcons() {
    var btn = this._playPauseEl;
    if (btn) btn.firstChild.textContent = this._running ? 'Pause' : 'Play';
    this._updateProgress();
  }

  // Wire up the progress-bar scrubber: click to jump, drag to scrub.
  _initScrubber() {
    var bar = this._progressEl;
    if (!bar) return;
    var self = this;

    bar.addEventListener('mousedown', function(e) {
      if (!self.isActive) return;
      e.preventDefault();

      var startX   = e.clientX;
      var dragging = false;
      if (self._running) self.pause();

      function fraction(clientX) {
        var rect = bar.getBoundingClientRect();
        return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      }
      function nearestIdx(f) {
        return Math.round(f * (self._entries.length - 1));
      }

      function onMove(e) {
        if (!dragging) {
          if (Math.abs(e.clientX - startX) < 4) return;
          dragging = true;
          bar.classList.add('sim-dragging');
          self._dragging = true;
          self._renderTicks();
        }
        self._jumpToDisplay(nearestIdx(fraction(e.clientX)));
      }

      function onUp(e) {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        if (dragging) {
          bar.classList.remove('sim-dragging');
          self._dragging = false;
        } else {
          self._jumpToDisplay(nearestIdx(fraction(e.clientX)));
        }
      }

      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }
}
