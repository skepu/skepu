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
//   engine.start();   engine.pause();   engine.stop();   (a click in the canvas outside every item stops: 'clear' with background)
//   engine.toggle();  engine.step(+1);  engine.ganttClick(nodeId, iterIdx);
//   engine.stepFromSelection(+1)   (arrow key with a unique selection and no simulation: starts one, paused, one step on)
//   engine.setProgressBarEnabled(on)   (opts.onActivate is called when a simulation starts)
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
  // opts.onActivate  — called when the simulation starts (the page shows the progress bar then)
  constructor(provider, opts) {
    opts = opts || {};
    this._provider    = provider;
    this._progressEl  = opts.progressEl  || null;
    this._fillEl      = opts.fillEl      || null;
    this._handleEl    = opts.handleEl    || null;
    this._ticksEl     = opts.ticksEl     || null;
    this._playPauseEl = opts.playPauseEl || null;
    this._onActivate  = opts.onActivate  || null;
    this._barEnabled  = false;  // the progress bar is shown (a setting of the page; the bar says where the selected node, or the simulation, is in the sequence)
    this._preview     = null;   // without a simulation: [position, length] of the selected node in the sequence, if it is unique

    this._running  = false;   // timer is active and stepping
    this._paused   = false;   // paused mid-sequence (index preserved)
    this._dragging = false;   // scrubber drag in progress
    this._timer    = null;
    this._entries  = [];      // flat sequence of {id, iterIdx, order, traceIdx}
    this._index    = 0;       // next entry to visit

    // What the user has selected in any view (the last 'select' of the provider), to start from: { id, iterIdx, ambiguous }.
    // A new graph has new node ids, so a selection does not outlive the graph that it was made in.
    this._selection = null;
    var self = this;
    provider.on('select', function(ev) {
      self._selection = {
        id:        ev.focusId != null ? ev.focusId : ev.nodeId,
        iterIdx:   ev.iterIdx,                                    // (undefined: the node as a whole, as a click in the graph says)
        ambiguous: !!(ev.allIds && ev.allIds.length > 1),         // (a gutter badge of several nodes)
      };
      self._refreshPreview();
    });
    // (a click in the canvas, outside of every item, deselects: that also ends a simulation, which is the way to stop it)
    provider.on('clear',      function(ev) {
      self._selection = null;
      if (ev && ev.background && self.isActive) self.stop(); else self._refreshPreview();
    });
    provider.on('fetchstart', function() { self._selection = null; self._preview = null; self._updateProgress(); });

    this._initScrubber();
  }

  // ── Public read-only state ──────────────────────────────────────────────────

  // Show or hide the progress bar (a setting; it is on when a simulation has started, until the user turns it off).
  setProgressBarEnabled(on) {
    this._barEnabled = !!on;
    this._refreshPreview();
  }

  // True when playing or paused mid-sequence.
  get isActive()   { return this._running || this._paused; }

  // True while the user is dragging the scrubber handle.
  get isDragging() { return this._dragging; }

  // ── Public playback controls ────────────────────────────────────────────────

  // Start playback from the selected node (if it is unique) or from the beginning, or resume after a pause.
  start() {
    if (this._running) return;
    if (this._paused) {
      this._paused  = false;
      this._running = true;
    } else {
      this._buildNodeList();
      // From the selected node, if there is one and it is unique (and not the last): its next one is the first step; otherwise from the beginning
      var at = this._selection && !this._selection.ambiguous ? this._uniqueEntry(this._selection) : -1;
      this._index   = at >= 0 && at + 1 < this._entries.length ? at + 1 : 0;
      this._running = true;
      this._paused  = false;
      if (!this._entries.length) { this._running = false; this._syncMenuIcons(); return; }
    }
    this._activated();
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
    this._refreshPreview();   // (the bar then says where the selected node is, if there is one)
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

  // Arrow keys with no simulation: starts one, paused, at the selected point and steps from it (+1 next, -1 previous). The point must be
  // unique (one node with one entry in the sequence); otherwise nothing is done. Returns whether it did something.
  stepFromSelection(delta) {
    if (this.isActive || !this._selection || this._selection.ambiguous) return false;
    this._buildNodeList();
    var at = this._uniqueEntry(this._selection);
    var target = at + delta;
    if (at < 0 || target < 0 || target >= this._entries.length) { this._entries = []; return false; }
    this._index   = at + 1;   // (the selection is the one that is shown)
    this._paused  = true;
    this._activated();
    this._syncMenuIcons();
    this._jumpToDisplay(target);
    return true;
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

  // The simulation has started: the page shows the progress bar.
  _activated() {
    if (this._onActivate) this._onActivate();
  }

  // The place of the selected node in the sequence, when there is no simulation: the progress bar says it, if the node is unique in the sequence
  // (the same rule as for the arrow keys). Otherwise the bar is left as it is, empty if there is no selection.
  _refreshPreview() {
    if (this.isActive) { this._updateProgress(); return; }
    var sel = this._selection;
    if (!sel) { this._preview = null; this._updateProgress(); return; }
    if (!sel.ambiguous) {
      this._buildNodeList();
      var at = this._uniqueEntry(sel);
      if (at >= 0) this._preview = [at + 1, this._entries.length];
      this._entries = [];
    }
    this._updateProgress();
  }

  // The index of the entry of the selection if there is exactly one, else -1.
  _uniqueEntry(sel) {
    var at = -1, found = 0;
    for (var i = 0; i < this._entries.length; i++) {
      var e = this._entries[i];
      if (e.id === sel.id && (sel.iterIdx == null || e.iterIdx === sel.iterIdx)) { at = i; found++; }
    }
    return found === 1 ? at : -1;
  }

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
    bar.style.display = this._barEnabled ? 'block' : 'none';
    var pct = 0;
    if (active) {
      if (this._entries.length > 0) pct = this._index / this._entries.length * 100;
    } else if (this._preview) {
      pct = this._preview[0] / this._preview[1] * 100;
    }
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
    if (btn) btn.firstChild.textContent = this._running ? 'Pause' : 'Simulate';
    this._updateProgress();
  }

  // Wire up the progress-bar scrubber: click to jump, drag to scrub.
  _initScrubber() {
    var bar = this._progressEl;
    if (!bar) return;
    var self = this;

    bar.addEventListener('mousedown', function(e) {
      if (!self.isActive) {
        // (the bar is shown without a simulation, with the selected node on it: a click or a drag on it starts one, paused)
        self._buildNodeList();
        if (!self._entries.length) return;
        self._index  = 0;
        self._paused = true;
        self._activated();
        self._syncMenuIcons();
      }
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
