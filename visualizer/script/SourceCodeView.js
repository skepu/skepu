// SourceCodeView.js — Source / trace code pane: tabs, hljs theming,
//                     syntax highlighting, line highlighting, and badge
//                     placement + click cycling.
//
// ── Usage ─────────────────────────────────────────────────────────────────────
//
//   const src = new SourceCodeView({
//     onBadgeClick: (focusId, focusIterIdx, allIds) => { … },
//   });
//
//   // After page load — highlight any pre-rendered panes:
//   src.highlightAllBlocks(document.getElementById('cpp_tab_panes'));
//
//   // After a trace load — rebuild panes from new files:
//   src.rebuild(cppFiles, traceJson, traceFilename, traceLineRanges);
//
//   // From InfoView / simulation:
//   src.highlightSourceLine(lineNr, filePath);
//   src.highlightTraceEntry(traceIndex);
//   src.clearHighlight();
//
//   // Badge lifecycle (called by GraphView render callbacks):
//   src.clearBadges();
//   src.placeBadges(badgeNodes);

'use strict';

// ── Highlight.js dark-theme set ───────────────────────────────────────────────
// Names of themes that look best on dark backgrounds; used to toggle the
// hljs-theme-dark class on <body> so custom CSS rules can adapt.
const _HLJS_DARK_THEMES = new Set([
  'a11y-dark', 'agate', 'an-old-hope', 'androidstudio', 'atom-one-dark',
  'atom-one-dark-reasonable', 'dark', 'devibeans', 'github-dark',
  'github-dark-dimmed', 'gradient-dark', 'hybrid', 'ir-black',
  'isbl-editor-dark', 'kimbie-dark', 'monokai', 'monokai-sublime',
  'night-owl', 'nnfx-dark', 'nord', 'obsidian', 'panda-syntax-dark',
  'paraiso-dark', 'qtcreator-dark', 'rose-pine', 'rose-pine-moon',
  'shades-of-purple', 'srcery', 'stackoverflow-dark', 'sunburst',
  'tokyo-night-dark', 'tomorrow-night-blue', 'tomorrow-night-bright',
  'vs2015', 'xt256',
]);

// ── SourceCodeView ────────────────────────────────────────────────────────────

export class SourceCodeView {

  // ── Constructor ─────────────────────────────────────────────────────────────

  /**
   * @param {object}      [options]
   * @param {HTMLElement} [options.tabBarEl]         Element containing the .cpp-tab buttons
   * @param {HTMLElement} [options.tabPanesEl]       Element containing the .cpp-tab-pane divs
   * @param {HTMLElement} [options.hljsThemeLinkEl]  <link id="hljs-theme"> stylesheet element
   * @param {HTMLElement} [options.themeTargetEl]    Element to receive the hljs-theme-dark class
   *                                                 (defaults to document.body for the main page;
   *                                                 pass a container element for embedded use)
   * @param {string}      [options.initialTheme]     Starting hljs theme name
   * @param {Function}    [options.onThemeChange]    theme → void  (persist / sync State)
   * @param {Function}    [options.onBadgeClick]     (focusId, focusIterIdx, allIds) → void
   */
  constructor(options) {
    options = options || {};
    this._tabBarEl        = options.tabBarEl        || document.getElementById('cpp_tab_bar');
    this._tabPanesEl      = options.tabPanesEl      || document.getElementById('cpp_tab_panes');
    this._hljsThemeLinkEl = options.hljsThemeLinkEl || document.getElementById('hljs-theme');
    this._themeTargetEl   = options.themeTargetEl   || document.body;
    this._hljsTheme       = options.initialTheme    || 'default';
    this._onThemeChange   = options.onThemeChange   || null;
    this._onBadgeClick    = options.onBadgeClick    || null;

    this._codePaths       = options.codePaths !== undefined ? options.codePaths : true;
    this._traceLineRanges = [];
    this._badgeCycleIndex = {};
    this._badgeNodeMap    = new Map();  // nodeId → gutter element; rebuilt by _placeBadgesInDOM

    // Provider attachment (set by attachProvider).
    this._attachedProvider = null;
    this._boundFetchStart  = null;
    this._boundRenderEnd   = null;
    this._boundOnSelect    = null;
    this._boundOnClear     = null;

    // Badge clicks: handled via event delegation on the panes container so that
    // each SourceCodeView instance handles its own badges without a global
    // window.badgeClick dispatcher.
    var self = this;
    if (this._tabPanesEl) {
      this._tabPanesEl.addEventListener('click', function(e) {
        var badge = e.target.closest('[data-badge-entries]');
        if (!badge) return;
        try {
          self.badgeClick(JSON.parse(badge.dataset.badgeEntries));
        } catch (err) {
          console.error('[SourceCodeView] badge click parse error:', err);
        }
      });
    }
  }

  // ── Provider attachment ──────────────────────────────────────────────────────

  /**
   * Register this view as a listener on a TraceDataProvider so that it
   * clears and repaints automatically on every render cycle:
   *
   *   'fetchstart'  →  clearHighlight() + clearBadges()
   *   'renderend'   →  clearBadges()   + placeBadges(stats.badgeNodes)
   *
   * Safe to call more than once — the previous attachment is removed first.
   * The SourceCodeView does not need to be rebuilt separately; call rebuild()
   * once after loading a new trace, then let the provider drive badge updates.
   *
   * @param {TraceDataProvider} provider
   * @returns {this}
   */
  attachProvider(provider) {
    this.detachProvider();

    var self = this;
    this._boundFetchStart = function() {
      self.clearHighlight();
      self.clearBadges();
    };
    this._boundRenderEnd = function(stats) {
      self.clearBadges();
      self.placeBadges(stats.badgeNodes);
    };
    // Incoming selection from another view — highlight the node in the source pane.
    // Skip when source === 'code-listing' to avoid echo-back for our own badge clicks.
    this._boundOnSelect = function(ev) {
      if (ev.source === 'code-listing') return;
      var nodeId = ev.focusId || ev.nodeId;
      var nodes  = provider.lastGraphNodes;
      for (var i = 0; i < nodes.length; i++) {
        if (nodes[i].data.id === nodeId) {
          self.highlightNode(nodes[i].data, true);
          break;
        }
      }
    };

    this._boundOnClear = function() {
      self.clearHighlight();
    };

    provider.on('fetchstart', this._boundFetchStart);
    provider.on('renderend',  this._boundRenderEnd);
    provider.on('select',     this._boundOnSelect);
    provider.on('clear',      this._boundOnClear);
    this._attachedProvider = provider;
    return this;
  }

  /**
   * Remove the listeners registered by attachProvider().
   * A no-op when no provider is currently attached.
   *
   * @returns {this}
   */
  detachProvider() {
    if (this._attachedProvider) {
      this._attachedProvider.off('fetchstart', this._boundFetchStart);
      this._attachedProvider.off('renderend',  this._boundRenderEnd);
      this._attachedProvider.off('select',     this._boundOnSelect);
      this._attachedProvider.off('clear',      this._boundOnClear);
      this._attachedProvider = null;
      this._boundFetchStart  = null;
      this._boundRenderEnd   = null;
      this._boundOnSelect    = null;
      this._boundOnClear     = null;
    }
    return this;
  }

  // ── Tab management ───────────────────────────────────────────────────────────

  switchTab(btn, index) {
    this._tabBarEl.querySelectorAll('.cpp-tab').forEach(function(t) {
      t.classList.remove('active');
    });
    this._tabPanesEl.querySelectorAll('.cpp-tab-pane').forEach(function(p) {
      p.classList.remove('active');
    });
    if (btn) {
      btn.classList.add('active');
      btn.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
    }
    var panes = this._tabPanesEl.querySelectorAll('.cpp-tab-pane');
    if (panes[index]) panes[index].classList.add('active');

    // Redraw arrows for the newly active pane (hidden panes can't be measured).
    var self = this;
    requestAnimationFrame(function() {
      var activePane = self._tabPanesEl.querySelector('.cpp-tab-pane.active');
      if (activePane && activePane._badgeArrowData) {
        self._drawBadgeArrows(activePane, activePane._badgeArrowData);
      }
    });
  }

  // ── Highlight.js theme management ────────────────────────────────────────────

  /**
   * Apply a new hljs theme: swap the stylesheet, update body class, sync menu
   * icons, and fire onThemeChange so the caller can persist it.
   *
   * @param {string} theme  Theme name (no path or extension).
   */
  /**
   * Enable or disable the badge-arrow overlay.  When disabled all overlays are
   * removed immediately; when re-enabled they are redrawn for the active pane.
   */
  setCodePaths(enabled) {
    this._codePaths = !!enabled;
    var self = this;
    if (!enabled) {
      // SVG lives inside the <pre> now; querySelectorAll on the panes container
      // still finds it (no `:scope >` restriction needed).
      this._tabPanesEl.querySelectorAll('.badge-arrow-svg').forEach(function(el) { el.remove(); });
    } else {
      requestAnimationFrame(function() {
        var activePane = self._tabPanesEl.querySelector('.cpp-tab-pane.active');
        if (activePane && activePane._badgeArrowData) {
          self._drawBadgeArrows(activePane, activePane._badgeArrowData);
        }
      });
    }
  }

  setTheme(theme) {
    this._hljsTheme = theme;
    this._applyThemeCss(theme);
    this.syncThemeIcons();
    if (this._onThemeChange) this._onThemeChange(theme);
  }

  /**
   * Restore a persisted theme: update the instance property and swap the
   * stylesheet, but do NOT fire onThemeChange (no redundant save).
   * Called via window._applyHljsTheme from settings.js loadSettings().
   *
   * @param {string} theme
   */
  restoreTheme(theme) {
    this._hljsTheme = theme;
    this._applyThemeCss(theme);
  }

  /**
   * Swap the hljs stylesheet and the hljs-theme-dark body class.
   * Pure CSS operation — does not touch instance state.
   *
   * @param {string} theme
   */
  _applyThemeCss(theme) {
    var link = this._hljsThemeLinkEl;
    if (link) {
      var base = link.href.replace(/\/[^/]+\.css(\?.*)?$/, '/');
      link.href = base + theme + '.css';
    }
    this._themeTargetEl.classList.toggle('hljs-theme-dark', _HLJS_DARK_THEMES.has(theme));
  }

  /** Update the ✓ checkmarks in the Source Code Theme submenu. */
  syncThemeIcons() {
    var current = this._hljsTheme;
    document.querySelectorAll('.hljs-theme-radio').forEach(function(btn) {
      btn.querySelector('.menu-icon').textContent =
        btn.dataset.theme === current ? '✓' : '';
    });
  }

  // ── Source pane rebuild ──────────────────────────────────────────────────────

  /**
   * Replace the tab bar and panes with new content after a trace load.
   * Resets TRACE_LINE_RANGES and triggers hljs highlighting on the new blocks.
   *
   * @param {Object}      cppFiles         { basename → content }
   * @param {string|null} traceJson        Raw trace JSON text, or null
   * @param {string}      traceFilename    Display name for the trace tab
   * @param {Array}       traceLineRanges  Per-trace-event [startLine, endLine] pairs
   */
  rebuild(cppFiles, traceJson, traceFilename, traceLineRanges) {
    this._traceLineRanges    = traceLineRanges || [];
    // Keep the window global in sync for any legacy code that reads it directly.
    window.TRACE_LINE_RANGES = this._traceLineRanges;

    var tabBar   = this._tabBarEl;
    var tabPanes = this._tabPanesEl;
    if (!tabBar || !tabPanes) return;

    tabBar.innerHTML   = '';
    tabPanes.innerHTML = '';

    var tabIndex = 0;
    var firstTab = true;
    var self     = this;

    // ── Source-file tabs ──────────────────────────────────────────────────────
    Object.keys(cppFiles).forEach(function(filename) {
      var isActive = firstTab;
      firstTab     = false;

      var btn = document.createElement('button');
      btn.className   = 'cpp-tab' + (isActive ? ' active' : '');
      btn.textContent = filename;
      (function(idx) {
        btn.onclick = function() { self.switchTab(this, idx); };
      })(tabIndex);
      tabBar.appendChild(btn);

      var pane = document.createElement('div');
      pane.className        = 'cpp-tab-pane' + (isActive ? ' active' : '') + ' hljs-loading';
      pane.dataset.filename = filename;

      var pre  = document.createElement('pre');
      var code = document.createElement('code');
      code.textContent = cppFiles[filename];
      pre.appendChild(code);
      pane.appendChild(pre);
      tabPanes.appendChild(pane);

      tabIndex++;
    });

    // ── Trace JSON tab ────────────────────────────────────────────────────────
    if (traceJson) {
      var isTraceActive = firstTab;   // true only when no cpp files

      var traceBtn = document.createElement('button');
      traceBtn.className   = 'cpp-tab' + (isTraceActive ? ' active' : '');
      traceBtn.textContent = traceFilename;
      (function(idx) {
        traceBtn.onclick = function() { self.switchTab(this, idx); };
      })(tabIndex);
      tabBar.appendChild(traceBtn);

      var tracePane = document.createElement('div');
      tracePane.className         = 'cpp-tab-pane'
                                  + (isTraceActive ? ' active' : '') + ' hljs-loading';
      tracePane.dataset.filename  = traceFilename;
      tracePane.dataset.tracePane = 'true';

      var tPre  = document.createElement('pre');
      var tCode = document.createElement('code');
      tCode.className   = 'language-json';
      tCode.textContent = traceJson;
      tPre.appendChild(tCode);
      tracePane.appendChild(tPre);
      tabPanes.appendChild(tracePane);
    }

    this._highlightAllBlocks(tabPanes);
  }

  // ── Syntax highlighting ──────────────────────────────────────────────────────

  /**
   * Apply hljs + line-numbers to all <code> blocks inside `container`.
   * The active pane is highlighted synchronously first so it is immediately
   * readable; remaining panes are deferred one per event-loop turn to keep the
   * page responsive.
   *
   * @param {HTMLElement} container  Element containing .cpp-tab-pane children.
   */
  highlightAllBlocks(container) {
    if (!container || typeof hljs === 'undefined') return;

    function _do(block) {
      if (!block.classList.contains('hljs')) {
        var fn = hljs.highlightElement || hljs.highlightBlock;
        if (fn) fn.call(hljs, block);
      }
      if (hljs.lineNumbersBlock && !block.querySelector('table'))
        hljs.lineNumbersBlock(block);
      var pane = block.closest('.cpp-tab-pane');
      if (pane) pane.classList.remove('hljs-loading');
    }

    var allBlocks   = Array.from(container.querySelectorAll('.cpp-tab-pane code'));
    var activeBlock = container.querySelector('.cpp-tab-pane.active code');

    if (activeBlock) {
      _do(activeBlock);
      allBlocks = allBlocks.filter(function(b) { return b !== activeBlock; });
    }

    (function _next(i) {
      if (i >= allBlocks.length) return;
      _do(allBlocks[i]);
      setTimeout(function() { _next(i + 1); }, 0);
    })(0);
  }

  // Alias used internally by rebuild() — same logic.
  _highlightAllBlocks(container) { this.highlightAllBlocks(container); }

  // ── Line highlighting ────────────────────────────────────────────────────────

  /**
   * Highlight a node's location in whichever tab is currently active.
   *
   * - If the trace JSON tab is active: highlight the trace entry so the user
   *   stays in that tab (no switch to a source file tab).
   * - Otherwise: switch to the relevant source file tab and highlight the line.
   *
   * This is the preferred single-call entry point for node-click and badge-click
   * handlers; it avoids hard-coding the tab-detection logic in each caller.
   *
   * @param {object}  data     Plain node-data object (from GraphView onNodeClick
   *                           or looked up from stats.graphNodes).
   *                           Relevant fields: trace_index, trace_indices, file, line.
   * @param {boolean} [animate=true]  Passed through to the underlying highlight method.
   */
  highlightNode(data, animate) {
    if (!data) return;
    var tracePane = this._tabPanesEl.querySelector('.cpp-tab-pane.active[data-trace-pane="true"]');
    if (tracePane) {
      // Trace tab is open — highlight there and do not switch tabs.
      var idx = (data.trace_indices && data.trace_indices.length)
        ? data.trace_indices[0]
        : data.trace_index;
      if (idx != null) this.highlightTraceEntry(idx, animate);
    } else {
      // Source tab is open (or none yet) — highlight the source line, switching
      // to the right source-file tab if necessary.
      // Accept both Cytoscape element data fields (line, file — lowercase) and
      // server-response fields (Line, File — capitalised) so this method works
      // as the single entry point from both InfoView and direct node-click handlers.
      var line = data.line != null ? data.line : data.Line;
      var file = data.file         ? data.file : data.File;
      if (line != null && line !== -1 && file)
        this.highlightSourceLine(line, file, animate);
    }
  }

  clearHighlight() {
    this._tabPanesEl.querySelectorAll('.hljs-ln-line').forEach(function(el) {
      el.classList.remove('line-selected');
    });
  }

  /**
   * Switch to the source tab for `filePath`, scroll to `lineNr`, and highlight
   * the corresponding hljs line element.
   *
   * @param {number}  lineNr    1-based line number; -1 = do nothing.
   * @param {string}  filePath  Absolute or relative path; basename is matched
   *                            against cpp-tab-pane data-filename attributes.
   * @param {boolean} [animate=true]  Pass false to skip the scroll animation
   *                                  (e.g. while the simulation scrubber is dragging).
   */
  highlightSourceLine(lineNr, filePath, animate) {
    this.clearHighlight();
    if (lineNr == -1) return;

    var self      = this;
    var basename  = filePath ? filePath.split('/').pop().split('\\').pop() : '';
    var foundFile = false;

    if (filePath) {
      self._tabPanesEl.querySelectorAll('.cpp-tab-pane').forEach(function(pane, i) {
        if (pane.dataset.filename === basename) {
          self.switchTab(self._tabBarEl.querySelectorAll('.cpp-tab')[i], i);
          foundFile = true;
        }
      });
    }

    if (!foundFile) return;

    var animMs     = animate === false ? 0 : SourceCodeView.ANIM_SCROLL_MS;
    var activePane = self._tabPanesEl.querySelector('.cpp-tab-pane.active');
    var targets    = activePane
      ? $(activePane).find('.hljs-ln-line[data-line-number="' + lineNr + '"]')
      : $(self._tabPanesEl).find('.hljs-ln-line[data-line-number="' + lineNr + '"]');

    targets.addClass('line-selected');
    if (targets[0])
      $(activePane || self._tabPanesEl).scrollTo(
        targets[0], animMs, { over: { top: -5 } });
  }

  /**
   * Highlight a range of lines in the trace JSON pane corresponding to a single
   * trace-event entry.  Only called when the trace tab is already active.
   *
   * @param {number}  traceIndex  Index into this._traceLineRanges.
   * @param {boolean} [animate=true]  Pass false to skip the scroll animation.
   */
  highlightTraceEntry(traceIndex, animate) {
    this.clearHighlight();

    if (traceIndex == null) return;
    var range = this._traceLineRanges[traceIndex];
    if (!range) return;

    var animMs    = animate === false ? 0 : SourceCodeView.ANIM_SCROLL_MS;
    var tracePane = this._tabPanesEl.querySelector('.cpp-tab-pane[data-trace-pane="true"]');
    if (!tracePane) return;

    var firstTarget = null;
    for (var ln = range[0]; ln <= range[1]; ln++) {
      var els = $(tracePane).find('.hljs-ln-line[data-line-number="' + ln + '"]');
      els.addClass('line-selected');
      if (!firstTarget && els[0]) firstTarget = els[0];
    }
    if (firstTarget)
      $(tracePane).scrollTo(firstTarget, animMs, { over: { top: -5 } });
  }

  // ── Badge management ─────────────────────────────────────────────────────────

  /** Remove all badge pills and arrow overlays from the source panes. */
  clearBadges() {
    this._tabPanesEl.querySelectorAll('.badge').forEach(function(el) { el.remove(); });
    this._tabPanesEl.querySelectorAll('.badge-arrow-svg').forEach(function(el) { el.remove(); });
    this._tabPanesEl.querySelectorAll('.cpp-tab-pane').forEach(function(pane) {
      delete pane._badgeArrowData;
    });
    this._badgeNodeMap = new Map();
  }

  /**
   * Add a hover-highlight class to the badge gutter element associated with
   * the given node ID (if one exists).  Clears any previous highlight first.
   * @param {string} nodeId
   */
  highlightBadgeNode(nodeId) {
    this.clearBadgeHighlight();
    if (!this._badgeNodeMap) return;
    var gutterEl = this._badgeNodeMap.get(nodeId);
    if (!gutterEl) return;
    gutterEl.querySelectorAll('.badge').forEach(function(b) {
      b.classList.add('badge-node-hover');
    });
    // If the badge lives in an inactive tab, highlight that tab's title button.
    var pane = gutterEl.closest('.cpp-tab-pane');
    if (pane && !pane.classList.contains('active')) {
      var panes = Array.from(this._tabPanesEl.querySelectorAll('.cpp-tab-pane'));
      var tabs  = Array.from(this._tabBarEl.querySelectorAll('.cpp-tab'));
      var idx   = panes.indexOf(pane);
      if (idx >= 0 && tabs[idx]) tabs[idx].classList.add('badge-tab-hover');
    }
  }

  /** Remove the hover-highlight class from all badges and tab titles. */
  clearBadgeHighlight() {
    this._tabPanesEl.querySelectorAll('.badge.badge-node-hover').forEach(function(b) {
      b.classList.remove('badge-node-hover');
    });
    this._tabBarEl.querySelectorAll('.cpp-tab.badge-tab-hover').forEach(function(t) {
      t.classList.remove('badge-tab-hover');
    });
  }

  /**
   * Add a hover-highlight class to any SVG arrow path whose source badge
   * contains fromId AND whose target badge contains toId.
   * @param {string} fromId  source node ID
   * @param {string} toId    target node ID
   */
  highlightBadgeArrow(fromId, toId) {
    this.clearBadgeArrowHighlight();
    if (!fromId || !toId) return;
    var hlColor = getComputedStyle(document.documentElement)
                    .getPropertyValue('--highlight').trim() || '#e6820a';
    this._tabPanesEl.querySelectorAll('path[data-from-ids][data-to-ids]').forEach(function(path) {
      var froms = path.dataset.fromIds ? path.dataset.fromIds.split(',') : [];
      var tos   = path.dataset.toIds   ? path.dataset.toIds.split(',')   : [];
      if (froms.indexOf(fromId) !== -1 && tos.indexOf(toId) !== -1) {
        path.classList.add('badge-arrow-hover');
        // Update the arrowhead marker fill — CSS can't reach into <marker> defs
        // from the referencing path, so we do it directly in JS.
        var mUrl = path.getAttribute('marker-end') || '';
        var mId  = mUrl.replace(/^url\(#(.+)\)$/, '$1');
        if (mId) {
          var markerEl = document.getElementById(mId);
          if (markerEl) {
            var mTip = markerEl.querySelector('.badge-arrow-marker-tip');
            if (mTip) mTip.setAttribute('fill', hlColor);
          }
        }
      }
    });
  }

  /** Remove the hover-highlight class from all arrow paths and restore marker fills. */
  clearBadgeArrowHighlight() {
    this._tabPanesEl.querySelectorAll('path.badge-arrow-hover').forEach(function(p) {
      var mUrl = p.getAttribute('marker-end') || '';
      var mId  = mUrl.replace(/^url\(#(.+)\)$/, '$1');
      if (mId) {
        var markerEl = document.getElementById(mId);
        if (markerEl) {
          var mTip = markerEl.querySelector('.badge-arrow-marker-tip');
          if (mTip) mTip.setAttribute('fill', '#e6820a');
        }
      }
      p.classList.remove('badge-arrow-hover');
    });
  }

  /**
   * Compute badge data from badgeNodes and defer DOM insertion by
   * BADGE_PLACEMENT_DELAY_MS (giving the pane time to render first).
   *
   * @param {Array} badgeNodes  [{data:{type, file, line, id, ...}}] from /graph.
   */
  placeBadges(badgeNodes) {
    var badges      = {};   // "file:line" → {count, hiddenCount, entries, file, line}
    var traceBadges = {};   // traceKey    → {count, hiddenCount, entries, range}
    var tlr         = this._traceLineRanges;

    // Map node id → minimum total_order (for arrow sequencing).
    var orderMap = {};
    badgeNodes.forEach(function(node) {
      var orders = node.data.total_orders || [node.data.total_order];
      var min = Infinity;
      orders.forEach(function(o) { if (o != null && o < min) min = o; });
      if (isFinite(min)) orderMap[node.data.id] = min;
    });

    badgeNodes.forEach(function(node) {
      var iters    = node.data.total_orders ? node.data.total_orders.length : 1;
      var isHidden = !!node.data.hidden;

      // ── Source-file badges ────────────────────────────────────────────────
      if ((node.data.type === 'skeleton_call' ||
           node.data.type === 'external'      ||
           node.data.type === 'region')
          && node.data.file && node.data.line !== -1) {
        var srcKey = node.data.file + ':' + node.data.line;
        if (!badges[srcKey])
          badges[srcKey] = { count: 0, hiddenCount: 0, entries: [],
                             file: node.data.file, line: node.data.line };
        badges[srcKey].count += iters;
        if (isHidden) badges[srcKey].hiddenCount += iters;
        for (var i = 0; i < iters; i++)
          badges[srcKey].entries.push({ id: node.data.id, iterIdx: i });
      }

      // ── Trace-file badges ─────────────────────────────────────────────────
      var traceIdxList = node.data.trace_indices || [node.data.trace_index];
      traceIdxList.forEach(function(traceKey, i) {
        var range = tlr[traceKey];
        if (!traceBadges[traceKey])
          traceBadges[traceKey] = { count: 0, hiddenCount: 0, entries: [], range: range };
        traceBadges[traceKey].count += 1;
        if (isHidden) traceBadges[traceKey].hiddenCount += 1;
        traceBadges[traceKey].entries.push({ id: node.data.id, iterIdx: i });
      });
    });

    var self = this;
    setTimeout(function() {
      self.resetBadgeCycleIndex();
      self._placeBadgesInDOM(badges, traceBadges, orderMap);
    }, SourceCodeView.BADGE_PLACEMENT_DELAY_MS);
  }

  /**
   * Reset per-badge cycle counters.  Called automatically by placeBadges; also
   * available for external callers (e.g. after a graph re-render).
   */
  resetBadgeCycleIndex() {
    this._badgeCycleIndex = {};
  }

  /**
   * Handle a gutter badge click.  Advances the cycle index for the badge group,
   * then fires onBadgeClick with the resolved focus node and full ID set.
   *
   * @param {Array} entries  [{id, iterIdx}, …]  — same array embedded in the HTML
   */
  badgeClick(entries) {
    var key = entries[0].id;
    if (!this._badgeCycleIndex.hasOwnProperty(key)) this._badgeCycleIndex[key] = 0;
    else this._badgeCycleIndex[key] = (this._badgeCycleIndex[key] + 1) % entries.length;

    var focusEntry = entries[this._badgeCycleIndex[key]];
    var focusId    = focusEntry.id;

    var seenIds = {}, allIds = [];
    entries.forEach(function(e) {
      if (!seenIds[e.id]) { seenIds[e.id] = true; allIds.push(e.id); }
    });

    // In provider mode: broadcast via the shared event bus so every attached view
    // (GraphView, TimelineView, …) can update itself.  The 'code-listing' source
    // tag prevents this SourceCodeView from receiving its own event back.
    // In standalone mode: fall back to the onBadgeClick callback.
    if (this._attachedProvider) {
      this._attachedProvider._emit('select', {
        nodeId:  focusId,
        source:  'code-listing',
        focusId: focusId,
        allIds:  allIds,
        iterIdx: focusEntry.iterIdx,
      });
    } else if (this._onBadgeClick) {
      this._onBadgeClick(focusId, focusEntry.iterIdx, allIds);
    }
  }

  // ── Private badge helpers ────────────────────────────────────────────────────

  _buildBadgeHTML(count, hiddenCount, entriesJson) {
    var visCount = count - hiddenCount;
    // Embed entries as a data attribute; the event listener in the constructor
    // delegates clicks to this.badgeClick() — no global window.badgeClick needed.
    // JSON uses double quotes so a single-quoted attribute is safe; escape any
    // stray single quotes defensively.
    var attr = " data-badge-entries='" + entriesJson.replace(/'/g, '&#39;') + "'";
    if (hiddenCount === 0)
      return "<span class='badge'" + attr + ">" + count + "</span>";
    if (visCount === 0)
      return "<span class='badge' style='opacity:0.35'" + attr + ">" + count + "</span>";
    return "<span class='badge'"                      + attr + ">" + visCount   + "</span>"
         + "<span class='badge' style='opacity:0.35'" + attr + ">+" + hiddenCount + "</span>";
  }

  _placeBadgesInDOM(badges, traceBadges, orderMap) {
    var self = this;
    orderMap = orderMap || {};

    // Build pane lookup once — O(panes) instead of O(badges × panes).
    var paneByFile = {};
    self._tabPanesEl.querySelectorAll('.cpp-tab-pane[data-filename]').forEach(function(pane) {
      paneByFile[pane.dataset.filename] = pane;
    });

    // Lazily build lineNumber → gutter-cell maps per pane.
    var lineMaps = {};
    function getLineMap(key, pane) {
      if (lineMaps[key]) return lineMaps[key];
      var map = {};
      pane.querySelectorAll('.hljs-ln-n[data-line-number]').forEach(function(el) {
        map[el.dataset.lineNumber] = el;
      });
      return (lineMaps[key] = map);
    }

    // Per-pane arrow data: pane → [{gutterEl, totalOrder, nodeIds}]
    var paneArrows = new Map();

    // nodeId → gutterEl (for badge hover lookup).
    self._badgeNodeMap = new Map();

    // ── Source-file badges ────────────────────────────────────────────────────
    for (var key in badges) {
      var b        = badges[key];
      var basename = b.file ? b.file.split('/').pop().split('\\').pop() : '';
      var pane     = paneByFile[basename];
      if (!pane) continue;
      var el = getLineMap(basename, pane)[b.line];
      if (!el) continue;
      el.innerHTML = self._buildBadgeHTML(b.count, b.hiddenCount, JSON.stringify(b.entries));

      // Collect unique node IDs for this badge position.
      var nodeIds = [];
      var seen = {};
      b.entries.forEach(function(entry) {
        if (!seen[entry.id]) { seen[entry.id] = true; nodeIds.push(entry.id); }
        self._badgeNodeMap.set(entry.id, el);
      });

      // Determine minimum total_order for this badge position.
      var minOrder = Infinity;
      b.entries.forEach(function(entry) {
        var o = orderMap[entry.id];
        if (o != null && o < minOrder) minOrder = o;
      });
      if (isFinite(minOrder)) {
        if (!paneArrows.has(pane)) paneArrows.set(pane, []);
        paneArrows.get(pane).push({ gutterEl: el, totalOrder: minOrder, nodeIds: nodeIds });
      }
    }

    // ── Trace-file badges ─────────────────────────────────────────────────────
    var tracePane = self._tabPanesEl.querySelector('.cpp-tab-pane[data-trace-pane="true"]');
    if (tracePane) {
      var traceLineMap = getLineMap('__trace__', tracePane);
      for (var tkey in traceBadges) {
        var tb = traceBadges[tkey];
        if (!tb.range) continue;
        var tel = traceLineMap[tb.range[0]];
        if (tel) tel.innerHTML = self._buildBadgeHTML(tb.count, tb.hiddenCount,
                                                       JSON.stringify(tb.entries));
      }
    }

    // ── Store arrow data on panes; draw for the active one ────────────────────
    paneArrows.forEach(function(arrowData, pane) {
      pane._badgeArrowData = arrowData;
    });

    requestAnimationFrame(function() {
      var activePane = self._tabPanesEl.querySelector('.cpp-tab-pane.active');
      if (activePane && activePane._badgeArrowData) {
        self._drawBadgeArrows(activePane, activePane._badgeArrowData);
      }
    });
  }

  // ── Badge arrow overlay ──────────────────────────────────────────────────────

  /**
   * Draw SVG Bézier arrows between source-file badges in `pane`, ordered by
   * total_order.  The SVG is an in-flow zero-height element at the top of the
   * pane so it scrolls with the content without needing scroll-event updates.
   *
   * @param {HTMLElement} pane       A .cpp-tab-pane element.
   * @param {Array}       arrowData  [{gutterEl, totalOrder}, …]
   */
  _drawBadgeArrows(pane, arrowData) {
    // Remove any previous overlay for this pane.
    pane.querySelectorAll('.badge-arrow-svg').forEach(function(el) { el.remove(); });

    if (!this._codePaths) return;
    if (!arrowData || arrowData.length < 2) return;

    // Sort a copy by total_order.
    var sorted = arrowData.slice().sort(function(a, b) { return a.totalOrder - b.totalOrder; });

    // Unique marker id per SVG instance to avoid duplicate-id collisions.
    var markerId = 'bah-' + (++SourceCodeView._arrowCounter);

    var svgNS = 'http://www.w3.org/2000/svg';
    var svg   = document.createElementNS(svgNS, 'svg');
    svg.classList.add('badge-arrow-svg');
    svg.setAttribute('aria-hidden', 'true');
    // Absolutely positioned inside the <pre> element (its natural containing block
    // after we set position:relative below).  The <pre> scrolls with the pane, so
    // the SVG scrolls with it — no scroll-event recomputation needed.
    svg.style.cssText =
      'position:absolute;top:0;left:0;width:100%;height:100%;' +
      'overflow:visible;pointer-events:none;z-index:2;';

    // Arrowhead marker.
    var defs   = document.createElementNS(svgNS, 'defs');
    var marker = document.createElementNS(svgNS, 'marker');
    marker.id = markerId;
    marker.setAttribute('viewBox',      '0 0 8 8');
    marker.setAttribute('refX',         '7');
    marker.setAttribute('refY',         '4');
    marker.setAttribute('markerWidth',  '6');
    marker.setAttribute('markerHeight', '6');
    marker.setAttribute('orient',       'auto');
    var tip = document.createElementNS(svgNS, 'path');
    tip.setAttribute('d',    'M0,0 L0,8 L7,4 z');  // full-height base = wider head
    tip.setAttribute('fill', '#e6820a');
    tip.classList.add('badge-arrow-marker-tip');   // targeted by highlight helpers
    marker.appendChild(tip);
    defs.appendChild(marker);
    svg.appendChild(defs);

    // Anchor the SVG inside the <code> element.  The <code> is in the <pre>'s
    // scroll content (not a scroll container itself), so it moves with every
    // scroll direction — horizontal (pre scrolls) and vertical (pane scrolls) —
    // carrying the absolutely-positioned SVG with it.
    //
    // Why NOT <pre>: <pre> is the horizontal scroll container (UA overflow:auto).
    // position:absolute inside a scroll container is pinned to the container's
    // viewport edge, not its content, causing the paths to stay fixed while the
    // badges scroll away.
    //
    // Fallback chain: code → pre → pane.
    var container = pane.querySelector('code') ||
                    pane.querySelector('pre')  ||
                    pane;
    // Block display ensures getBoundingClientRect() reports the full content
    // dimensions (the hljs table stretches the code element to its natural size).
    container.style.display  = 'block';
    container.style.position = 'relative';
    container.appendChild(svg);

    // Coordinates are in the container's LOCAL space.
    // getBoundingClientRect() differences are scroll-independent: when any
    // ancestor scrolls, both the container rect and the badge rects shift by
    // the same viewport amount, so their difference stays constant.
    var containerRect = container.getBoundingClientRect();

    for (var i = 0; i < sorted.length - 1; i++) {
      var from = sorted[i];
      var to   = sorted[i + 1];

      // Connect to the right edge of the badge pill (not the gutter cell, which
      // is narrower — the pill overhangs it via position:relative; right:-1.75em).
      var fromBadge = from.gutterEl.querySelector('.badge') || from.gutterEl;
      var toBadge   = to.gutterEl.querySelector('.badge')   || to.gutterEl;
      var fromRect  = fromBadge.getBoundingClientRect();
      var toRect    = toBadge.getBoundingClientRect();

      // Coordinates relative to container origin.
      var x1 = fromRect.right  - containerRect.left;
      var y1 = (fromRect.top  + fromRect.bottom)  / 2 - containerRect.top;
      var x2 = toRect.right    - containerRect.left;
      var y2 = (toRect.top    + toRect.bottom)    / 2 - containerRect.top;

      if (y1 === y2) continue;   // skip degenerate same-position pairs

      // Bézier control points bend rightward into the code area.
      // CP1 leaves horizontally; CP2 arrives from the near-side at ~45° so the
      // arrowhead points correctly into the badge regardless of direction:
      //   downward arrow (dy > 0): CP2 above endpoint  → arrives from upper-right
      //   upward   arrow (dy < 0): CP2 below endpoint  → arrives from lower-right
      var dy    = y2 - y1;
      var bend  = Math.min(28, Math.max(10, Math.abs(dy) * 0.38));
      var cp2y  = y2 - Math.sign(dy) * bend;   // flip side for upward arrows

      var path = document.createElementNS(svgNS, 'path');
      path.setAttribute('d',
        'M' + x1 + ',' + y1 +
        ' C' + (x1 + bend) + ',' + y1 +
        ' ' + (x2 + bend) + ',' + cp2y +
        ' ' + x2 + ',' + y2);
      path.setAttribute('stroke',       '#e6820a');
      path.setAttribute('stroke-width', '1.5');
      path.setAttribute('fill',         'none');
      path.setAttribute('opacity',      '0.6');
      path.setAttribute('marker-end',   'url(#' + markerId + ')');
      if (from.nodeIds) path.dataset.fromIds = from.nodeIds.join(',');
      if (to.nodeIds)   path.dataset.toIds   = to.nodeIds.join(',');
      svg.appendChild(path);
    }
  }
}

// ── Module-level defaults ─────────────────────────────────────────────────────
// Downstream modules (graph.js) overwrite these at initialisation time with the
// canonical values from state.js.  The defaults match those values so the view
// is functional even when used standalone.
SourceCodeView.BADGE_PLACEMENT_DELAY_MS = 200;  // ms before badge DOM insertion
SourceCodeView.ANIM_SCROLL_MS           = 400;  // ms for animated line scrolling
SourceCodeView._arrowCounter            = 0;    // unique marker-id counter
