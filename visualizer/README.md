# SkeVU — SkePU Visualizer

Browser-based DAG visualizer for SkePU trace files.
Renders task graphs, Gantt timelines, and annotated source views from a JSON
trace produced by an instrumented SkePU program.

See **[USER_MANUAL.md](USER_MANUAL.md)** for the full interface guide.

---

## Setup

Install test dependencies (only needed to run the test suite):

```bash
npm install
```

No build step is required — the app runs directly from source.

## Running

The visualizer is a fully static web app.  Serve the repository root over HTTP
and open `index.html` in a browser.  Any static file server works:

```bash
# Python (no extra install needed)
python3 -m http.server 5001
# → open http://localhost:5001

# or Node
npx serve -l 5001
# → open http://localhost:5001
```

Upload a trace file via the file picker, or load one of the built-in examples
from the **Files** menu.

> **Why HTTP?**  The visualizer spawns Web Workers for layout and graph
> processing.  Browsers block Worker creation from `file://` URLs, so a local
> HTTP server is required.

---

## File structure

```
index.html              upload / landing page
main.html               visualizer UI

script/
  ── View layer ──────────────────────────────────────────────────────────────
  GraphView.js          Cytoscape graph renderer
  LegendView.js         the legend of a graph (node and edge kinds; the spectrum, or the skeletons or backends that the calls are coloured by)
  TimelineView.js       Gantt / timeline canvas chart
  InfoView.js           node info pane
  SourceCodeView.js     annotated source code browser (line highlighting and smooth scrolling, badges, code paths)

  ── Shared by SkeVU and the Designer (designer.html) ────────────────────────
  graph_style.js        the Cytoscape style of the graph, and its colours (COLORS, regionColor)
  graph_interaction.js  the selection (what is marked) and the hover of a graph
  panzoom.js            pan by the wheel, zoom by the pinch, and the smooth pan and zoom to nodes
  tooltip.js            the tooltip of a node or an edge, and its text
  menu.js               the menus of the menu bar (open, close, hover; items that would do nothing are disabled)
  resizer.js            the bars between panes: pointer and keyboard
  appearance.js         dark mode, code theme and colour spectrum: what is remembered, and their menus
  early.js              classic script for <head>: the dark mode before the first paint
  format.js             numbers for people (durations)

  ── Application layer ───────────────────────────────────────────────────────
  graph.js              wires GraphView, TimelineView, LegendView, SimulationEngine
  ── Backend (Web Worker) ────────────────────────────────────────────────────
  backend/
    visualizer.js       Worker entry point — handles load / graph / getData
    graph.js            DirectedGraph construction from trace events
    node.js             node type hierarchy
    edge.js             edge type hierarchy

  ── Layout workers ──────────────────────────────────────────────────────────
  workers/
    worker-loader.js    spawns the visualizer worker, wires transport.js
    dagre-worker.js     off-thread Dagre layout
    elk-worker.js       off-thread ELK layout

  static/               vendored third-party libraries (Cytoscape, Dagre, ELK, Highlight.js and the themes that the menu offers)

style/                  CSS

tests/
  helpers.js            shared fixtures and graph builders
  setup.js              vitest environment setup
  unit/
    test_construction.js  graph construction from trace events
    test_serial_fusion.js findSerialFusions() algorithm
    test_cytoscape.js     toCytoscape() and infoData() schema contracts

examples/               built-in example trace files
```

---

## Tests

Tests use **[Vitest](https://vitest.dev)** and cover the JS backend modules.

```bash
npm test            # run all tests once
npm run test:watch  # re-run on file changes
```

### What each module covers

**`test_construction.js`** — Verifies that `DirectedGraph.fromEvents()` correctly
parses every event type into the right node and edge objects, that edges are
wired bidirectionally, and that `GraphSettings` flags (updates, allocations,
deallocations, …) gate the correct node types.  Also runs the full pipeline
(depths → critical path → `toCytoscape`) over every shipped example trace as a
smoke test.

**`test_serial_fusion.js`** — Targets `findSerialFusions()`, the greedy
backward traversal that identifies chains of skeletons whose intermediate
results have a single consumer.  Tests cover the happy path (3-Map chain
producing two fusion pairs sharing one `FusedNode`), fan-out blocking (shared
intermediate prevents fusion), cross-pattern detection (Map→Reduce), and the
structural reason the second seeding pass is needed for reduction sinks.

**`test_cytoscape.js`** — Schema regression tests for every node and edge type.
Pins the exact set of keys emitted by `toCytoscape()` and `infoData()` so that
a renamed field, dropped `super()` call, or missing type-specific key surfaces
as an explicit failure rather than silent frontend breakage.

---

## Optional: SkePU library headers

The source pane can show SkePU library headers alongside user code.  Copy them
in with:

```bash
make skepu-lib   # requires ../skepu-headers/ checked out next to this repo
```
