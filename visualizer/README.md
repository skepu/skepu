# SkeVU — SkePU Visualizer

Browser-based DAG visualizer for SkePU trace files.
Renders task graphs, Gantt timelines, and annotated source views from a JSON
trace produced by an instrumented SkePU program.

The interface is explained by the guided tour (Help menu) and the shortcuts dialog (`?`).

---

## Setup

Install test dependencies (only needed to run the test suite):

```bash
npm install
```

No build step is needed for the web page; the VS Code extension has one (below).

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

While you edit SkeVU, the browser may keep serving old copies of the scripts and styles (`python3 -m http.server`
sends no cache headers, so the browser guesses). `serve.py` is the same server with `Cache-Control: no-store`:

```bash
python3 serve.py 5005        # from any directory; → open http://localhost:5005
```

Upload a trace file via the file picker, or load one of the built-in examples
from the **Files** menu.

> **Why HTTP?**  The visualizer spawns Web Workers for layout and graph
> processing.  Browsers block Worker creation from `file://` URLs, so a local
> HTTP server is required.

---

## File structure

```
index.html              the viewer
designer.html           the Designer

script/
  ── View layer ──────────────────────────────────────────────────────────────
  GraphView.js          Cytoscape graph renderer
  LegendView.js         the legend of a graph (node and edge kinds; the spectrum, or the skeletons or backends that the calls are coloured by)
  TimelineView.js       Gantt / timeline canvas chart
  InfoView.js           node info pane
  scalarplot.js         the plot of the values of a scalar by iteration (info pane)
  SourceCodeView.js     annotated source code browser (line highlighting and smooth scrolling, badges, code paths)

  ── Shared by SkeVU and the Designer (designer.html) ────────────────────────
  graph_style.js        the Cytoscape style of the graph, and its colours (COLORS, regionColor)
  graph_interaction.js  the selection (what is marked) and the hover of a graph
  panzoom.js            pan by the wheel or a two-finger swipe, zoom by the pinch (Safari) or Ctrl/Cmd+wheel (also the pinch in Chromium), and the smooth pan and zoom to nodes
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

examples/               built-in example trace files (the trace files of an MPI program: an array of paths; `cpp_files` and `snapshot_archive` can
                        have an entry for each backend; the backend pills show MPI last)
```

---

## The values of scalars

The trace has the value of every scalar that a skeleton returns (the result of a reduction: an error norm, a sum, a residual; `scalar_value` in
`TRACE_FORMAT.md`), written automatically. A scalar node has the value in its info pane (the **Value** row, and **Result of** names the call).
When iterations were coalesced into the node, the info pane plots the values by iteration (the x axis): a click on a point selects that iteration,
the hover shows its value, and a plot of positive numbers over three orders of magnitude or more (a convergence plot) starts with a logarithmic
y axis (the checkbox switches). The scalars that different calls make, and the results of one call that has several, are separate nodes and
separate series. (A value that is not finite leaves a gap.) A scalar node is in the graph when **View > Scalars** is on.

## Where the instances and user functions are defined

A trace of a program that the precompiler made has, in its header, a table of the skeleton instances with the lines of the instance and of
its user functions (see `TRACE_FORMAT.md`). SkeVU shows them for each call:

- **Info pane**: an `Instance` row and a `User function` row for each function, as links that show the lines in the source (a link to a
  file that is not open is grey).
- **Source panes**: when a call is selected, the lines of its instance and of its user functions are shaded in the colour of the selection,
  with less opacity than the line that is selected (the call). The shading fades out when the selection is cleared.
  The menu item **View > Definitions** keeps the shading of all the definitions.
- **From the source to the calls**: the line number of a line that defines an instance or a user function can be clicked (it is underlined):
  it selects the calls that use it, and the next click goes on to the next of them.
- **From a node to the source**: shift-click on a node of the graph shows the instance of the call, alt-click shows its user function (the next one
  each time, for a skeleton with more than one).

A lambda is a user function on the line of the instance, and an anonymous skeleton has its instance on the line of the call: the marks and the
badge of the line are then both there.

## VS Code extension

The viewer also runs in an editor: a custom editor ("SkeVU Visualizer") shows the trace in a tab, and the badges and highlights of its nodes are
in the source files, which are the ones that are open in the editor. It opens by itself for the files that the tracer names (`skepu_trace*.json`,
also the files of the processes of an MPI program), for those that the examples' Makefile names (`trace_*.json`) and for `*.skevu`; any other `*.json` file is opened with it by *Open With…*. Each source file
of the trace is looked for in the workspace and used only if its SHA-256 is the one in `header.files` of the trace (see TRACE_FORMAT.md).

```bash
npm install
npm run build      # dist/: the viewer and its two workers, bundled (esbuild.js)
npm run compile    # out/: the extension (src/, tsc)
```

To have it in every window, make a package and install it (`@vscode/vsce` is a dev dependency: `npm install` first):

```bash
make vsix                                              # npm run package, with the library headers in it
codium --install-extension skepu-visualizer-1.0.0.vsix  # or in the editor: Extensions, the ... menu, Install from VSIX...
```

To try it without installing, start an editor with it as a development extension (this is a window of its own, and the only one that has the
extension): `codium --extensionDevelopmentPath=$PWD <folder>`, or *Run Extension* (F5) in this folder. The page of the webview is made from `index.html` (src/webviewHtml.ts): the same markup as the
browser page, with a content security policy that runs only scripts with a nonce, so the controls are `data-click` / `data-input` (script/actions.js)
and not inline handlers (tests/unit/test_markup_csp.js checks it). What differs from the browser is in `script/*_vscode.js` and `skevu.js` (the
messages to and from the extension). The library headers (`static/skepu-lib`) are made by `make`, as for the browser page.

## Tests

Tests use [Vitest](https://vitest.dev), in `tests/unit/`.

```bash
npm test            # run all tests once
npm run test:watch  # re-run on file changes
```

---

## Optional: SkePU library headers

The source pane can show SkePU library headers alongside user code.  Copy them
in with:

```bash
make skepu-lib   # requires ../skepu-headers/ checked out next to this repo
```
