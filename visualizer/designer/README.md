# SkeVU Designer (proof of concept)

The reverse of tracing: draw a graph of containers and skeletons, and get the program. Open `designer.html` next to
`index.html` (any static server, e.g. `python3 -m http.server` in `visualizer/`).

## Editor

Graph, legend and code view are those of SkeVU (`script/graph_style.js`, `LegendView`, `SourceCodeView`), and so are the menus, resizers,
pan and zoom, and appearance settings. Selecting in the graph scrolls the code, and the other way round. The properties of the selected
node or edge are in the pane below the code.

* **Modes** (badge in the menu bar; click for the next): Explore (select, move), Add (the checked type is what a click on the canvas adds;
  `A`), Connect (`C`: data to a skeleton input, a skeleton to data, a skeleton to a skeleton, or data to data for updates of one object).
  `Esc` returns to Explore.
* **Regions** group calls (`SKEPU_TRACE_SCOPE`): Add › Region, Connect to a call, or Edit › Group Nodes (`G`). A region that repeats
  its calls is a loop (`L`, or the Loop property).
* **Undo/redo** (`⌘Z`, `⇧⌘Z`; `history.js`): a step is the design before a change; the last 100 are kept.
* **Edit › Duplicate** (`D`) copies the selected nodes; a copied call writes new data of its own.
* **Data nodes are updates.** A node is one state of an object: a call writes an update, and an update has one writer. Nodes with the
  same name are one object (one declaration, one allocation, one format); updates are numbered in subscript. A node nothing writes is the
  allocation. Giving a node another object's name, or connecting two data nodes, joins them; Detach separates.
* **Order of calls** follows the data flow; where that leaves a choice, the vertical position decides, with a warning when the values
  depend on it. View › Arrange lays the graph out from the data flow.
* **Data kinds**: Vector, Matrix, Tensor3, Tensor4 (the Kind property). Formats of data written by a call are inferred from the call
  (`inferOutput`) unless set by hand.
* **Other nodes**: External (host code, with a body per language), Random Stream (`skepu::PRNG`), I/O (prints a connected container or a
  reduction result; `skepu::io::cout` / `print`).
* **Transfers** between host and device calls are drawn as in SkeVU; they are derived from the design (`model.js`: `transfers`).
* **Generate** (Design › Generate) makes a random design: size, arity, shape and seed (same settings and seed give the same design).

## Code

* `model.js`: the design is plain JSON; `lower()` checks it and makes the program (names, ordered steps, user function stubs).
* `codegen.js`: `emitCpp()` and `emitPython()` (SkePU C++ for the precompiler, SkePY). Each line knows its design nodes; select a node to
  see its lines, click a line to select its node.
* `trace.js`: the third code tab is the trace that the program could have (`TRACE_FORMAT.md`), made from a model of SkePU, not by
  running it. Order, ids, labels, sizes and transfers follow the design; the times are made up. Long loops are cut.
* Code menu: User Functions (free-standing or lambdas), Container Placement and Skeleton Placement (grouped or before first use),
  Whitespace (balanced, readable, compact). The command line tools take the same options (`--lambdas`, `--first-use`,
  `--skeletons-first-use`, `--whitespace=...`).

## Command line

* `node designer/generate.js`: prints a program from a design file.
* `node designer/roundtrip.mjs --all`: builds the C++ program with the precompiler, runs it on each backend and compares the trace with
  the design (`compare.js`).
* `node designer/skepy_compare.mjs --count 5 --size medium`: builds random designs as C++ and as SkePY Python (needs `shedskin/` and
  Python 3.14), runs both with the sequential backend and compares the output.

Open: control flow other than loops, merging user function bodies when the design changes, scalars between skeletons, the fusion forms
(`>>`, `||`).
