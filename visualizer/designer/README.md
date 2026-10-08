# SkeVU Designer (proof of concept)

The reverse of tracing: draw a graph of containers and skeletons, and get the program. Open `designer.html` next to
`index.html` (any static server: `python3 -m http.server` in `visualizer/`).

* **Editor** (`designer.html`, `editor.js`, and its parts `state.js`, `graphview.js`, `codeview.js`, `status.js`, `inspector.js`, `actions.js`, `menus.js`: the design is the state, `changed()` says that it changed, and each part registers what it does about that): the page is built from the parts of SkeVU: its layout, menus and info pane
  (`style/main.css`), its graph style (`script/graph_style.js`, shared with `GraphView`: data nodes, skeleton calls coloured by
  skeleton or backend, regions for loops, dashed element-wise and solid proxy edges), its legend, and its source view
  (`SourceCodeView`: a tab for each program, line numbers, gutter badges with how many times a call runs, code paths between the
  calls). Properties of the selected node or edge are in the pane below the code, always visible. Sizes, element types and
  parameters are properties, not part of the drawing (yet).
  * **Modes**, shown by a badge at the right edge of the menu bar (click it for the next mode: Explore, Add, Connect, and round again; `A` and `C` toggle the Add and Connect modes): *Explore* (select and move), *Add* and *Connect*. In the **Add** menu each node type is a
    mode: the checked type is what a click on the canvas adds (a click inside a region adds to it); `A` turns the Add mode on and off,
    going back to the type that was added last. **Connect** (`C`) two nodes: data to a skeleton (an input), a skeleton to data (its
    output), **a skeleton to a skeleton**, which makes the data between them, or **data to data** (updates of the same data, see below). `Esc` is back to Explore.
  * **Regions** group calls (a trace region, `SKEPU_TRACE_SCOPE`): the *Region* type of the Add mode makes an empty one, and *Connect* between it and a
    call (or another region) puts the call in it; or select several nodes and *Edit › Group Nodes* (`G`). A region that repeats its
    calls is a **loop** (*Loop Around*, `L`, or the *Loop* property of a region): `for` and a scope for each iteration; a plain
    region is a block with a scope, run once. *Ungroup* takes nodes out of their region, or removes the selected region. The Edit
    menu has an item enabled only when it would do something now.
  * **Undo and redo** (*Edit › Undo*, `⌘Z` / `Ctrl+Z`; *Redo*, `⇧⌘Z` / `Ctrl+Y`; `history.js`): the design is all of the state, so a step is the design
    before a change (the last 100 are kept, in memory). Every change of the design is a step (a node added, a connection, a property, *Arrange*,
    *Generate*, *New*, *Open*), but not every event: a text field makes its change when it is left, not for each key; a drag when it ends, not
    for each pixel, and the nodes of one drag are one step; the keys that are typed in the body of a function are one step (until a pause of a
    second); and what one action does in several calls is one step. In a text field the keys are the browser's own undo. What comes back after an
    undo is selected.
  * **Edit › Duplicate** (`D`) copies the selected nodes (a region brings what is in it) and selects the copies. What is between copied
    nodes is between the copies; a copied call reads what the original reads (shared data, shared random streams) and writes new data
    of its own in the format of the original's, so that two calls do not write the same data.
  * **Data nodes are updates, as in SkeVU.** A data node is one state of an *object*, not an object of its own: a call writes an update, and an
    update has one writer (two writers of one node is a problem). Nodes with the same name are the same object (a hidden key in the design
    JSON, `object`): one declaration in the program, one allocation in the trace, one format and one place for the transfers, and each
    update has its number as a subscript (`values₁`, `values₂`, none for the only one). A node that nothing writes is the allocation of the
    object. A new node is an object of its own; it becomes an update by **giving it a name that another object has** (the *Name* property:
    renaming an object renames all its nodes, a name that exists joins it) or by **Connect between two data nodes** (the first one's name).
    *Detach* (properties) makes a node an object of its own again. The name, format and initial value are those of the object (they are
    edited in any of its nodes); a random stream and data are never one object.
  * **The data that a call writes is in the region of the call** (it is drawn inside it and is not set separately), so moving a call
    into a region moves its output; a region with no calls has a warning.
  * The calls are made **in the order of the
    data flow** (an update is written before it is read; the updates of an object come in the order of their writers, and what reads an update
    comes before the write of the next one); **where that leaves a choice, the place from the top decides**. When the data flow does not order
    two updates of an object, their places decide and there is a **warning**: the values depend on it (make one call depend on the other).
    The calls of a region stay together; a cycle of dependences is ordered by place, with a warning. **View › Arrange** lays the graph out from the data flow (and **View › Reset View** fits it).
* **Generate** (`random.js`): *Design › Generate* replaces the design with a random one, as the *Generate* menu says: **size** (small,
  medium, large: about 12, 40 and 100 nodes), **arity** (how many inputs, proxies and uniforms the calls have, and how much the
  externals read and write), **shape** (deep: each call reads what the last ones wrote, wide: many sources and data that is read again
  and again) and a **seed** (the same settings and seed make the same design; empty: a seed is made, and shown in a message in the menu bar and in
  the name of the design). The settings bias the choices; they do not fix them. The designs have no problems: every skeleton, externals,
  random streams, loops and regions (nested) are used, some calls have a backend of their own, and the result is arranged. As SkePU programs do, about a quarter of the calls write a new update of data that exists, under the same name (in place, for a Map; or data that the call depends on), and now and then the calls that draw random numbers (Map, MapReduce, MapOverlap, MapPairs, MapPairsReduce) are a chain on one stream.
* **Menu bar**: the *file name* (the name of the design and of the program files) is the first item of the *Code* menu. The statistics
  of the design (data, streams, skeletons, externals, loops, regions, edges, problems, warnings, events of the trace) are in the ⓘ menu,
  as the statistics of a trace are in SkeVU; what the editor says for a few seconds (an error, a result) is a message beside it.
  The splash screen of SkeVU links to the designer (*Design a program*).
* **Shared with SkeVU** (`script/`, see the README there): the graph style and its colours, the legend (`LegendView`: for calls coloured by
  skeleton or by backend it shows the ones that the graph has, as SkeVU does), the hover (tooltip, legend entry, badge), the selection marks,
  the menus (open, close, disabled items), the resizers (pointer and keyboard), the pan and zoom, and the dark mode, code theme and colour
  spectrum (View › Color Spectrum, View › Source Code Theme, View › Dark Mode), which are remembered in the same entry as SkeVU's.
* **Pan and zoom** are those of the graph of SkeVU (`script/panzoom.js`, which `GraphView` uses too): the wheel or a two-finger swipe pans, the pinch zooms around the pointer, zoom from 0.01 to 5.
* **Graph and code follow each other**, as in SkeVU: choosing a node or an edge in the graph makes the code scroll, smoothly, to put the
  place of it at the top (for a call, the line where it is called; for data, its declaration; for the trace tab, its first event); choosing
  something in the code (a line, a badge, a problem) makes the graph pan and zoom, smoothly, to it (a node with room around it, a region or
  several nodes fitted). The one that was clicked stays where it is, and what puts the selection back after an edit moves nothing.
* **Data kinds**: Vector, Matrix and the 3D and 4D tensors (`Tensor3`, `Tensor4`: *Add › Tensor (3D)* / *Tensor (4D)*; the *Kind* property switches between them and gives the size of each dimension). A MapOverlap has 1 to 4 dimensions (a radius for each; `Region1D` to `Region4D` in its function) and is on the kind of that rank; an indexed Map has an `Index1D` to `Index4D`; a proxy is a `Vec`, `Mat`, `Ten3` or `Ten4`. The *Add* menu has the data, the skeletons, *External* (host code) and *Region* in sections of their own.
* **I/O** (`Add › I/O`, in the section of *External*): a block that prints, drawn as an external block as in SkeVU, where `skepu::io::cout << container`
  is an external block named "I/O" that reads the container. Connect the data to print to it (it is host code: a read of what a device call wrote is
  a transfer first), and a Reduce or MapReduce to print its result (a dotted scalar edge). In C++ it is `skepu::io::cout << data << "\n"` for each
  (`print(data)` in Python), and the trace has an I/O event for each container that it prints. A result that no I/O block prints is printed by the call
  that makes it, as before. Generate makes I/O blocks for the data and for the results of reductions.
* **Other nodes**: *External* (`Add › External`) is host code (`skepu::external` / `with skepu.external(...)`): data connected into
  it is what it reads, data connected out of it is what it writes, and its body is a property (per language). It runs on the
  host. A *Random Stream* (`Add › Random Stream`, a `skepu::PRNG`) is a node like data, drawn as in SkeVU (the allocation of a stream that calls draw from is a violet hexagon): it is not updated like data: it has one edge out, to the first call that draws from it, and the next call is connected from the one before it (a random-dependence edge between the two calls, as in SkeVU), so the calls that share a stream are a chain in the order of the chain. Map, MapReduce, MapOverlap, MapPairs and MapPairsReduce can draw random numbers (a `Random` parameter after the index, in C++ and in Python); Reduce and Scan cannot. Connecting from a stream that has an edge already connects from the last call of its chain; deleting a call from the middle of a chain joins the chain past it
  (they share its state); the seed is a property. Data that nothing writes is drawn as an *allocation* (the star of SkeVU), data
  that a call writes as a data node.
* **Transfers** (`model.js`: `transfers`): where two calls on different explicit backends meet at some data, one on the host (CPU,
  OpenMP) and one on the device (OpenCL), the graph shows a transfer node (as SkeVU does) between the data and the call that
  needs it moved, with the direction. The transfer is where the data is anchored on that side: every call on that memory space that reads the copy (an
  external block is on the host) is connected from the transfer, not from the data, until the data is written again. Made from the design, not part of it; it is a node like the others, as in SkeVU (it starts between its ends and is where it is moved or laid out after that: nothing keeps it between them); calls on the backend that the program is started with
  are not followed. Loops are followed twice, for what moves from one iteration to the next. Externals run on the host.
* **Inferred formats** (`model.js`: `inferOutput`, `effectiveContainers`): the data that a call writes can have its kind, element
  type and size from the call (Map and Scan: the first input; MapOverlap: its input; MapPairs: a matrix of the lengths of its two
  inputs; MapPairsReduce: a vector as long as the vertical input, or as the horizontal one when it is reduced along the columns). Data made by connecting two skeletons is `auto`: it follows its inputs, through a chain of calls, until it is set by
  hand (the *Format* checkbox in its properties). A reduction writes a scalar, and a Map of indices has no input to give a size.
* **Model** (`model.js`): the design is plain JSON; `lower()` checks it (number of inputs and outputs of each skeleton, sizes,
  types, loops whose calls are not consecutive) and makes the program: names, ordered steps, user functions with stubs.
* **Code generation** (`codegen.js`): `emitCpp()` and `emitPython()` (SkePU C++ for the precompiler, SkePY Python). Each line knows
  the design nodes that it is for: select a node to see its lines, click a line to select its node. `sites` are the lines of
  the calls and loops, for the badges.
* **Trace** (`trace.js`): the third tab of the code is the trace that the program could have (`TRACE_FORMAT.md`), made from a small
  model of SkePU and not by running it: the calls in order with their containers, elements and parameters, the transfers between host
  and device, the regions of the loops, the I/O of a printed reduction. The order, ids, labels, elements and transfers are what the
  design says; the times are made up (a rough cost of each backend; a call without a backend runs on the CPU). Long loops are cut to
  their first iterations (the name of the file says how many). Selecting a node marks its events, and a click on an event selects its
  node. `compareTrace` finds the calls of the design in it, and SkeVU can read it.
* **Code menu**: *User Functions* are **free-standing** (a function for each, before `main`: the default) or **lambdas** (in the
  declaration of the skeleton, `skepu::Map<1>([](float a) -> float { ... })`; a MapReduce has two). In Python a user function is a
  lambda (`skepu.Map(lambda a: a + 1)`) only when its body is one line, `return expression` (a comment or empty lines apart; a
  comment on the line makes it a function); the others stay functions. `emitCpp(program, { userFunctions: 'lambdas' })`,
  `generate.js --lambdas` and `roundtrip.mjs --lambdas` (which builds and runs the lambda form too).
  *Container Placement*: **grouped** (all the containers together before the skeletons: the default) or **before first use** (each
  container is declared just before the first step that uses it, in the block where all its uses are: not inside a loop, which would
  make it again for each iteration, but inside a plain region that has all its uses; random streams and data that no call uses stay at
  the start). The synthesized trace follows it: a container is allocated where it is declared and freed at the end of its block.
  `--first-use` for `generate.js` and `roundtrip.mjs`.
  *Skeleton Placement*: **grouped** (all the skeletons together before the calls: the default) or **before first use** (each skeleton is
  declared just before its call, not inside a loop but inside a plain region that has the call). `--skeletons-first-use` for
  `generate.js` and `roundtrip.mjs`. The files are made again only when something that they are made from changed (the design without the
  places of its nodes, the order of the calls, the options): moving a node does not make them again, so the code view keeps its place.
  *Whitespace*: **balanced** (the default), **readable** (a blank line between what comes from different nodes: the declarations of
  skeletons, the calls and the regions; the setters of a skeleton, the containers that are declared before the first use of a call, and
  the print of a result stay with it; no blank line after the opening of a block) and **compact** (no comments of its own: no note on
  top, no names of the sections, no note that a function is a stub, so an empty external has an empty body, or `pass`; what the user
  wrote stays). `--whitespace=...` for `generate.js` and `roundtrip.mjs`.
* **C++ against SkePY** (`skepy_compare.mjs`, `skepy_compare.py`): `node designer/skepy_compare.mjs --count 5 --size medium` writes the programs of
  random designs (or of design files) in C++ and in Python, with function bodies that compute something and an I/O block that prints all the
  data at the end, builds both with the sequential header-only SkePU (the C++ with `c++`, the Python with SkePY: `shedskin`, needs the SkePY
  repository in `shedskin/` and Python 3.14), runs them and compares what they print. Numbers are compared with a tolerance; SkePU and SkePY
  print a tensor differently, so only the numbers of tensors are compared. It found that a MapOverlap with the edge mode None was given an output
  of the size of its input, where SkePU wants it smaller by the overlap on each side (checked and inferred now).
* **Round trip** (`roundtrip.mjs`, `compare.js`): builds the C++ program with the precompiler, runs it on each backend and
  compares the trace with the design: `node designer/roundtrip.mjs --all` (or a design file). `generate.js` prints a program from
  a design file.

Decisions that the proof of concept leaves open: control flow other than loops, merging the user's function bodies when the design
changes (they are kept in the design, per language), scalars between skeletons (a reduction prints its result), host code,
the fusion forms (`>>`, `||`), and a preview of the trace of a design without building it.
