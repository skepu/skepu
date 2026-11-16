# SkePU User Guide

The SkePU user guide, a [Quarto](https://quarto.org) book (HTML, with interactive skeleton
illustrations, SkeVU trace panels and runnable WebAssembly examples).

```bash
make          # render the book to _book/
make preview  # live-reloading server (`quarto preview`)
make viz      # (re)install the visualizer into _book/viz/ without rendering
make simple   # the simple edition in _book-simple/: no SkeVU panels, no WebAssembly, no copy of the visualizer (what the SkeVU extension includes)
make pdf      # PDF of the book in _pdf/ (needs LaTeX; the interactive parts are left out)
make demos    # rebuild the WebAssembly demos (needs Emscripten)
make clean
```

Needs Quarto (≥ 1.4) and Python 3. The PDF also needs a LaTeX distribution with the packages
Quarto asks for (`lualatex-math`, `framed` and `xurl` were missing from BasicTeX). Serve `_book/` over HTTP to use it, for instance
`python3 -m http.server -d _book`: the panels use `fetch()` and WebAssembly, which browsers
block on `file://`.

## Layout

```
index.qmd, src/*.qmd      chapters (list in _quarto.yml)
bibliography.bib, ieee.csl references (the citation style is a third-party file, see below)
theme.scss, custom.css    look and layout
figures/                  diagrams (PDF for print, PNG)
demos/                    SkePU programs compiled to WebAssembly, see below
_extensions/skevu/        shortcodes trace-embed, trace-wasm and skeleton-viz, see below
scripts/install_viz.py    post-render step: installs ../visualizer as _book/viz/
scripts/build_listings.py pre-render step: builds the listings marked to run, see below
```

## Skeleton widgets (`_extensions/skevu/`)

`{{< skeleton-viz map >}}` shows the access pattern of a skeleton; `all`, or a list (`map,scan`), gives tabs.
`skeletons.js` is the registry (`SkePUSkeletons`, `SkePUEntries`): the drawing, the listing and the controls of each
skeleton. `skeleton-viz.js` builds the page from it.

## SkeVU panels and WebAssembly runners (`_extensions/skevu/`)

`trace-embed` shows a trace of the visualizer's examples, `trace-wasm` runs a WebAssembly
program in the page and shows the trace it writes; see the header of `skevu.lua` for the
parameters. `skevu.js` and `skevu.css` are added to the pages that use them.

The panels use the visualizer in `../visualizer` as it is, and keep no copy of it in this
directory: after each render `scripts/install_viz.py` (a Quarto post-render step) installs its
`script/`, `static/` and `examples/` as `_book/viz/`, and the shortcodes pass the relative path
to it to the panels. The visualizer does not depend on the book. A fixed trace is selected by
its index in `visualizer/examples/examples.json`.

## Runnable listings (`listings/`)

A C++ listing marked `{.cpp .run}` gets a runner after it. Attributes: `argv="10"` (the field of the arguments is shown only with it),
`views="graph,timeline"`, `label="..."`.

- `scripts/build_listings.py` (a pre-render step; `make listings` runs it by hand) stages each marked listing as `listings/<hash>.cpp` and
  builds it with the precompiler and Emscripten into `listings/wasm/`. Builds are kept by content: only changed listings are rebuilt.
- A listing that does not compile is reported with its file and line, and the render fails (`--keep-going`: only reported).
- Without Emscripten or a configured SkePU build system it warns and builds nothing. The simple edition and `make pdf` skip it (`SKEPU_LISTINGS=off`).
- `--annotate` marks the complete programs that print.
- `_extensions/skevu/listings.lua` adds the runners; a listing without a build has none.
- The output of a listing is a block after it, ` ```{.text .output} `. It is shown where there is no runner and left out where the runner shows the real output.
  The build runs each built listing under `node` (`scripts/run_wasm.js`) and warns, with a diff, when its output differs from the block. The block is updated by hand.

## Demos (`demos/`)

`trace-wasm` needs the Emscripten glue (`demos/wasm/NAME.js` and `.wasm`, committed so that the
book builds without Emscripten). To add one: put `NAME.cpp` in `demos/`, add it to `DEMOS` in
`demos/Makefile`, run `make demos` (needs `skepu-tool` and `em++`), then use

```
{{< trace-wasm js="../demos/wasm/NAME.js" src="demos/NAME.cpp" argv="10" views="graph,source" >}}
```

The program must be linked with the flags in the Makefile (`MODULARIZE`, `EXPORT_NAME`,
`callMain` and `FS` exported); tracing is enabled with `-DSKEPU_TRACING -DSKEPU_TRACE_FILE=trace.json`.
`views="none" code="true"` gives a plain runner with the source listing above it.

## Editing

Chapters are static Quarto Markdown (no code execution). Citations are `[@key]` from
`bibliography.bib`; a listing with a caption is a fenced block `{#lst:name .cpp}` preceded by a
`{.listing-caption}` div.

## Publishing

Not set up yet: `make` produces the complete site in `_book/`.

## Third-party files

`ieee.csl` is the IEEE citation style from the [Citation Style Language styles](https://github.com/citation-style-language/styles) collection, kept
unmodified (version of 29 November 2023, authors and contributors in the file). It is licensed under
[Creative Commons Attribution-ShareAlike 3.0](http://creativecommons.org/licenses/by-sa/3.0/), as the file says, and not under the SkePU license. It is a
local copy so that the book can be built without a network: a dependent style, such as the IEEE Access one that was used before, makes pandoc
download its parent from zotero.org on every build.
