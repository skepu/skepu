# SkePU User Guide

The SkePU user guide, a [Quarto](https://quarto.org) book (HTML, with interactive skeleton
illustrations, SkeVU trace panels and runnable WebAssembly examples).

```bash
make          # render the book to _book/
make preview  # live-reloading server (`quarto preview`)
make viz      # (re)install the visualizer into _book/viz/ without rendering
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
bibliography.bib, *.csl   references
theme.scss, custom.css    look and layout
figures/                  diagrams (PDF for print, PNG)
visualizers/              skeleton access-pattern widgets, see below
demos/                    SkePU programs compiled to WebAssembly, see below
_extensions/skevu/        shortcodes trace-embed and trace-wasm, see below
scripts/install_viz.py    post-render step: installs ../visualizer as _book/viz/
```

## Skeleton widgets (`visualizers/`)

`explorer.html` (all skeletons, tabs) and one file per skeleton (`map.html`, ...) are
included into chapters with `{{< include ../visualizers/map.html >}}`. They share
`viz-shared.html` (`VizUtils`) and `viz-skeletons.html` (`SkePUSkeletons`, one entry per
skeleton), which the chapters using them list under `include-in-header` in their front matter.

## SkeVU panels and WebAssembly runners (`_extensions/skevu/`)

`trace-embed` shows a trace of the visualizer's examples, `trace-wasm` runs a WebAssembly
program in the page and shows the trace it writes; see the header of `skevu.lua` for the
parameters. `skevu.js` and `skevu.css` are added to the pages that use them.

The panels use the visualizer in `../visualizer` as it is, and keep no copy of it in this
directory: after each render `scripts/install_viz.py` (a Quarto post-render step) installs its
`script/`, `static/` and `examples/` as `_book/viz/`, and the shortcodes pass the relative path
to it to the panels. The visualizer does not depend on the book. A fixed trace is selected by
its index in `visualizer/examples/examples.json`.

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
