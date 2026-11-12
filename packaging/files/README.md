# SkePU package

A complete SkePU for this computer: the precompiler (`bin/skepu-tool`), the SkePU headers (`include/`), the language server (`bin/clangd`, if the package has it) and
`skepu.mk`, which builds programs. Nothing needs to be installed; the directory can be unpacked anywhere.

## Building programs

A directory with programs has a Makefile that lists them and includes `skepu.mk` (`template/` is a small one to start from):

```
cd template
make                        # build the programs of the Makefile into bin/ (next to the Makefile)
make run/hello              # build hello if out of date, then run it with lookup.hello
bin/hello OpenMP            # or run it directly (the last argument is usually the backend: CPU, OpenMP, OpenCL or CUDA, as the program decides)
```

`make run/<program>` runs the program from the package root, where labs keep their data. Its arguments are `lookup.<program>` in the Makefile, or `ARGS="..."`
(`make run/hello ARGS=CPU`).

The default backends are OpenMP, CUDA and OpenCL. If the computer lacks one, build without it:

```
make BACKENDS="-openmp"     # or set BACKENDS = -openmp in local.mk (next to skepu.mk), which every make reads
```

`skepu.mk` documents all settings: backends, compiler, include directories, files linked into every program.

## Seeing what a program does

`make TRACE=1 run/<program>` builds with tracing and runs the program; the trace is `trace/trace_<program>.json` in the package root. SkeVU, the SkePU
visualizer, shows it in a browser (Files > Load Trace, then the trace and the source files). A program run directly writes `trace/trace_<program>.json` relative to
the current directory, if that directory exists. `TRACE_SNAPSHOTS=1` also saves the containers that the program marks with `SKEPU_TRACE_SNAPSHOT`, to `snapshots/`
and, after `make run/<program>`, to `snapshots_<program>.zip` (the form SkeVU loads). If `extras/` has `skepu-visualizer.vsix`, that is the SkeVU extension
for VS Code: `code --install-extension extras/skepu-visualizer.vsix` (`codium` for VSCodium).

## Language server

`bin/clangd` is a clangd that knows SkePU (hovers for skeleton instances, signature help for user functions). See `LSP.md` to use it in an editor.

## What is where

- `bin/`: `skepu-tool` and `clangd`; `lib/SkePU/clang-headers`: the headers that the precompiler needs
- `include/`: SkePU (`skepu`, `skepu3/`, `skepu-lib/`) and what it includes (`external/`)
- `extras/`: `lodepng`, for reading and writing PNG files
- `skepu.mk`, `local.mk`: how programs are built, and the settings of this computer; `template/`: a small program with a Makefile
- `VERSION.txt`: which version of SkePU this is (and of the added material, if the package has some); `licenses/`
