# SkePU package

A complete SkePU for this computer: the precompiler (`bin/skepu-tool`), the SkePU headers (`include/`), the language server (`bin/clangd`, if the package has it) and
`skepu.mk`, which builds programs. Nothing needs to be installed; the directory can be unpacked anywhere.

## Building programs

A directory with programs has a Makefile that lists them and includes `skepu.mk` (`template/` is a small one to start from):

```
cd template
make                        # builds the programs of the Makefile into bin/ (next to the Makefile)
make run/hello              # builds hello if it is out of date, and runs it with its arguments (lookup.hello in the Makefile)
bin/hello OpenMP            # or run it yourself: the last argument of a program is usually the backend: CPU, OpenMP, OpenCL or CUDA (the program decides)
```

`make run/<program>` runs the program from this directory, where a lab keeps its data, with `lookup.<program>` of the Makefile as the arguments, or with `ARGS="..."`
(`make run/hello ARGS=CPU`).

The default backends are OpenMP, CUDA and OpenCL. If the computer lacks one, build without it:

```
make BACKENDS="-openmp"     # or put BACKENDS = -openmp in local.mk (next to skepu.mk), which every make reads
```

`skepu.mk` has all the settings and their documentation: the backends, the compiler, more include directories, files that are built into every program.

## Seeing what a program does

`make TRACE=1` builds with tracing: a run writes `skepu_trace.json` (or the file in `$SKEPU_TRACE_FILE`; `make run/<program>` writes `trace_<program>.json` here), which SkeVU, the SkePU visualizer, shows in a browser (Files >
Load Trace, then the trace and the source files). `make TRACE=1 TRACE_SNAPSHOTS=1` also saves the containers that the program marks with `SKEPU_TRACE_SNAPSHOT` (in `snapshots/` when the program is run with `make run/<program>`).

## Language server

`bin/clangd` is a clangd that knows SkePU (hovers for skeleton instances, signature help for user functions). See `LSP.md` to use it in an editor.

## What is where

- `bin/`: `skepu-tool` and `clangd`; `lib/SkePU/clang-headers`: the headers that the precompiler needs
- `include/`: SkePU (`skepu`, `skepu3/`, `skepu-lib/`) and what it includes (`external/`)
- `extras/`: `lodepng`, for reading and writing PNG files
- `skepu.mk`, `local.mk`: how programs are built, and the settings of this computer; `template/`: a small program with a Makefile
- `VERSION`: which version of SkePU this is (and of the added material, if the package has some); `licenses/`
