# SkeVU

SkeVU is a visualizer for the execution traces of SkePU programs. It shows the skeleton calls and the containers of a run as a task graph,
the work of each backend on a timeline, the data transfers between host and devices, and the source lines that the calls come from.

## Input

A trace is a JSON file that a SkePU program writes when it is built with tracing (`SKEPU_TRACING`). The format is described in
`visualizer/TRACE_FORMAT.md` of the SkePU repository. Files with the extension `.skevu` are read as traces too.

The extension registers the custom editor **SkeVU Visualizer** (`skepu.skevu`) for `*.json` and `*.skevu` files with the priority *option*: a file is opened
in it with *Open With…* or *Reopen Editor With…*. The commands that run a program open the trace they produce in this editor.

## Source files

The calls of a trace are marked in the source files that the trace names, with badges and line highlights. A source file is annotated only if its SHA-256
hash equals the hash recorded for it in `header.files` of the trace. The file is looked up at the recorded path, and otherwise in the workspace by
name. A file that has changed since the run is not annotated. The SkePU library headers (`skepu-lib`) that a program used are annotated in the same way.

## Running a program

`SkeVU: Run and Show the Trace` (`skevu.runTrace`) is available for `.cpp` files, in the run menu of the editor title and in the Explorer context
menu. It uses the Makefile in the directory of the source file, which includes `skepu.mk` (a SkePU package, or `examples/Makefile` of the
repository):

1. `make` builds `bin/<program>` with `TRACE=1`.
2. The program is run from `RUN_CWD` with the arguments `lookup.<program>` of the Makefile.
3. The trace `trace/trace_<program>.json` is opened in SkeVU, beside the editor.

`SkeVU: Run and Show the Trace with Options…` (`skevu.runOptions`) sets, for one run, the backend (`RUN_BACKEND`), the program arguments (these replace
`lookup.<program>`), tracing, container snapshots and a forced rebuild (`make -B`). The values that are not set are the Makefile's.

## Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `skevu.run.make` | `make` | The program that runs the Makefile rules. |

## User guide

`SkeVU: Open the User Guide` and `SkeVU: Open the User Guide in a New Window` show the SkePU user guide in an editor panel. The hover of a skeleton
name in a source file links to the section of that skeleton. The guide is included only if the extension was built with Quarto; the commands report
its absence otherwise.

## Requirements

* Visual Studio Code 1.100 or later.
* For the run commands: `make` and a SkePU toolchain as `skepu.mk` uses it (`skepu-tool`, a C++ compiler, and the libraries of the backends).

## Licence

SkeVU is part of SkePU (<https://github.com/skepu/skepu>). The licence is in `LICENSE.txt` and the notices of included components in `NOTICE`.
