# SkePU 3

SkePU 3 consists of four parts:

1. A sequential interface and accompanying implementation for C++ skeleton
programming.
2. A source-to-source precompiler tool built on top of the Clang C-language
compiler front-end, transforming code written for the sequential interface for
parallel and heterogeneous execution.
3. Minor patches to Clang to support said tool.
4. A collection of parallel and/or heterogeneous back-ends targeting various
architectures and systems.

An optional, experimental Python frontend (SkePY) is described in [SKEPY.md](SKEPY.md).

## Setting up the SkePU source-to-source compiler

### Cloning with submodules

Clone the repository with `git clone --recursive --shallow-submodules $URL` to get the most recent commits of the submodules it links to. `git submodule update --init` clones the submodules of an existing clone.

Note: configuring the build patches files inside the `llvm` submodule, so it always counts as modified. `.gitmodules` therefore sets `ignore = dirty` and `git status` does not list it. Do not commit changes to the submodule.

### Building skepu-tool

To compile skepu-tool, run the following commands in the project root:

`$ mkdir build && cd build`

`$ cmake -DCMAKE_BUILD_TYPE=Release ..`

`$ make -jN`

N is the number of parallel build jobs, for example `-j12`.

To include the tests, run this cmake command instead of the one above:

`$ cmake -DCMAKE_BUILD_TYPE=Release -DSKEPU_ENABLE_TESTING=ON ..`

To install SkePU into \<path\>, run the following command from
\<project_root\>/build:

`$ make DESTDIR=<path> install`

If cmake does not generate Makefiles, use the option `-G "Unix Makefiles"`.
`cmake -LAH` in the build directory lists the build options. Compilers, linkers
and flags are configured as described in the [CMake documentation](https://cmake.org).

## Setting up the SkePU language server

The language server is a clangd with SkePU support. It is built in the same LLVM build tree as `skepu-tool` (the `build/llvm` directory of the project build), so that the two share the compiled LLVM and Clang libraries. Configure the project with `SKEPU_LSP` on, in the project root:

`$ cmake -B build -DCMAKE_BUILD_TYPE=Release -DSKEPU_LSP=ON`

`$ cmake --build build -jN`

N is the number of parallel build jobs, for example `-j12`. The first configuration applies `clangd.patch` to the `llvm` submodule (as `llvm.patch` is applied for `skepu-tool`) and adds `clang-tools-extra` to the LLVM projects; with `SKEPU_LSP` off (the default) clangd is not built.

The SkePU additions to clangd are in `src/skepu-lsp/` (and the analysis of `skepu-tool` in `src/skepu-tool/`); `clangd.patch` only adds the files to clangd's build and calls them from three places (hover, document symbols and signature help). To use it, set the clangd path of the editor to `<project_root>/build/llvm/bin/clangd` (the VS Code clangd extension has the setting `clangd.path`).

The language server needs a `compile_commands.json` to find the SkePU headers (`#include <skepu>` and similar) and the sources of a program. `make compile-commands` in `examples/` writes one for the example programs, and in the directory of any Makefile that includes `skepu.mk` one for its programs. The editor (or the language server) has to be restarted after the file is generated.

## A package for a course, lab or tutorial

`packaging/make-package.sh` makes a tar file of SkePU for a group that does not build it itself. See [packaging/README.md](packaging/README.md).

## Compatibility with SkePU 1 and 2

SkePU 3 is not compatible with SkePU 2 or SkePU 1, and SkePU 2 is not
compatible with SkePU 1. The user guide describes the differences.

SkePU 2 is based on the concepts of SkePU 1 and has the same data structures,
so a SkePU 1 project is expected to be straightforward to port to SkePU 2.
Fitting the SkePU 2 precompiler into a large project with a non-trivial build
system can take more work.

## Directory layout

### `skepu-headers/src/`

SkePU 3 is a header library. This directory has the headers and source files
of the SkePU runtime; it is called \<skepu-headers\> below.

#### `<skepu-headers>/skepu3`

Contains the serial skeleton interface (headers with inline implementations),
along with headers for SkePU containers.

##### `<skepu-headers>/skepu3/impl`

Helpers.

##### `<skepu-headers>/skepu3/backend`

Contains header files for the SkePU skeleton backends.

##### `<skepu-headers>/skepu3/backend/impl`

Contains implementations of the various SkePU skeleton backends and containers.

##### `<skepu-headers>/skepu3/mpi`

Contains the implementation of the MPI backend (distributed containers over the
other backends). See its README.

### `src/skepu-tool`

The source of the source-to-source precompiler, `skepu-tool`. The configuration
links `<project_root>/llvm/clang/tools/skepu-tool` here and builds the tool in the
LLVM tree.

### `src/skepu-lsp`

The SkePU additions to clangd (the language server).

### `llvm`

The LLVM submodule. The configuration applies `llvm.patch` to it.

### `examples`

The SkePU 3 example programs, and `programs.txt`, the table of them (flags and
run arguments). `make` there builds with the `skepu-tool` in
`<project_root>/build`. The paths are in `make/repo.mk`, and the settings of one
computer in `make/local.mk` (copied from `make/local.mk.template`).

### `make`

The build rules for SkePU programs: `skepu.mk`, shared by the examples and the
packages.

### `packaging`

`make-package.sh` and the files that go into a package.

### `visualizer`

SkeVU, the trace visualizer: a web page and a VS Code extension.

### `tests`

The test programs, built with `-DSKEPU_ENABLE_TESTING=ON`.

### `user-guide`

The SkePU user guide (Quarto).
