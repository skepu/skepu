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

This repository should be cloned with `git clone --recursive --shallow-submodules $URL` in
order to also clone the most recent commits of the submodules it links to. Furthermore, `git submodule update --init` can be used to clone submodules to an existing repository.

Note: configuring the build patches files inside the `llvm` submodule, so it always counts as modified. `.gitmodules` therefore sets `ignore = dirty` and `git status` does not list it. Do not commit changes to the submodule.

### Building skepu-tool

To compile skepu-tool, run the following commands in the project root:

`$ mkdir build && cd build`

`$ cmake -DCMAKE_BUILD_TYPE=Release ..`

`$ make -jN`

Replace N in the last command with how many threads you want to build with in parallel, e.g. -j12.

If you also want to include the tests, run the following cmake command instead of the one above:

`$ cmake -DCMAKE_BUILD_TYPE=Release -DSKEPU_ENABLE_TESTING=ON ..`

To install SkePU into \<path\>, run the following command from
\<project_root\>/build:

`$ make DESTDIR=<path> install`

If cmake does not generate Makefiles, use the option `-G "Unix Makefiles"`.
For more information about build options, run ´cmake -LAH` in the build
directory. For information about how to configure cmake with compilers,
linkers, flags, and other such options, please refer to the cmake manuals at
the [cmake homepage](https://cmake.org).

## Setting up the SkePU language server

The language server is a clangd with SkePU support. It is built in the same LLVM build tree as `skepu-tool` (the `build/llvm` directory of the project build), so that the two share the compiled LLVM and Clang libraries. Configure the project with `SKEPU_LSP` on, in the project root:

`$ cmake -B build -DCMAKE_BUILD_TYPE=Release -DSKEPU_LSP=ON`

`$ cmake --build build -jN`

Replace N with how many threads you want to build with in parallel, e.g. -j12. The first configuration applies `clangd.patch` to the `llvm` submodule (as `llvm.patch` is applied for `skepu-tool`) and adds `clang-tools-extra` to the LLVM projects; with `SKEPU_LSP` off (the default) clangd is not built.

The SkePU additions to clangd are in `src/skepu-lsp/` (and the analysis of `skepu-tool` in `src/skepu-tool/`); `clangd.patch` only adds the files to clangd's build and calls them from three places (hover, document symbols and signature help). After this you have built the clangd tool with modifications specific to SkePU. To use it, point to the clangd executable in \<project_root\>/build/llvm/bin/. For example, VSCode has a clangd extension which has an option to specify the path to a clangd executable.

To make the LSP aware of SkePU and its files, you need to generate a compile_commands.json file. In the examples folder, `make compile-commands` writes one for the example programs. After doing this, the LSP should be able to find where the `#include <skepu>` and similar are fetched from. You might need to restart your code editor after generating the compile_commands.json for the changes to take effect.

## A package for a course, lab or tutorial

For a group of students that should not build SkePU and LLVM themselves, `packaging/make-package.sh` makes one tar file of what is needed: `skepu-tool`, the headers, the language
server (if it is built, `-DSKEPU_LSP=ON`) and `skepu.mk`, which builds programs for OpenMP, CUDA and OpenCL by default (`packaging/files/`). Run it on a computer like the ones the
students use, from a SkePU that has been built; it builds what is out of date, installs into a temporary directory and writes the tar file. What is particular to a course (its programs,
Makefiles, data and README) is a directory, the overlay, added on top: `--overlay DIR`, or `-DSKEPU_PACKAGE_OVERLAY=DIR` with `cmake --build build --target skepu-package`. The package
is tested by unpacking it and building a small program with it. See the top of the script for the options.

## Compatibility with SkePU 1 and 2

SkePU 3 is not compatible with SkePU 2 nor with SkePU 1. Check the user
guide for more information.

SkePU 1 code is not compatible with SkePU 2 and vice-versa. SkePU 2 is in large
part based on concepts from SkePU 1, and the data structures are the same, so
it should be fairly straightworward to port a SkePU 1 project to SkePU 2. It
may require some effort to fit the SkePU 2 precompiler into a large project
with non-trivial build system, however.

## Directory layout

### `skepu-headers/src/`

As SkePU 3 is a header library, this directory contains headers and source
files for the SkePU runtime. Herafter refered to as \<skepu-headers\>.

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

### `examples`

Contains SkePU 3 example programs. Note that the Makefile requires skepu-tool
to have been built in <project root>/build to find skepu-tool.

### `llvm`

Sources for the source-to-source Clang tool. This directory is part of the LLVM
directory tree, and a symlink named
`<project_root>/llvm/tools/clang/tools/skepu-tool` should link here.
