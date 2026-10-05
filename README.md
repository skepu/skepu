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

To build the SkePU LSP, run the following commands in the project root:

`$ cd llvm`

`$ patch -Np1 -i ../clangd.patch`

`$ mkdir build && cd build` (Note that this is creates a build folder within the llvm folder, it is separate from the build folder at the project root)

`$ cmake ../llvm/ -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra"`

`$ cmake --build . --target clangd -jN`

Replace N in the last command with how many threads you want to build with in parallel, e.g. -j12.

After running these commands, you will have built the clangd tool with modifications specific to SkePU. To use this, you can point to the clangd executable in \<project_root\>/llvm/build/bin/. For example, VSCode has a clangd extension which has an option to specify the path to a clangd executable.

To make the LSP aware of SkePU and its files, you need to generate a compile_commands.json file. See the Project setup section of the clangd [installation](https://clangd.llvm.org/installation) guide on how to do this. A simple option is to run bear on the make command, e.g. `bear -- make sequential/dotprod` in the examples folder. After doing this, the LSP should be able to find where the `#include <skepu>` and similar are fetched from. You might need to restart your code editor after generating the compile_commands.json for the changes to take effect.

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

##### `<skepu-headers>skepu3/cluster`

Containes the implementation of the StarPU MPI backend.

### `examples`

Contains SkePU 3 example programs. Note that the Makefile requires skepu-tool
to have been built in <project root>/build to find skepu-tool.

### `llvm`

Sources for the source-to-source Clang tool. This directory is part of the LLVM
directory tree, and a symlink named
`<project_root>/llvm/tools/clang/tools/skepu-tool` should link here.
