[![pipeline status](https://gitlab.ida.liu.se/exa2pro/skepu/badges/master/pipeline.svg)](https://gitlab.ida.liu.se/exa2pro/skepu/commits/master)
[![coverage report](https://gitlab.ida.liu.se/exa2pro/skepu/badges/master/coverage.svg)](https://gitlab.ida.liu.se/exa2pro/skepu/commits/master)

# SkePU 3

SkePU 3 consists of four parts:

1. A sequential interface and accompanying implementation for C++ skeleton programming.
2. A source-to-source precompiler tool built on top of the Clang C-language compiler front-end,
	transforming code written for the sequential interface for parallel and heterogeneous execution.
3. Minor patches to Clang to support said tool.
4. A collection of parallel and/or heterogeneous back-ends targeting various architectures and systems.

## Setting up the SkePU source-to-source compiler

### Cloning with submodules

This repository should be cloned with `git clone --recursive $URL` in
order to also clone the submodules it links to. `git submodule update --init`
can be used to clone submodules to an existing repository.

### Building - Automatically

Run

`$ make`

in the project root.

The automated build depends on CMake and rsync. It is recommendend to
build with the `-j` flag to get a parallel build, for example `$ make
-j$(nproc)`. The binary will be available in `build/bin/skepu`.


### Building - Manually

Running the `Makefile` in the project root will execute the following
steps with some minor changes, but will use `rsync` and a separate
copy of the source tree for building.

The steps followed in building SkePU "by hand" is outlined below.

1. Clone LLVM and Clang Git repositories and confirm that Clang builds.
2. Patch Clang (see below) to add SkePU attributes and diagnostics to Clang.
	It also instructs the Clang build system to build the SkePU tool.
3. Create a directory symlink in `<clang source>/tools` named `skepu` pointing to `clang_precompiler`.
	Clang will now find the source files for the SkePU tool.
4. Run `ninja skepu` in your Clang build directory.
5. If successful, the SkePU precompiler binary should now be in `<clang build>/bin/skepu`.
6. Run `skepu -help` to confirm that everything worked and to see available options.


#### Symlink SkePU sources in Clang

`$ ln -s <skepu source>/clang_precompiler <clang source>/tools/skepu-tool`

#### Building SkePU precompiler

Create and/or move to build directory.

`$ cd <clang build>`

Set up CMake (once)

`$ cmake -G "Unix Makefiles" <llvm source> -DCMAKE_BUILD_TYPE=Release`

(Release mode is faster and much more space efficient, remove option to build in Debug mode.)

Build the SkePU tool (once for SkePU users)

`$ make skepu-tool` (in the `build` folder previously created)

## Compatibility with SkePU 1 and 2

SkePU v3 is not compatible with version 2 nor with version 1. Check the user
guide for more information.

SkePU 1 code is not compatible with SkePU 2 and vice-versa. SkePU 2 is in large part based on concepts from SkePU 1, and the data structures are the same, so it should be fairly straightworward to port a SkePU 1 project to SkePU 2. It may require some effort to fit the SkePU 2 precompiler into a large project with non-trivial build system, however.

## Directory layout


### `include`

As SkePU 3 is a header library, this directory contains headers and source files for the SkePU runtime.

#### `include/skepu`

Contains the serial skeleton interface (headers with inline implementations), along with headers for SkePU containers.

##### `include/skepu3/impl`

Helpers.

##### `include/skepu3/backend`

Contains header files for the SkePU skeleton backends.

##### `include/skepu3/backend/impl`

Contains implementations of the various SkePU skeleton backends and containers.

### `examples`

Contains SkePU 3 example programs.

### `clang_precompiler`

Sources for the source-to-source Clang tool. This directory is part of the LLVM directory tree, and a symlink named `<path_to_llvm>/llvm/tools/clang/tools/skepu` should link here.
