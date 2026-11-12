# SkePU packages

`make-package.sh` makes one tar file of SkePU for a group that does not build SkePU and LLVM itself: a course, a lab or a tutorial. The package is unpacked
and used in place; nothing is installed. The options are listed by `make-package.sh --help`.

## Contents of a package

| Path | Contents |
| --- | --- |
| `bin/skepu-tool` | The precompiler. |
| `bin/clangd` | The language server, if SkePU was configured with `-DSKEPU_LSP=ON`. |
| `include/`, `lib/` | The SkePU headers, the Clang headers that `skepu-tool` uses, and the CMake export files. |
| `skepu.mk` | The build rules (`make/skepu.mk`): OpenMP, CUDA and OpenCL by default. |
| `local.mk` | The template for the settings of one computer (`make/local.mk.template`). |
| `template/` | A small program with a Makefile to start from (without an overlay; with one, the overlay's programs take its place). |
| `extras/` | `lodepng`, and the SkeVU extension if it was requested. |
| `licenses/`, `VERSION.txt` | The licences; the SkePU version, the date and computer of the build, and the overlay. |
| `README.md`, `LSP.md` | The text for the package user (`files/`). |

A Linux package is for Linux computers like the one it was made on: the programs that it needs, such as libstdc++, are the computer's.

## Making a package

SkePU has to be built first (`cmake --build build`; the LLVM build is long). The script is run on a computer like the target computers:

```bash
packaging/make-package.sh --name NAME --output DIR [--overlay DIR] [--vsix auto]
```

It builds what is out of date (`--no-build` skips that), installs SkePU into a temporary directory, adds the other files and writes `DIR/NAME.tar.gz` with its
SHA-256. Through CMake: `cmake --build build --target skepu-package`, with `-DSKEPU_PACKAGE_OVERLAY=DIR` and `-DSKEPU_PACKAGE_VSIX=AUTO` (or a file) for the options below.

## Overlay

The files of a course (programs, Makefiles, data, README) are a directory, the overlay, added on top of the package with `--overlay DIR`. A file with the name of a
package file replaces it (a `README.md` of the overlay replaces `files/README.md`). `.git` and the patterns that the overlay lists in `.packageignore`, one per
line, are not added. The Makefiles of the overlay include `skepu.mk` of the package root.

## SkeVU extension

`--vsix auto` puts `visualizer/skepu-visualizer.vsix` in `extras/`, made first by `make vsix` in `visualizer/` if a source is newer. That needs `npm install` in
`visualizer/`, and Quarto for the user guide that the extension contains (without Quarto the extension has no guide, with a warning). `--vsix FILE` takes a file that is
built already. In the package the file is always `extras/skepu-visualizer.vsix`.

## Test

After writing the tar file, the script unpacks it in a temporary directory and, with the package's `skepu.mk`:

- without an overlay: builds and runs `template/`;
- with an overlay: runs `make` in the overlay's Makefile directory (its root, or `programs/`); the programs are built, not run. An overlay without a Makefile there is not tested.

The environment variables `TEST_CXX` (default `g++`) and `TEST_BACKENDS` (default none: the sequential backend, which needs no OpenMP, OpenCL or CUDA) select the compiler and the backends. `--no-test` skips the test.

## Known issue

On macOS, `skepu-tool` finds the system headers through the active toolchain. With `CPATH` (and `SDKROOT`) set in the environment, the precompiler did not find the C++ library's
`stddef.h` and the test failed. The variables were exported by Apple's Command Line Tools Python launcher; running the script from a shell without them is fine.

## Files

`make-package.sh`; `files/`, copied to the root of the package.
