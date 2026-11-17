#!/bin/bash
# Makes a SkePU package: skepu-tool, the headers, the language server and skepu.mk (make/skepu.mk: the build rules) in one tar file, unpacked and used in place.
# The same for any course, lab or tutorial: what is particular to one is a directory, the overlay, added on top. Run it on a computer like the lab's, from a
# built SkePU (cmake --build build); it does not build, and warns if the build is out of date. A Linux package is for Linux computers like this one: the programs it needs, such as
# libstdc++, are the computer's.
#
#   packaging/make-package.sh [--build-dir DIR] [--overlay DIR] [--vsix [FILE|auto]] [--name NAME] [--output DIR] [--no-test]
#
#   --build-dir DIR   the build directory of SkePU (default: build)
#   --overlay DIR     the course's programs, Makefiles, data and documentation, added on top of the package; a file of the same name replaces the package's.
#                     Its Makefiles include skepu.mk of the package root. Not added: .git, and the patterns listed one per line in the overlay's
#                     .packageignore (PACKAGING.md, .gitignore).
#   --vsix [FILE|auto] the SkeVU extension for VS Code, put in extras/ of the package as skepu-visualizer.vsix. auto: visualizer/skepu-visualizer.vsix, made by make vsix
#                     in visualizer/ (which needs npm install, and builds it only if a source is newer); it is not built unless asked for
#   --name NAME       package name, also the directory inside it (default: skepu-lab-<version>)
#   --output DIR      where the tar file is written (default: the current directory)
#   --no-test         skip unpacking the package and building with it
#
# Without an overlay the package has template/, a small program to start from, and the test builds and runs it. With an overlay it has the overlay's
# programs instead, and the test builds those.
#
# Through CMake: cmake --build build --target skepu-package (-DSKEPU_PACKAGE_OVERLAY=... for the overlay, -DSKEPU_PACKAGE_VSIX=file.vsix or AUTO for the extension).
# Environment for the test: TEST_CXX (compiler, default g++), TEST_BACKENDS (default none: the sequential backend, which every computer has).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR=$ROOT/build
OVERLAY=
VSIX=
NAME=
OUTPUT=$PWD
TEST=1
while [ $# -gt 0 ]; do
	case $1 in
		--build-dir) BUILD_DIR=$(cd "$2" && pwd); shift 2 ;;
		--overlay)   OVERLAY=$(cd "$2" && pwd); shift 2 ;;
		--vsix)
			if [ $# -ge 2 ] && [ "$2" != auto ] && [ "${2#--}" = "$2" ]; then
				[ -f "$2" ] || { echo "no such file: $2" >&2; exit 2; }
				VSIX=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift 2
			else
				VSIX=auto; shift
				if [ "${1:-}" = auto ]; then shift; fi
			fi ;;
		--name)      NAME=$2; shift 2 ;;
		--output)    mkdir -p "$2"; OUTPUT=$(cd "$2" && pwd); shift 2 ;;
		--no-test)   TEST=0; shift ;;
		-h|--help)   sed -n '2,/^# Environment/p' "$0"; exit 0 ;;
		*) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
	esac
done

VERSION=$(git -C "$ROOT" describe --always --dirty 2>/dev/null || echo unknown)
NAME=${NAME:-skepu-$VERSION}
[ -d "$BUILD_DIR" ] || { echo "no build directory: $BUILD_DIR (configure and build SkePU first)" >&2; exit 1; }

# Not built here: build first, in parallel (cmake --build DIR --parallel N)
TOOL=$BUILD_DIR/llvm/bin/skepu-tool
if [ ! -x "$TOOL" ]; then
	echo "no skepu-tool in $BUILD_DIR/llvm/bin: build SkePU first" >&2; exit 1
fi
NEWER=$(find "$ROOT/src" "$ROOT/CMakeLists.txt" "$ROOT/CMake" -type f -newer "$TOOL" 2>/dev/null | head -3)
if [ -n "$NEWER" ]; then
	echo "warning: the build is out of date (newer than skepu-tool):" >&2
	echo "$NEWER" | sed 's/^/  /' >&2
fi

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/skepu-package.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
PKG=$STAGE/$NAME

if [ "$VSIX" = auto ]; then
	echo "== the SkeVU extension (make vsix in visualizer/)"
	make -C "$ROOT/visualizer" vsix > "$STAGE/vsix.log" 2>&1 || { cat "$STAGE/vsix.log" >&2; exit 1; }
	grep -i warning "$STAGE/vsix.log" | head -3 >&2 || true
	grep -q "Nothing to be done\|is up to date" "$STAGE/vsix.log" && echo "   up to date"
	VSIX=$ROOT/visualizer/skepu-visualizer.vsix
	[ -f "$VSIX" ] || { echo "make vsix made no $VSIX" >&2; exit 1; }
fi

echo "== installing SkePU"
cmake --install "$BUILD_DIR" --prefix "$PKG" > "$STAGE/install.log" || { cat "$STAGE/install.log" >&2; exit 1; }
strip "$PKG/bin/skepu-tool" 2>/dev/null || echo "(skepu-tool is not stripped)"

if [ -x "$BUILD_DIR/llvm/bin/clangd" ]; then
	cp "$BUILD_DIR/llvm/bin/clangd" "$PKG/bin/clangd"
	strip "$PKG/bin/clangd" 2>/dev/null || true
else
	echo "warning: no language server in $BUILD_DIR/llvm/bin (configure with -DSKEPU_LSP=ON): the package has none" >&2
fi

echo "== adding skepu.mk and the template"
cp -R "$ROOT/packaging/files/." "$PKG/"
if [ -n "$OVERLAY" ]; then rm -rf "$PKG/template"; fi
cp "$ROOT/make/skepu.mk" "$PKG/skepu.mk"
if [ -f "$ROOT/make/local.mk" ]; then
	cp "$ROOT/make/local.mk" "$PKG/local.mk"
else
	echo "warning: no make/local.mk: the package has the template, with defaults only" >&2
	cp "$ROOT/make/local.mk.template" "$PKG/local.mk"
fi
mkdir -p "$PKG/extras" "$PKG/licenses"
cp "$ROOT/examples/lodepng.cpp" "$ROOT/examples/lodepng.h" "$PKG/extras/"
[ -n "$VSIX" ] && cp "$VSIX" "$PKG/extras/skepu-visualizer.vsix"
cp "$ROOT/LICENSE.txt" "$ROOT/CONTRIBUTORS.txt" "$PKG/licenses/"
[ -f "$ROOT/llvm/clang/LICENSE.TXT" ] && cp "$ROOT/llvm/clang/LICENSE.TXT" "$PKG/licenses/LLVM-LICENSE.TXT"
printf 'SkePU %s, packaged %s on %s (%s)\n' "$VERSION" "$(date +%Y-%m-%d)" "$(uname -n)" "$(uname -sm)" > "$PKG/VERSION.txt"

if [ -n "$OVERLAY" ]; then
	echo "== adding the overlay: $OVERLAY"
	# not .git, nor what .packageignore lists (staff files, .gitignore)
	IGNORE=(--exclude=.git --exclude=.packageignore)
	[ -f "$OVERLAY/.packageignore" ] && IGNORE+=(--exclude-from="$OVERLAY/.packageignore")
	tar -C "$OVERLAY" "${IGNORE[@]}" -cf - . | tar -C "$PKG" -xf -
	printf "overlay: %s\n" "$(git -C "$OVERLAY" describe --always --dirty 2>/dev/null || basename "$OVERLAY")" >> "$PKG/VERSION.txt"
fi

# Finder metadata of macOS: in the overlay, and in the installed headers if the source tree has it
find "$PKG" -name .DS_Store -type f -delete

TAR=$OUTPUT/$NAME.tar.gz
tar -C "$STAGE" -czf "$TAR" "$NAME"
echo "== wrote $TAR ($(du -h "$TAR" | cut -f1))"

if [ $TEST = 1 ]; then
	CHECK=$(mktemp -d "${TMPDIR:-/tmp}/skepu-package-test.XXXXXX")
	trap 'rm -rf "$STAGE" "$CHECK"' EXIT
	tar -C "$CHECK" -xzf "$TAR"
	MAKE_ARGS=(BACKENDS="${TEST_BACKENDS:-}" CXX="${TEST_CXX:-g++}")   # no backend flags: precompiled for the sequential backend
	if [ -z "$OVERLAY" ]; then
		echo "== testing: building and running the template of the unpacked package"
		( cd "$CHECK/$NAME/template" \
			&& make "${MAKE_ARGS[@]}" bin/hello > "$CHECK/make.log" 2>&1 \
			&& make "${MAKE_ARGS[@]}" run/hello lookup.hello=CPU > "$CHECK/run.log" 2>&1 ) \
			|| { echo "the test failed:" >&2; tail -20 "$CHECK/make.log" "$CHECK/run.log" >&2 2>/dev/null; exit 1; }
		echo "   the template was built and run"
	else
		tested=0
		for dir in . programs; do   # the Makefile of the overlay: in its root, or in programs/
			[ -f "$OVERLAY/$dir/Makefile" ] || continue
			echo "== testing: building the programs of the overlay (make in $dir of the unpacked package)"
			( cd "$CHECK/$NAME/$dir" && make "${MAKE_ARGS[@]}" > "$CHECK/overlay.log" 2>&1 ) \
				|| { echo "the overlay does not build with the package:" >&2; tail -20 "$CHECK/overlay.log" >&2; exit 1; }
			echo "   built with BACKENDS='${TEST_BACKENDS:-}'; not run"
			tested=1
			break
		done
		[ $tested = 1 ] || echo "warning: no Makefile in the root or in programs/ of the overlay: nothing was tested" >&2
	fi
fi
( cd "$OUTPUT" && { shasum -a 256 "$NAME.tar.gz" 2>/dev/null || sha256sum "$NAME.tar.gz"; } )
