#!/bin/bash
# Makes a SkePU package: skepu-tool, the headers, the language server and skepu.mk (how programs are built), in one tar file that is unpacked and used where it is. It is
# the same for any course, lab or tutorial: what is particular to one is a directory, the overlay, that is added on top. Run it on
# a computer like the ones of the lab, from a SkePU that has been built (cmake --build build); it builds what is out of date first, which is nothing if the build is
# up to date. A Linux package is for Linux computers like this one (the programs it needs, such as libstdc++, are those of the computer).
#
#   packaging/make-package.sh [--build-dir DIR] [--overlay DIR] [--name NAME] [--output DIR] [--no-build] [--no-test]
#
#   --build-dir DIR   the build directory of SkePU (default: build)
#   --overlay DIR     what is particular to the course: its programs, Makefiles, data, documentation. Added on top of the package, so a file of the same
#                     name replaces ours (README.md, for example). Its Makefiles include skepu.mk of the package root.
#                     Not added: .git, and what the overlay's .packageignore lists (patterns, one on each line: PACKAGING.md, .gitignore).
#   --name NAME       the name of the package, and of the directory in it (default: skepu-lab-<version>)
#   --output DIR      where the tar file is written (default: the current directory)
#   --no-build        do not run the build first (cmake --build)
#   --no-test         do not unpack the package and build a program with it, as the check that it works
#
# The same through CMake: cmake --build build --target skepu-package (with -DSKEPU_PACKAGE_OVERLAY=... for the overlay).
# Environment for the test: SMOKE_CXX is the compiler (g++), SMOKE_BACKENDS the backends to test with (-openmp).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR=$ROOT/build
OVERLAY=
NAME=
OUTPUT=$PWD
BUILD=1
TEST=1
while [ $# -gt 0 ]; do
	case $1 in
		--build-dir) BUILD_DIR=$(cd "$2" && pwd); shift 2 ;;
		--overlay)   OVERLAY=$(cd "$2" && pwd); shift 2 ;;
		--name)      NAME=$2; shift 2 ;;
		--output)    mkdir -p "$2"; OUTPUT=$(cd "$2" && pwd); shift 2 ;;
		--no-build)  BUILD=0; shift ;;
		--no-test)   TEST=0; shift ;;
		-h|--help)   sed -n '2,17p' "$0"; exit 0 ;;
		*) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
	esac
done

VERSION=$(git -C "$ROOT" describe --always --dirty 2>/dev/null || echo unknown)
NAME=${NAME:-skepu-$VERSION}
[ -d "$BUILD_DIR" ] || { echo "no build directory: $BUILD_DIR (configure and build SkePU first)" >&2; exit 1; }

if [ $BUILD = 1 ]; then
	echo "== building what is out of date"
	cmake --build "$BUILD_DIR"
fi

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/skepu-package.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
PKG=$STAGE/$NAME

echo "== installing SkePU"
cmake --install "$BUILD_DIR" --prefix "$PKG" > "$STAGE/install.log" || { cat "$STAGE/install.log" >&2; exit 1; }
strip "$PKG/bin/skepu-tool" 2>/dev/null || echo "(skepu-tool is not stripped)"

if [ -x "$BUILD_DIR/llvm/bin/clangd" ]; then
	cp "$BUILD_DIR/llvm/bin/clangd" "$PKG/bin/clangd"
	strip "$PKG/bin/clangd" 2>/dev/null || true
else
	echo "warning: no language server in $BUILD_DIR/llvm/bin (configure with -DSKEPU_LSP=ON): the package has none" >&2
	rm -f "$STAGE/lsp-missing"; touch "$STAGE/lsp-missing"
fi

echo "== adding skepu.mk, the documentation and the template"
cp -R "$ROOT/packaging/files/." "$PKG/"
mkdir -p "$PKG/extras" "$PKG/licenses"
cp "$ROOT/examples/lodepng.cpp" "$ROOT/examples/lodepng.h" "$PKG/extras/"
cp "$ROOT/LICENSE.txt" "$ROOT/CONTRIBUTORS.txt" "$PKG/licenses/"
[ -f "$ROOT/llvm/clang/LICENSE.TXT" ] && cp "$ROOT/llvm/clang/LICENSE.TXT" "$PKG/licenses/LLVM-LICENSE.TXT"
[ -f "$STAGE/lsp-missing" ] && rm -f "$PKG/LSP.md"
printf 'SkePU %s, packaged %s on %s (%s)\n' "$VERSION" "$(date +%Y-%m-%d)" "$(uname -n)" "$(uname -sm)" > "$PKG/VERSION"

if [ -n "$OVERLAY" ]; then
	echo "== adding the overlay: $OVERLAY"
	# (not .git, and not what the overlay lists in its .packageignore, one pattern on each line: the files of the staff, the .gitignore)
	IGNORE=(--exclude=.git --exclude=.packageignore)
	[ -f "$OVERLAY/.packageignore" ] && IGNORE+=(--exclude-from="$OVERLAY/.packageignore")
	tar -C "$OVERLAY" "${IGNORE[@]}" -cf - . | tar -C "$PKG" -xf -
	printf "overlay: %s\n" "$(git -C "$OVERLAY" describe --always --dirty 2>/dev/null || basename "$OVERLAY")" >> "$PKG/VERSION"
fi

TAR=$OUTPUT/$NAME.tar.gz
tar -C "$STAGE" -czf "$TAR" "$NAME"
echo "== wrote $TAR ($(du -h "$TAR" | cut -f1))"

if [ $TEST = 1 ]; then
	echo "== testing: unpacking and building a program with the package"
	CHECK=$(mktemp -d "${TMPDIR:-/tmp}/skepu-package-test.XXXXXX")
	trap 'rm -rf "$STAGE" "$CHECK"' EXIT
	tar -C "$CHECK" -xzf "$TAR"
	cp -R "$ROOT/packaging/smoke" "$CHECK/$NAME/smoke_test"   # (a small program of ours, not part of the package: it is built with the package's skepu.mk)
	( cd "$CHECK/$NAME/smoke_test" \
		&& make BACKENDS="${SMOKE_BACKENDS:--openmp}" CXX="${SMOKE_CXX:-g++}" bin/smoke > "$CHECK/make.log" 2>&1 \
		&& ./bin/smoke OpenMP > "$CHECK/run.log" 2>&1 ) \
		|| { echo "the test failed:" >&2; tail -20 "$CHECK/make.log" "$CHECK/run.log" >&2 2>/dev/null; exit 1; }
	echo "   a program was built and run from the unpacked package"
fi
( cd "$OUTPUT" && { shasum -a 256 "$NAME.tar.gz" 2>/dev/null || sha256sum "$NAME.tar.gz"; } )
