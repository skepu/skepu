#!/usr/bin/env bash
# Regression test for SkePU-DNN: builds the DNN examples (sequential, no precompiler), runs a set of
# short fixed-seed training runs and compares their logs (losses, accuracies, confusion matrices)
# with the golden logs in golden/.
#
#   run.sh [--update] [--strict] [--tol REL] [--cxx COMPILER] [--build-dir DIR]
#
#   --update     overwrite the golden logs with the current output (after an intended change)
#   --strict     require identical logs (only meaningful with the same compiler and flags as the goldens)
#   --tol REL    relative tolerance for numbers (default 1e-3)
#
# Cases that need data sets missing from examples/data (MNIST, CIFAR-10) are skipped.
# Exit status is 0 if no case failed.
set -u

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
examples="$root/examples"
programs="$examples/dnn"
golden="$here/golden"

cxx="${CXX:-clang++}"
tol="1e-3"
strict=""
update=0
build_dir=""
while [ $# -gt 0 ]; do
	case "$1" in
		--update) update=1 ;;
		--strict) strict="--strict" ;;
		--tol) tol="$2"; shift ;;
		--cxx) cxx="$2"; shift ;;
		--build-dir) build_dir="$2"; shift ;;
		*) echo "unknown option $1"; exit 2 ;;
	esac
	shift
done
[ -z "$build_dir" ] && build_dir="$(mktemp -d)"
mkdir -p "$build_dir"

# name | example program | data needed (directory in examples/data, or - for none) | arguments
cases=(
	"mnist_mlp_sgd|dnn_mnist10|mnist|mlp 32 0.1 2 cpu 6000 1000"
	"mnist_cnn_sgd|dnn_mnist10|mnist|cnn 32 0.05 1 cpu 1600 330"
	"mnist_conv1d_adam|dnn_mnist10|mnist|conv1d 32 0.001 1 cpu 3200 330 --adam"
	"signals_conv1d_adam|dnn_signals1d|-|conv1d 32 0.003 2 cpu 1600 330 --adam"
	"cifar_small_adam|dnn_cifar10|cifar-10-batches-bin|small 32 0.001 1 cpu 1600 330 --adam"
	"binary_mnist_sgd|dnn|mnist|16 0.05 2 cpu 1"
)

# Lines of the example output that are compared
filter() { grep -E "Train Batch|^Epoch|^Test|^Matrix|^[0-9 ]+$"; }

built=" "
failed=0; passed=0; skipped=0
for c in "${cases[@]}"; do
	IFS='|' read -r name prog data args <<< "$c"

	if [ "$data" != "-" ] && [ ! -d "$examples/data/$data" ]; then
		echo "SKIP  $name (no data set in examples/data/$data)"
		skipped=$((skipped + 1))
		continue
	fi

	if [[ "$built" != *" $prog "* ]]; then
		if ! "$cxx" -std=c++20 -O2 -I "$root/skepu-headers/src" -Wno-attributes "$programs/$prog.cpp" -o "$build_dir/$prog" > "$build_dir/$prog.build.log" 2>&1; then
			echo "FAIL  $name (build of $prog failed, see $build_dir/$prog.build.log)"
			failed=$((failed + 1))
			continue
		fi
		built="$built$prog "
	fi

	# The examples read their data relative to their directory (../data)
	# shellcheck disable=SC2086
	(cd "$programs" && "$build_dir/$prog" $args) > "$build_dir/$name.log" 2>&1
	filter < "$build_dir/$name.log" > "$build_dir/$name.txt"

	if [ "$update" = 1 ]; then
		cp "$build_dir/$name.txt" "$golden/$name.txt"
		echo "UPDATED $name"
		continue
	fi

	if [ ! -f "$golden/$name.txt" ]; then
		echo "FAIL  $name (no golden log, run with --update)"
		failed=$((failed + 1))
	elif [ ! -s "$build_dir/$name.txt" ]; then
		echo "FAIL  $name (no output, see $build_dir/$name.log)"
		failed=$((failed + 1))
	elif python3 "$here/compare.py" "$golden/$name.txt" "$build_dir/$name.txt" --tol "$tol" $strict; then
		echo "PASS  $name"
		passed=$((passed + 1))
	else
		echo "FAIL  $name (logs in $build_dir)"
		failed=$((failed + 1))
	fi
done

echo "$passed passed, $failed failed, $skipped skipped"
[ "$failed" = 0 ]
