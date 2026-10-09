#!/bin/bash
# Builds tests/mpi/trace_map.cpp (C++20: the source locations of the events) for the MPI backend with SKEPU_TRACING, runs it on several numbers of processes, and checks the
# trace files with check_trace.py.
#
#   trace.sh [skepu-tool flags: -openmp ...]
#
# Environment: as for compare.sh (SKEPU_TOOL, SKEPU_HEADERS, CXX, RANKS, SKEPU_SYSTEM_FLAGS, WORK), and SKEPU_TRACE_TEST_SKEW_US
# (start the clocks of the processes at different times, to test that they are aligned).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
BACKENDS="$@"
: ${SKEPU_TOOL:?set SKEPU_TOOL}
: ${SKEPU_HEADERS:?set SKEPU_HEADERS}
: ${CXX:=g++}
: ${RANKS:="1 2 3 4"}
WORK=${WORK:-$(mktemp -d)}
mkdir -p $WORK
OMP=""; case "$BACKENDS" in *openmp*) OMP="-fopenmp";; esac

$SKEPU_TOOL -mpi $BACKENDS -silent -name trace_map -dir=$WORK $HERE/trace_map.cpp -- $SKEPU_SYSTEM_FLAGS -I$SKEPU_HEADERS
OMPI_CXX=$CXX MPICH_CXX=$CXX mpicxx -O1 -std=gnu++20 -DSKEPU_TRACING -DSKEPU_TRACE_SNAPSHOT_DIR=snapshots $EXTRA $OMP -I$SKEPU_HEADERS $WORK/trace_map.cpp -o $WORK/trace_map

for n in $RANKS; do
  rm -rf $WORK/skepu_trace*.json $WORK/snapshots; mkdir -p $WORK/snapshots
  (cd $WORK && mpirun --oversubscribe -n $n ./trace_map > /dev/null)
  python3 $HERE/check_trace.py $WORK $n
done
