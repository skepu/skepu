#!/bin/bash
# Builds a SkePU program for a single node and for the MPI backend, and compares the output of the two on several
# numbers of processes. The source is written for one node.
#
#   compare.sh prog.cpp [skepu-tool flags: -openmp ...]
#
# Environment: SKEPU_TOOL (skepu-tool), SKEPU_HEADERS (skepu-headers/src), CXX (the C++ compiler), RANKS (the
# numbers of processes, default "1 2 3 4 5 7"), EXTRA (more compiler flags), SKEPU_SYSTEM_FLAGS (the system include
# directories that skepu-tool needs, as on the command line after --), WORK (the directory for the results).
set -e
SRC=$1; shift
BACKENDS="$@"
NAME=$(basename $SRC .cpp)
: ${SKEPU_TOOL:?set SKEPU_TOOL}
: ${SKEPU_HEADERS:?set SKEPU_HEADERS}
: ${CXX:=g++}
: ${RANKS:="1 2 3 4 5 7"}
WORK=${WORK:-$(mktemp -d)}
mkdir -p $WORK/single $WORK/mpi
OMP=""; case "$BACKENDS" in *openmp*) OMP="-fopenmp";; esac

$SKEPU_TOOL $BACKENDS -silent -name ${NAME}_single -dir=$WORK/single $SRC -- $SKEPU_SYSTEM_FLAGS -I$SKEPU_HEADERS
$CXX -O1 -std=gnu++17 $EXTRA $OMP -I$SKEPU_HEADERS $WORK/single/${NAME}_single.cpp -o $WORK/single/$NAME
(cd $WORK/single && ./$NAME > output.txt 2>&1) || echo "single node: exit status $?"

$SKEPU_TOOL -mpi $BACKENDS -silent -name ${NAME}_mpi -dir=$WORK/mpi $SRC -- $SKEPU_SYSTEM_FLAGS -I$SKEPU_HEADERS
OMPI_CXX=$CXX MPICH_CXX=$CXX mpicxx -O1 -std=gnu++17 $EXTRA $OMP -I$SKEPU_HEADERS $WORK/mpi/${NAME}_mpi.cpp -o $WORK/mpi/$NAME

status=0
for n in $RANKS; do
	(cd $WORK/mpi && mpirun --oversubscribe -n $n ./$NAME > output_$n.txt 2>&1) || echo "$n processes: exit status $?"
	if cmp -s $WORK/single/output.txt $WORK/mpi/output_$n.txt; then
		echo "$n processes: same output as a single node"
	else
		echo "$n processes: DIFFERENT output (diff $WORK/single/output.txt $WORK/mpi/output_$n.txt)"; status=1
	fi
done
exit $status
