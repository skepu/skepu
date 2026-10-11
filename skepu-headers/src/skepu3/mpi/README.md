# SkePU MPI backend (prototype)

Distributed containers over the node-level SkePU backends, with MPI-3.1 for the communication. The goal is the
single-source model of SkePU: a program written for one node gives the same result on any number of processes.
It replaces the earlier StarPU-MPI backend, which has been removed (it is in the history of the repository).

## Using it

Precompile with `skepu-tool -mpi` (combined with `-openmp`, ...), compile the result with an MPI compiler (the
same C++ compiler as for the other backends), run with `mpirun`. No change of the source.

```
skepu-tool -mpi -openmp -name prog_mpi -dir=out prog.cpp -- <the usual flags>
OMPI_CXX=g++ mpicxx -std=gnu++17 -fopenmp -Iskepu-headers/src out/prog_mpi.cpp -o prog
mpirun -n 4 ./prog
```

`tests/mpi/compare.sh` builds a program for one node and for MPI, and compares the output on several numbers
of processes.

## Design

* Every process runs `main` (as in any MPI program) and makes the same calls in the same order. Nothing else is
  needed to keep the processes in agreement: the state of a container (which of its copies is valid) changes only in
  such calls, so the processes always know when to communicate.
* `skepu::Vector<T>` is a distributed container: a block of consecutive elements per process, held in a node-level
  vector (the *tile*). Skeleton calls work on the tiles with the node-level skeleton of the backends that the
  program is built for, so those need nothing of MPI. A tile knows its global position
  (`Vector::setIndexOrigin`), so that index-aware user functions get global indices.
* Element access from the host (`v(i)`, iterators, printing) gives access to the whole vector: the first access
  gathers it to every process (collective), and it stays until a skeleton changes the vector. Changes made by
  the host are given to the owners before the next skeleton call.
* `skepu::external` runs its code on the first process only, with the containers that it reads gathered and
  the containers that it writes scattered afterwards. A container used in it must be listed in
  `skepu::read(...)` or `skepu::write(...)` (it is an error otherwise: no communication is possible there).
* Only the first process writes to standard output (`SKEPU_MPI_ALL_OUTPUT=1` for output from all).
* The skeleton instances are wrappers of the node-level skeletons: `skepu::backend::Map`, `Reduce1D`. Setters
  can be chained, as for the node-level skeletons.
* A *Reduce* combines the partial results of the processes in the order of the processes, on every process, so
  all processes have the same value and a function that is not commutative works. A *start value* is the first
  operand of the whole reduction.

### The namespace

An MPI program includes the node-level SkePU under the name `skepu_node` (`<skepu>` defines `skepu` as `skepu_node`
while it includes it), and `namespace skepu` has everything of it through a using-directive, except what the
backend replaces. A qualified name `skepu::X` is therefore the backend's if it has one. A `using namespace skepu;`
makes the name `Vector` ambiguous. This is a prototype device that goes away if `Vector` itself gets a
distribution.

## State

How each skeleton is distributed is described in the comments next to each wrapper in `skeletons.hpp` and `overlap.hpp`. Every
skeleton is tested by a program in `tests/mpi/` (see Tests below).

Supported, with the node-level backends sequential, OpenMP, OpenCL and OpenMP+OpenCL:

* the containers `Vector`, `Matrix`, `Tensor3`, `Tensor4` (distributed by blocks of the first dimension), and `external`;
* `Map`, `MapReduce` (with element-wise arguments or over the indices of a default size), `Reduce` (a whole container, the
  rows or columns of a matrix, two functions), `Scan`, `MapPairs`, `MapPairsReduce`, `MapOverlap` in 1D (vectors and matrices), 2D, 3D and 4D,
  and `MapPool` in 2D, 3D and 4D;
* random numbers in the skeletons: the elements get the numbers of a single node.

A random access argument (`Vec`, `Mat`, `MatCol`, `Ten3`, `Ten4`) is gathered to every process, and is only read,
whether or not the user function declares it `const`. A `MatRow` argument is not: element `i` uses row `i`, so a process has
only the rows of its elements (Map, MapReduce, MapPairs). With a matrix of as many rows as there are elements nothing moves.
With more rows the data that moves is the least of two choices (`rows.hpp`): the rows go to the processes that have the
elements, or the elements (the results and the element-wise containers) go to the processes that have the rows, and the
results are sent back afterwards. (`SKEPU_MPI_TRACE_LAYOUT=1` prints the choice.) The row-wise patterns that have all the
rows of an element in one process anyway (the reductions of rows, `MapOverlap` along the rows, `MapPairsReduce` over rows) work on
the tiles without communication.

An error in the node-level library (`SKEPU_ERROR`) on any process stops all of them (`MPI_Abort`) instead of leaving the
others waiting in a collective operation.

Not yet: CUDA (a program built with `-mpi -cuda` does not compile: the kernels need the index origin of a distributed
container, as the OpenCL launches have it, in each launch variant), MapOverlap with an update mode other than Normal, strides, and random numbers in the mirror mode of MapOverlap (the edge mode None with an input of the size of the output: only the interior is computed, 1D to 4D; the elements that are computed get the numbers one after the other, which the processes do not know of each other yet); messages of more than 2^31-1 elements (MPI-3),
calls of a skeleton with iterators, `Matrix(std::initializer_list)`, printing of tensors in the format of the
single-node library.

A container that has only the value it was created with (`Vector(n)`, `Matrix(r, c, 7)`, `init(...)`) is not gathered to be read on the host or
to be listed in `skepu::external`: every process knows what is in it, and makes its copy without communication, until something writes it.

What an `skepu::external` call writes is not moved when the call ends. The data stays on the first process (which ran the code) until it is
needed: a skeleton gets the tiles (a scatter), the host a copy of the whole container (a broadcast). As in a single node, the data goes
where the computation needs it, when it does. (Making the tiles is a collective operation: a skeleton asks for the tiles of its arguments on
every process, also one that has no elements.)

An `skepu::external` call gets the containers that it reads to the first process only (a gather: the code runs there and nothing else needs
them). A host read on all the processes then gets the first process's copy to the others (a broadcast), and a skeleton uses the tiles, which
are still valid.

### Tracing (container allocation, all skeletons, `skepu::external` and the communication)

With `SKEPU_TRACING` each process writes a trace file. The first process writes the trace of the program (the events of a single
node program, with the ids of the program's containers), the others lean files with, for each call and each collective, the same
sequence number, the interval and a few numbers (no ids). The number of events is the number of processes times a constant for each
call, whatever the size of the problem. The events of the node-level library are not recorded; those that the program marks itself
are (regions, `SKEPU_TRACE_SNAPSHOT` of a distributed container, which every process takes part in as a gather, and the first one writes). The headers have what puts the times
on the clock of rank 0. See `skepu3/mpi/trace.hpp` and `visualizer/TRACE_FORMAT.md`; `tests/mpi/trace.sh` tests it.

### GPU backends

A tile is a part of a container, so the kernels must number its elements as in the whole container. The node-level
containers have an index origin (`indexOrigin`, `linearIndexOrigin`) that the host code adds to the index, and
the OpenCL launches add it to the index base of the kernel (which is separate from the offset of the data). As in
a single-node program, the GPU backends assume that a reduction function is commutative, and need the host code to
write containers in `skepu::external` and to `flush()` before reading elements.

### Tests

`tests/mpi/compare.sh` builds a program for one node and for MPI with the same backend flags (none, `-openmp`, `-opencl`,
`-openmp -opencl`), and compares the output on several numbers of processes. The programs in `tests/mpi/` are written for
one node: `prototype`, `containers`, `mapreduce`, `mapreduce_index`, `mappairs`, `mappairsreduce`, `reduce_matrix`, `scan`,
`mapoverlap`, `mapoverlap_mirror` (and `mapoverlap_mirror_1d`, `_3d`, `_4d`), `mapoverlap_matrix1d`, `mappool`, `random`, `api`, `matrow`, `trace_map` (with `trace.sh` and `check_trace.py`) (not with the hybrid backend, whose node-level
MapReduce has no MatRow). `noncommutative`, `noncommutative_scan` and `random_overlap` are for some of the backends only (see
their first lines). A few things that a single-node program needs in order to work with the GPU backends: C-style casts in
user functions (`(int)x`), `Random` parameters by reference and right after the index, host writes in `skepu::external`, and
`flush()` before host reads.
