# The trace format

What `SKEPU_TRACING` writes (`skepu3/impl/tracing.h`, `tracing.hpp`) and what SkeVU and the analysis server read: a JSON array of
events, in the order in which they were recorded (a trace is written when the program ends, or by `SKEPU_TRACE_FLUSH()`).
`tests/codegen/tracing_skeletons.cpp` checks the parts of it that are described here, on every backend.

A trace is a JSON array of events, or an object `{ header, data }` whose `data` is that array (the key can be anywhere in the
object). The header has what is known about the run as a whole. Source files: a key that is the path of a source file with the hash of its
contents as value (so that SkeVU can tell whether the source it has is the source that was run). The other sections have names that are not
paths: `instances` (below), `mpi` and `clock` (below). SkeVU reads both forms; a trace that is an array has no header.

Times are microseconds since the first event of the program. An event that takes time has `start` and `end`; one that
happens at a moment has `time`. Every event has the `region` it happened in, if any (the innermost
`SKEPU_TRACE_SCOPE`; regions nest per thread, so an event of a thread that did not open the region, such as
the CPU part of a hybrid call, has none), and most have `label`, `file` and `line` (`file` is empty and `line` is 0
for the host code that is not compiled as C++20, which has no `std::source_location`).

## Containers

`allocation` and `deallocation` (`object_id`, `label`, `time`): a Vector, Matrix or Tensor gets an id when it is constructed and
the label it was given (`skepu::Vector<float> v(n, "v")`); one without a label is `Vector #id` or `Matrix #id` (the id
counts in order of creation, so the name is the same in every run). The id belongs to the data: `swap` and the move
constructor swap ids. A PRNG stream is also allocated (label `[Global PRNG]` for the default stream) and never freed.
The temporaries of SkePU are labelled `Internal (...)`: `[Internal PRNG]`, `Internal (reduce-temp)`,
`Internal (overlap-wrap)`, `Internal (transpose)`. Scalars (the result of a Reduce) have an id and no allocation.

A container whose life starts when it gets memory in an assignment (an empty one, or one that was moved from), or that wraps
memory it was given (`Vector(ptr, size)`), is allocated there; one that wraps memory ends with a `deallocation` like the others.

`copy` (`source`, `object_id` of the new or assigned container, `label` of it, `elements`, `start`, `end`): the copy constructor and the
assignment of a Vector, Matrix or Tensor (the elements are copied; they were once shared by two matrices). It reads the source
(which is brought to the host first: a `transfer` inside the interval) and writes the destination. It has no `file` and `line`: the
constructor and the assignment operator have no place for them. A *move* has no event: the id goes with the data, so the call that
reads the moved container reads the id of the container that was moved from. `Vector::randomize`, `Matrix::randomize`, `randomizeReal`, a fill
(`matrix = value`) and the assignment of an initializer list are `external` events (label `randomize`, `fill`, `assign`) that write the container.

Not traced: element access through `operator()` and `operator[]`, which is only coherent inside `skepu::external` (traced there) or with
`Vector::elem`; `load`, `save`; SparseMatrix.

## Calls

`skeleton_call`:

| field | meaning |
|---|---|
| `object_id`, `label` | the skeleton instance (its variable name, or what `setLabel` gave it), `file` and `line` where it was defined |
| `pattern` | `Map`, `MapReduce`, `Reduce 1D-Full` (a vector or a matrix to one value), `Reduce 1D-Partial` (rows or columns of a matrix to a vector), `Reduce 2D`, `Scan`, `MapOverlap 1D`, `MapOverlap RowWise`, `MapOverlap ColWise`, `MapOverlap 2D`/`3D`/`4D`, `MapPool 2D`/`3D`/`4D` (an overlap skeleton with a pool), `MapPairs`, `MapPairsReduce` |
| `elements` | the extents of the iteration space, **the same on every backend**: the output of a Map, MapReduce and Scan (a Map over a matrix is its element count, and the stride is taken into account); the input of a Reduce (`size`, or `[rows, cols]` of the matrix as it is read: for columns it is the transpose); the output of a MapOverlap and MapPool of 2 to 4 dimensions; the input of a MapOverlap 1D (RowWise and ColWise: `[rows, cols]`); `[Hsize, Vsize]` of a MapPairs |
| `elwise_inputs`, `proxy_inputs`, `scalar_inputs`, `outputs` | object ids. A call that reduces to a scalar has the ids of the scalars as outputs. A reduction of columns reads the matrix itself (not the transpose that SkePU makes) |
| `prng` | the id of the random stream, if the function draws numbers |
| `backend` | `CPU`, `OpenMP`, `OpenCL`, `Hybrid` |
| `threads` | CPU: 1; OpenMP and Hybrid: the threads of the call |
| `devices` | OpenCL and Hybrid: the devices that the call used |
| `parameters` | what is not the default (below) |

A call on OpenCL is waited for, so its interval is its own. It includes the transfers it makes (a `transfer` between `start`
and `end` belongs to the call; one before `start` was made by something else, such as a host access), and the host
synchronisation before it does not. A call on the Hybrid backend has the share of the work that the device did (`elements`
is smaller than the call), and no event for the CPU share.

### Definitions of the instances

The header has a section `instances`: where each skeleton instance that made a call, and its user functions, are defined in the source. The key is
the `object_id` of the instance (what its calls have), and it is written with the first call of the instance (an instance that makes no call is
not there; the precompiler gives the lines, see `definition` in the generated struct of a user function):

```json
"instances": { "1000": {
    "label": "dotp",
    "instance": { "file": "/src/a.cpp", "line": 12, "end_line": 13 },
    "user_functions": [ { "name": "mult", "kind": "function", "file": "/src/a.cpp", "line": 4, "end_line": 6 } ] } }
```

`instance` is the declaration of a skeleton variable, or the call of an anonymous skeleton (`"anonymous": true`: the same place as its
calls). `user_functions` are the functions that the skeleton was given, in order (a `MapReduce` has the map and the reduce function): `kind`
is `function` or `lambda`; a lambda has no name, and its lines are the lambda expression (on the line of the instance when it is
written in the declaration). The lines are those of the file as it was given to the precompiler (the same as `file` of a call), and `end_line` is the line of the
end of the declaration or the function. An instance in a library (`skepu-lib`) has its lines in the files of the library. In an MPI program only the first process has the table.
`tests/codegen/tracing_definitions.cpp` checks it.

### Parameters

The parameters of the skeleton that change what a call does, as they were when the call was made. Only what differs from the default is recorded:

| skeleton | key | default |
|---|---|---|
| Map | `strides`: one for each container argument, the outputs first | all 1 |
| MapReduce | `strides` (the inputs); `start_value` (a number) | all 1; 0 |
| Reduce | `mode`: `ColWise` (reduction of columns); `start_value` | RowWise; 0 |
| Scan | `mode`: `Exclusive`; `start_value` | Inclusive; 0 |
| MapOverlap | `overlap` (one for each dimension), `stride`, `edge` (`None`, `Cyclic`, `Pad`), `pad`, `update_mode` | stride 1; Duplicate; Normal |
| MapPool | `pool: true`, `pool_size`, `stride` (always) | |

The windows of a pool overlap when the stride is smaller than the pool size, and an element is then read by `prod(ceil(pool_size / stride))`
outputs; a MapOverlap element by `prod(ceil((2 * overlap + 1) / stride))`. That is what the fusion analysis uses for the cost of
recomputing a Map that is fused into them.

## Transfers and host access

`transfer` (`object_id`, `label`, `elements`, `direction`: `host-to-device` or `device-to-host`, `backend`, `start`, `end`): the elements that were
copied (a part of a container if a part was needed). `line` is the line in the SkePU header, not in the program.

`element_access` (`object_id` of the value, `source` the container, `index`, `label`, `file`, `line`): `Vector::elem`.
`external` (`label`, `file`, `line`, `start`, `end`, `outputs`, and the containers it reads as `proxy_inputs`): `skepu::external`, the output of a
scalar, and the operations above that write a container. `snapshot` (`data_file`), `region` (`region_depth`) and `virtual_container` (`object_ids`) are as the macros of the same name.

## MPI programs (a pilot: allocation, `Map`, communication)

A program built with the MPI backend (`skepu-tool -mpi`) writes one file for each process. File names: `skepu_trace.json` for the first
process (rank 0) and `skepu_trace.rank<k>.json` for the others (`%r` in `SKEPU_TRACE_FILE` is the rank). The files are
`{ header, data }`; the header has `mpi` (`rank`, `ranks`, `detail_rank`: 0, `role`: `full` or `lean`, `host`, `pid`) and `clock`: the
clock of the process against that of rank 0, measured at the start and at the end of the program (`start` and `end`, each with
`local_us`, the time of the process when it was measured, `offset_us`, what to add to a time of this process to get that of
rank 0, and `uncertainty_us`, half of the shortest round trip). Times in the files are those of each process.

**Rank 0** has the trace of the program: `allocation`, `deallocation`, `skeleton_call` and `external` as in a single node program, with the ids of the
distributed containers and `elements` for the whole call. A call also has `seq`, the number of the call among the calls of the program
(the same on every process: the processes make the same calls in the same order), and `mpi`: `rank`, `ranks`, `local_elements`,
`work_start` and `work_end` (what the node-level skeleton did on this process), `comm` (the communication that the call made:
`bytes_sent`, `bytes_received`, `time_us`, `collectives`) and `signature` (a hash of the pattern and the label: the processes must agree).
An `external` (a skepu::external call, also what printing a container with `skepu::io::cout` is) with a label and a place has the same `seq` and `mpi` as a call: rank 0 runs the code (its `work_start` and `work_end`), the
others wait for it, and the gather of the containers it reads comes `before` it and the scatter of those it writes `after` it. (A call without a label and
without containers is not recorded, as in a single node program.) `communication` is a collective operation: `comm_seq` (the same on every process), `seq` (the call it is in, 0 for none), `phase` (for a collective in a call: `before` its work, such as the rows that it needs, or `after`, such as the results that go back to their owners), `operation`
(`allgather`, `scatter`, `alltoallv`), `reason` (why the program does it: host access to a distributed container, `skepu::external`, the
rows that a MatRow argument needs, the layout of a call, the halo of an overlap), `object_ids`, `start`, `end`, `bytes_sent` and
`bytes_received` (the payload to and from other processes) and `ranks`.

**The other processes** have lean events only: `rank_call` (`seq`, `pattern`, `signature`, `start`, `end`, `work_start`, `work_end`,
`local_elements`, `comm`) and `rank_communication` (`comm_seq`, `seq`, `phase`, `operation`, `reason`, `start`, `end`, `bytes_sent`,
`bytes_received`), with no ids. Events of the same `seq` or `comm_seq` in the files are the same call or the same collective. The events
of the node-level library (the calls on the parts of the containers, the transfers of the parts) are not recorded, and skeletons
other than `Map` are not traced yet.

A merged trace has the number of events of one process plus a constant number of events for each other process and each call:
nothing in it depends on the size of the problem. `tests/mpi/check_trace.py` checks what is said here.

### Merging the files (SkeVU, `script/backend/mpi.js`)

SkeVU merges the files of a program into the trace of rank 0 (its events, with their `idx`) and adds to each call and each collective what
every rank did, on the clock of rank 0 (the time of a rank plus the offset of its header, interpolated between the start and the end of the
program, so that a clock that runs fast or slow is followed):

* `rank_data`: one entry for each rank (`null` if its file is missing). For a call: `rank`, `start`, `end`, `work_start` and `work_end` (`null`
  for a rank that did no work), `local_elements`, `comm`. For a collective: `rank`, `start`, `end`, `bytes_sent`, `bytes_received`.
* `rank_summary`. For a call: `interval` (from the first start to the last end), `arrival_spread_us`, and `duration`, `work`, `comm_time`,
  `comm_bytes`, `local_elements` as `{ min, mean, max, argmax }`, `slowest_rank` and `imbalance` (the longest work divided by the mean over all ranks, where a rank that did no work counts as zero: 1 is balanced, and the number of ranks means that one rank does it all, as it does in an `external` call). For
  a collective: `interval`, `wait_us` (what each rank waited for the last one to arrive), `max_wait_us`, `last_arrival_rank`,
  `end_spread_us` (how far apart the ranks leave it: what is left of the misalignment of the clocks, for the collectives that no rank leaves before all have arrived (`allgather`); an `alltoallv` lets a rank leave early if it gets nothing from a late one, so the merge does not take its spread as a sign of misaligned clocks), and the bytes in total.

In the graph, a container of an MPI program is in two memory spaces: the tiles of the processes, which the elements of a skeleton are read
from, and the gathered copy, which the host (`external`, `element_access`) and a random access argument read. A `gather`, `allgather` or `bcast`
makes the copy and leaves the tiles as they were; a `scatter`, and an `alltoallv` after a call, make the tiles; a write of the container makes the
copy out of date. An exchange before a call (a halo, the rows of a MatRow) only reads.

The merge also says what is wrong with the files: a missing rank, a rank that makes another call or collective than rank 0
(`divergence`: the program does not do the same on all processes), more calls on a rank than on rank 0, and clocks that are not aligned well
(the collectives end far apart).

## Known limits

* The time of a call that is not on OpenCL is the time of its loop, without the synchronisation with the device before it (a
  `transfer` of its own).
* The host's own reads and writes through `operator[]` and `operator()` outside `skepu::external` do not leave events; a
  dependence through them is not in the trace, except that an OpenCL container they changed is transferred again.
* A call that was fused with `>>` or `||` is one call.
* Sequential programs that are not precompiled have no trace.
