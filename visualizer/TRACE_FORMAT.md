# The trace format

What `SKEPU_TRACING` writes (`skepu3/impl/tracing.h`, `tracing.hpp`) and what SkeVU and the analysis server read: a JSON array of
events, in the order in which they were recorded (a trace is written when the program ends, or by `SKEPU_TRACE_FLUSH()`).
`tests/codegen/tracing_skeletons.cpp` checks the parts of it that are described here, on every backend.

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

## Known limits

* The time of a call that is not on OpenCL is the time of its loop, without the synchronisation with the device before it (a
  `transfer` of its own).
* The host's own reads and writes through `operator[]` and `operator()` outside `skepu::external` do not leave events; a
  dependence through them is not in the trace, except that an OpenCL container they changed is transferred again.
* A call that was fused with `>>` or `||` is one call.
* Sequential programs that are not precompiled have no trace.
