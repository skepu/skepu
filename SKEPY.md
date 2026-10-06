# SkePY: a Python frontend for SkePU (optional, experimental)

SkePY lets you write SkePU programs in a restricted subset of Python. A
modified [Shed Skin](https://github.com/shedskin/shedskin) translates the
program to SkePU C++ (the `skepu` module in `shedskin/shedskin/lib/skepu.py`
mirrors the C++ API: `Vector`, `Matrix`, `Map`, `MapReduce`, `Reduce`, `Scan`,
`MapOverlap`, `MapPairs`, ...), which is then built like any other SkePU
program.

SkePY lives in its **own git repository**, not in this one. This repository only
carries this note and any patches to SkePU proper that SkePY needs (none so
far). It is not a submodule: clone it by hand into `shedskin/` (that path is in
`.gitignore`):

```bash
git clone <SkePY repository URL> shedskin
```

> The SkePY repository has no published remote yet; for now it exists only
> locally (branch `skepu-0.9.14-1`, based on upstream Shed Skin v0.9.14-1).

## Requirements

* Python 3.10 or newer (developed with 3.14), CMake, a C++20 compiler.
* The Boehm GC and PCRE2 libraries. `shedskin build` builds both into
  `~/Library/Caches/shedskin` (macOS) on first use, which downloads their sources.
* A SkePU installation for the CMake route (see below).

## Using it

```bash
cd shedskin
python3 -m shedskin translate examples/skepu/test_map_scan_overlap.py   # Python -> C++
```

Programs that `import skepu` are recognised automatically; all other Python
programs are translated exactly as by upstream Shed Skin.

Two ways to build the generated C++:

1. **`shedskin build`** (CMake, `skepu_add_executable`; verified with the OpenMP
   backend on macOS): SkePU programs are precompiled with `skepu-tool` and built
   against an installed SkePU. On macOS Apple clang has no OpenMP, so use a
   compiler that does, as the SkePU build here does with Homebrew GCC:

   ```bash
   cmake --install build --prefix ~/skepu-install        # an existing SkePU build
   cd shedskin
   CXX=g++-16 CC=gcc-16 CMAKE_PREFIX_PATH=~/skepu-install \
       python3 -m shedskin build examples/skepu/test_map_scan_overlap.py
   ./build/test_map_scan_overlap 10          # the program's own argument
   ```

2. **By hand, sequential backend** (no `skepu-tool` needed): compile the generated
   `.cpp`, `shedskin/shedskin/lib/{skepu,builtin,sys}.cpp` with `-std=c++20`,
   include paths for `skepu-headers/src`, `shedskin/shedskin/lib` and the GC/PCRE2
   headers, and link `-lgc -lgccpp -lpcre2-32`.

**OpenCL** (verified on an Apple GPU): add `--opencl`, and force the backend in the
program, e.g. `skepu.set_global_backend_spec(skepu.BackendSpec("opencl"))`
(the default `Auto` picks a backend by itself). GPUs without double precision
(Apple Silicon) need 32-bit floats, `--float32`:

```bash
CXX=g++-16 CC=gcc-16 CMAKE_PREFIX_PATH=~/skepu-install \
    python3 -m shedskin build --opencl --float32 examples/skepu/test_arithmetic_backends.py
./build/test_arithmetic_backends 1000 opencl
```

### Python extension modules

`shedskin build -e` (`--extmod`) builds a SkePU program as a Python extension module,
`NAME.so`, that CPython can import (the same options and backends as for programs, e.g.
`-e --opencl --float32`). This is the intended use of SkePY: write the performance
sensitive parts as a module, use SkePU in it, and call it from ordinary Python.

```python
# vecmod.py (translated)               # user.py (CPython)
import skepu                           import vecmod
def axpy_f(x, y, a): return a*x + y    vecmod.set_backend("opencl")
def set_backend(name: str) -> None:    print(vecmod.axpy(2.0, [1.0], [3.0]))
    skepu.set_global_backend_spec(skepu.BackendSpec(name))
def axpy(a: float, xs: List[float], ys: List[float]) -> List[float]: ...
if __name__ == '__main__':             # (calls every function once: Shed Skin infers
    ...                                #  the types of the exported functions from them)
```

What crosses the boundary (always as a copy): numbers, strings, lists, tuples (Shed Skin's own conversions),
`skepu.Vector` (as a list) and `skepu.Matrix` (as a list of rows), also as
attributes of exported classes (a class can keep its data in a SkePU container between
calls). Exceptions raised in the module
reach the caller. Not exported, with a warning at translation time: functions that take
or return a `skepu.dataclass` struct (a plain struct without a Python object; it can be
used inside the module), `Tensor3`/`Tensor4`, and skeleton objects. The module must
choose the backend itself (`skepu.set_global_backend_spec`), e.g. through an exported
function as above. The module needs the same `skepu-tool` build as a program
(`skepu_add_library` in `CMake/SkePUHelpers.cmake`); tests: `tests/skepu/test_ext_*.py`
with a `.driver.py` that CPython runs.

SkePU programs imply `-D__SS_NOZERO` (no zero-division exceptions): user
functions must not throw, which `skepu-tool` enforces and the GPU backends need.

Options added for SkePU: `--cuda` (CUDA backend and CUDA math calls) and
`--fastmod` (C-style `%`).

### Traces for SkeVU

SkePU's tracing (the JSON traces that SkeVU shows, `visualizer/`) works for SkePY programs on
the `cpu`, `openmp` and `opencl` routes (not `seq`: tracing needs the `skepu-tool` output):

```bash
CXX=g++-16 CC=gcc-16 CMAKE_PREFIX_PATH=~/skepu-install \
    python3 -m shedskin build --trace examples/skepu/cellular_automaton.py
SKEPU_TRACE_FILE=trace.json build/cellular_automaton 2 12 4 0.4 openmp
```

`--trace` builds with `SKEPU_TRACING` and adds to the generated code what the trace needs:

* every skeleton call, `with skepu.external(...)` block and the calls below tell the trace
  where they are in the Python source (file and line), so SkeVU can show the source: load
  the trace and the `.py` file (Files > Load Trace);
* skeletons are named after the variable they are called through (`update(...)`: "update";
  one that is not called through a variable is "Map(function)") and containers after the
  variable they are assigned to (`domain_a = skepu.Vector(...)`: "domain_a"), the elements of
  a tuple assignment after their variables, the elements of a list "pair[0]", a returned
  container after the function ("make()"), and any other one after its line
  ("Vector (line 15)"); `skepu.external(..., label="Read input")` names an external block;
* a call of a BLAS routine (`blas.axpy(...)`) is a region "blas.axpy" at the call in the
  Python source, with the skeletons that the routine runs inside it;
* the program can add to the trace, as in the C++ API (`SKEPU_TRACE_*`); all of these do nothing
  in a build without `--trace`:

| SkePY | C++ |
|---|---|
| `with skepu.scope("Iteration"): ...` | `SKEPU_TRACE_SCOPE`: a region around the calls in the block |
| `skepu.snapshot(container)` | `SKEPU_TRACE_SNAPSHOT`: the contents, in `NAME@N.txt` files in the working directory (`SKEPU_TRACE_SNAPSHOT_DIR`; SkeVU loads them from a zip) |
| `skepu.virtual_dataset("Domain", a, b)` | `SKEPU_TRACE_VIRTUAL_DATASET`: two containers shown as one data set |
| `with skepu.external(read=..., write=...): ...` | `SKEPU_TRACE_EXTERNAL`: made automatically, with the containers read and written |
| `print(container)` | `skepu::io::cout << container`: an "I/O" external event that reads the container, at the line of the `print` (made automatically; `print` with `sep=`/`end=` is not traced) |

Containers also get a deallocation event. A local variable that is assigned a new container
once, at the top level of a function, and is used only in ways that cannot let the container
outlive the function (indexing, a few methods, an argument of a skeleton or BLAS call, `print`,
`len`) is freed when the function ends, as Python does, and the event is made then. Other
containers (returned, referred to by another variable, in a list, assigned in a loop or branch,
module variables) are freed by the garbage collector, which runs no destructors itself, so SkePY
runs the containers' destructors from finalizers: when the collector finds a container unused,
and at exit for what is unused then (containers that module variables still refer to get none).
The finalizers are registered in every build, not only with `--trace`: without them the memory
of the containers would never be freed. The collector cannot see the containers' data (SkePU
allocates it), so the host memory taken since the last collection is counted and a collection is
made after 256 MiB.

An extension module built with `--trace` has a tracer of its own: its trace is written to
`skepu_trace_<module>.json`, or to `SKEPU_TRACE_FILE` with `%m` replaced by the name of the module
(a program with several modules should use it), and `skepu.trace_flush()` writes the trace so far.

`visualizer/examples/cellular_automaton_skepy/` is an example (`examples/skepu/cellular_automaton.py`,
which has the same trace calls as the C++ example). Tests: `tests/skepu/test_trace.py` (calls marked
in the source must be in the trace, with their lines), `test_trace_events.py` (names, regions,
snapshots, external blocks, virtual data sets, deallocations), `test_trace_names.py` (the other
names, external labels, BLAS regions), `test_release.py` (which containers are freed when),
`test_gc_pressure.py` (containers in a loop are freed during the loop) and `test_ext_trace.py` (a
traced extension module).

## Status

* Verified with both the sequential backend and the OpenMP backend (CMake route): `Map` (up to 3 inputs, 4+ mixed-type uniforms,
  dataclass uniforms, proxies, `Index2D`), `MapReduce`, `Reduce` (1D/2D),
  `Scan`, `MapOverlap`, `skepu.Constant`. Examples with expected output are in
  `shedskin/examples/skepu/`.
* OpenCL is verified for the same features (`Map` with several inputs, uniforms,
  dataclass and `Constant`, `MapReduce`, `Reduce`, `Scan`, `MapOverlap`, proxies),
  and for integer and float arithmetic in user functions (`//`, `%`, `**`, `abs`,
  `min`, `max`); results equal the sequential and OpenMP ones.
* `skepu.external` (`with skepu.external(read=..., write=...):`, also with several
  and mixed containers) and the smart containers (`Vector`, `Matrix`, `Tensor3`,
  `Tensor4`: construction, copies, aliases, `swap`, host/device synchronisation,
  `flush`, deallocation, many short-lived containers) are covered by tests on the
  sequential, OpenMP and OpenCL backends.
* `MapPairs` (single and multiple inputs, index, uniforms), `MapPairsReduce`
  (row-wise and column-wise) and `Random` (with `set_prng`) are verified on the
  sequential, OpenMP and OpenCL backends by self-checking tests.
* Not tested: the CUDA backend (no CUDA device available). `skepu-tool -cuda`
  generates code for all test programs, but it has not been compiled or run.
* User functions are limited to what can be written in plain C: numbers,
  comparisons, conditionals, `while` and `for ... in range(literal step)` loops,
  `and`/`or`/`not` in conditions, arithmetic including `//` `%` `**`, `float()`/`int()`,
  `abs`/`min`/`max` on two numbers, the `math` functions known to the CUDA/OpenCL
  mapper, SkePU containers/indices/regions and dataclasses. Lambdas work as user functions
  for every skeleton (`skepu.Map(lambda a, b: a + b)`); like functions they cannot capture
  variables of the enclosing function or read module variables other than `skepu.Constant`
  (pass them as uniforms). Lists, comprehensions, strings, `print` and the like are rejected
  with an error at the source line. Skeleton instances
  must be stored in variables (`m = skepu.Map(f)`, not `skepu.Map(f)(...)`).
* The upstream Shed Skin test suite: 105 of 121 pass; the other 16 fail
  identically with unmodified Shed Skin v0.9.14-1 on the same machine, apart
  from one regression that was fixed.

## Tests and examples

`shedskin/tests/skepu/` has the tests (self-checking Python programs with `.expected`
output) and `shedskin/examples/skepu/` the translations of SkePU's examples, whose
output is compared with the C++ originals. `tests/skepu/run_tests.py` builds and runs
them on the backends `seq` (SkePU's header-only implementation, no skepu-tool), `cpu`,
`openmp` and `opencl`:

```bash
cd shedskin
python3 tests/skepu/run_tests.py                       # sequential only, nothing to set up
python3 tests/skepu/run_tests.py --suite all --backend all \
    --skepu-prefix ~/skepu-install --cxx g++-16 --cc gcc-16
```

All programs pass on all four backends (180 runs, OpenCL on an Apple GPU).

## Design notes

* For programs that import `skepu`, tuple literals get length-specific types
  with per-position element types (and same-typed pairs/triples stay tuples
  instead of becoming vector-backed), because starred arguments such as
  `mapf(*ix(), *inargs, ...)` have to be expanded during type inference.
  Other programs keep upstream's tuple types.
* While a user function is generated, types are plain C (`long`/`int`,
  `float`/`double`, `unsigned char` for booleans) and literals are C casts,
  because `skepu-tool` copies the function text into the OpenCL/CUDA kernels,
  where Shed Skin's typedefs (`_ss_int`, `_ss_float`, `_ss_bool`) and helper
  templates do not exist. Dataclass structs are emitted at global scope for the
  same reason.
* Generated code uses `_ss_int`, `_shedskin_::` and `_skepu_*` (single
  underscore) instead of Shed Skin's reserved double-underscore names; the
  rename is a single mechanical commit in the SkePY repository
  (`scripts/rename_reserved_identifiers.py`).

## Changes in SkePU proper that SkePY needs

* `CMake/SkePUHelpers.cmake`: `skepu_add_executable` accepts absolute source
  paths and a `PRECOMPILE_OPTIONS` list that is passed to `skepu-tool`.
  `skepu_add_library` (an empty stub) is implemented (STATIC, SHARED or MODULE, same
  arguments as `skepu_add_executable`); the precompilation step is shared.
* `src/skepu-tool/visitor.cpp`: do not crash on skeleton calls that are not made
  through a variable (SkePY calls skeletons through a member of a wrapper class).
* `src/skepu-tool/code_gen.cpp`: do not crash on user functions that call
  explicit template specializations (Shed Skin's arithmetic helpers).
* `CMakeLists.txt`: `cmake --install` found no `skepu-tool` export files with
  recent CMake (they are written to a hashed directory); they are now located
  at install time.

* `src/skepu-tool/code_gen_cl.cpp`, `mapreduce_cl.cpp`: a function that is called from
  several places in a skeleton's user functions was written into the OpenCL source
  several times ("redefinition"); it is written once.
* `src/skepu-tool/code_gen_cl.cpp`: a `MatCol` argument of a `Map` with a vector output
  used the output width (1 for a vector) to find its column on the OpenCL backend, so
  every element got the sum of column 0; it uses the number of columns of the matrix
  (test: `tests/codegen/proxies.cpp`).
* `skepu-headers`: two leftover debug prints in the random number code removed.
* `src/skepu-tool/data_structures.cpp`: a user type that is only the return type of a
  user function got no struct definition in the OpenCL source (test:
  `tests/codegen/usertype_return.cpp`).
* `skepu-headers/.../backend/mapoverlap/mapoverlap_2d.h`: "mirror mode" (output of the
  input's size, only the interior computed, OpenMP only) was switched on for every
  same-size call, so the OpenMP backend left the border unwritten with the edge modes
  Pad/Cyclic/Duplicate; it is now limited to edge mode None, compares the columns with
  the input's, and no longer prints on every call (test:
  `tests/mapoverlap/mapoverlap_2d_edge_modes.cpp`).
* `src/skepu-tool/code_gen.cpp`, `visitor.cpp`: a skeleton instance in a function template
  was rewritten once per instantiation, over the same source text, which commented out
  the skeleton's setup and call (the skeleton then did nothing on the CPU/OpenMP
  backends). It is now rewritten once, in the template, and names its kernels through a
  selector class template with the template parameters; each instantiation gets a
  specialization of the selector (instantiations with the same user functions share
  their kernels), so instantiations with different types and user functions work. Non-type
  template arguments of user function templates are substituted in the user function code.
  Only function templates at namespace scope can have several instantiations with different
  user functions; in a member function template that is an error (tests:
  `tests/codegen/template_instances*.cpp`, `tests/skepu_lib/blas_mixed_precision.cpp`).
* `skepu-headers/src/skepu3/impl/tracing.h`, `tracing.hpp`: a program with several source
  files, some precompiled and some not (the runtime sources of a SkePU based language), aborted
  at exit when built with `SKEPU_TRACING`: the header switches tracing off in files that are
  not precompiled, so the tracer was a different class in the two kinds of file, and the
  linker used the destructor of one for objects of the other. The two variants are now in
  different inline namespaces. The default tracer was also one object per source file, all
  writing the same trace file (the last to be destroyed won); it is now one object for the
  program, created on first use (test: `tests/codegen/tracing_multi_*.cpp`).
* `skepu-headers/src/skepu3/impl/tracing.hpp`: the tracer takes events from several threads
  (it had no lock: concurrent `push_back`s crashed it; test: `tests/codegen/tracing_threads.cpp`),
  and no longer prints "Tracer destructor ..." to stdout when the program exits.
* `skepu-headers/src/skepu3/impl/tracing.h`, `tracing.hpp`: regions nest per thread (the events
  are replayed in recording order and the region stack was one for the program, so regions of
  threads that overlap in time nested into each other; test: `tests/codegen/tracing_regions_threads.cpp`);
  `Tracer::flush()` / `SKEPU_TRACE_FLUSH()` write the trace so far; a module built with
  `SKEPU_TRACE_MODULE=name` writes `skepu_trace_<name>.json` by default, and `%m` in
  `SKEPU_TRACE_FILE` is the name of the module (test: `tests/codegen/tracing_file_names.cpp`).
* `visualizer/tests/`: the unit tests imported `backend-js/` (now `script/backend/`) and read
  fixtures that had never been committed; the paths are fixed, the four fixtures are recreated, and
  `GraphSettings` has a default `mode` (without one `toCytoscape()` emitted no edges); `npm test` in
  `visualizer/` runs them (172).
* `skepu-headers/src/skepu3/impl/tracing.h`, `tracing.hpp`: `beginRegion`, `data_snapshot` and
  `virtual_dataset` can take the file and line as arguments (for programs that are not C++:
  SkePY gives the place in the Python source); the snapshot code no longer prints to stdout.
* `visualizer/script/backend/visualizer.js`: the snapshot stepper of a virtual data set missed
  snapshots (`A@1 B@1 B@2 B@3 B@4`): with the iterations coalesced and the regions shown a container
  is updated by several nodes (its initialisation and the loop's updates) and only the first
  one's snapshots were used (test: `visualizer/tests/unit/test_virtual_snapshots.js`).
* `visualizer/`: the source pane takes `.py` files (shown as plain text); example with a SkePY
  program.
* `CMakeLists.txt`: `cmake --install` did not install `external/nlohmann/json.hpp`, which the
  tracing code includes, so `SKEPU_TRACING` did not compile against an installed SkePU.
* `skepu-headers/src/skepu-lib/blas.hpp`: `gemv` did not compile for the skepu-tool
  backends (the `Index1D` parameter was not qualified), ignored `alpha` for `NoTrans`,
  computed a wrong `Trans`/`ConjTrans` product and assumed square matrices; `scal` had
  a different user function for a `float` and a `Scalar<float>` alpha (what `dot` results
  give). Fixed (test: `tests/skepu_lib/blas_float.cpp`, all four backends).
* `skepu-headers/src/skepu-lib/blas.hpp`, `complex.hpp`, checked by `tests/skepu_lib/blas_level1.cpp`,
  `blas_level23.cpp` (seq, CPU, OpenMP, OpenCL) and `blas_complex.cpp` (no OpenCL):
  `nrm2` did not compile for real types (`real`/`imag` of a real number added);
  `iamax` did not compile on the skepu-tool backends; `gemm` was wrong for every operation
  that is not NN or TT-like with non-square matrices (one user function with random access
  to A and B); `ger`/`geru` used MapPairs' in place mode, which the OpenCL and CUDA
  backends do not implement (see below), and now add a temporary outer product, which works
  on every backend.
* `skepu-headers/src/skepu3/vector.hpp`: the index of an element in a strided `Map` or
  `MapReduce` was its position in the vector on the sequential, CPU and OpenMP backends but
  its number in the strided sequence on OpenCL and CUDA; it is the latter everywhere (test:
  `tests/map/map_stride_index.cpp`). This is what made strided `gemv` and `iamax` wrong.
* `src/skepu-tool/mappairs_cl.cpp`, `skepu-headers/.../backend/impl/mappairs/mappairs_cl.inl`,
  `mappairs_cu.inl`: `MapPairs::setInPlace(true)` was silently ignored by the OpenCL and CUDA
  backends (wrong results). OpenCL now implements it for a single output of an arithmetic
  type (the output is uploaded, the kernel adds) and reports an error for other outputs;
  CUDA reports an error (not implemented, no device here to test it). Test:
  `tests/mappairs/mappairs_in_place.cpp` (seq, CPU, OpenMP, OpenCL). `ger`/`geru` still
  go through a temporary so that they work with CUDA too.
* `skepu-tool` (`data_structures.cpp`): the struct of a multi-valued return type's
  elements, not `std::tuple` itself, is looked up for the OpenCL source.
* `examples/conjugate_gradient.cpp`: made to work (random non-symmetric matrix, fixed
  5 iterations, no output): symmetric diagonally dominant `A`, iteration to a tolerance, the
  solution printed. `skepu.blas` (Python) is documented in `shedskin/examples/skepu/README.md`.
* `examples/coulombic.cpp`: the atoms are generated by a `Map` with `Random<4>` instead
  of libc `rand()` (whose sequence differs between platforms).
* `examples/cellular_automaton.cpp`: set a PRNG (the example aborted) and made the rule
  tables large enough for the neighbor counts (out-of-range reads).

Known SkePU issue that SkePY runs into, not fixed here: user functions named like
OpenCL C builtins (`half`, `trunc`, ...) do not compile for OpenCL.

Known, not fixed: `skepu-tool`'s output is not deterministic (the order of the
generated user-function structs varies between runs); it does not affect the
results.
