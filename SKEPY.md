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

> The SkePY repository has no published remote yet (branch `main`, based on Shed Skin v0.9.14-1).

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

`visualizer/examples/cellular_automaton_skepy/` is an example. Tests: `tests/skepu/test_trace*.py`,
`test_release.py`, `test_gc_pressure.py`, `test_ext_trace.py`.

### Random numbers and pooling

* `skepu.randomize` is SkePU's `skepu::randomize`: `set_prng(prng)` (the stream of the fills; `reset_prng()`),
  `uniform(container, lo, hi)`, `normal(container, mean, deviation)`, `bernoulli(container, p)` and
  `poisson(container, lambda)`, for vectors, matrices and tensors; the numbers are the same on every backend.
* `skepu.MapPool(f)` is the pooling skeleton: the user function gets a `Pool2D`, `Pool3D` or `Pool4D` (the window at
  the element: `pool[i, j]`, `pool.si`, ...), with `set_pool_size(...)`, `set_stride(...)` and
  `set_pool_size_and_stride(...)`. (1D pooling is not supported by SkePU's own backends.)

### Neural networks: `skepu.dnn`

`skepu.dnn` is SkePU-DNN, the C++ library `skepu-lib/ml.hpp` (every layer, loss and optimizer step is a
skeleton call), bound as a whole: the layers are plain configuration objects and the model is the C++ one.

```python
model = skepu.dnn.SequentialModel(batch, 28, 28, 1)          # batch size, height, width, channels
model.add(skepu.dnn.Flatten())
model.add(skepu.dnn.Dense(128, skepu.dnn.INIT_He_uniform, skepu.dnn.INIT_Zero, skepu.dnn.ACT_ReLU))
model.add(skepu.dnn.Dense(10, skepu.dnn.INIT_Xavier_uniform, skepu.dnn.INIT_Zero, skepu.dnn.ACT_SoftMax))
model.init()
model.configure(skepu.dnn.LOSS_CategoricalCrossEntropy, skepu.dnn.OPT_SGD)
model.fit(train_x, train_y, epochs, learning_rate, 0.1)      # validation split 0.1; Tensor4 of floats
loss, accuracy = model.evaluate(test_x, test_y)              # or evaluate_confusion(x, y, matrix)
```

* Layers: `Dense`, `Conv2D(filters, kh, kw)`, `Conv1D(filters, k)`, `MaxPooling2D(kh, kw)`, `AvgPooling2D`,
  `MaxPooling1D(k)`, `AvgPooling1D`, `Flatten`, `Dropout(rate)`; the attributes are the arguments `weights`, `biases`
  (`INIT_*`) and `activation` (`ACT_*`) of the layer functions. A model also has `predict`, `summary` and `save`.
* `pretrained="file.csv"` is an argument of `Dense`, `Conv2D` and `Conv1D`; `skepu.dnn.load_model(path, batch)` makes a
  model from a file written by `model.save(path)` (or described by hand, see `examples/dnn/models/`).
* Data preparation, all by skeleton calls: `chw_to_hwc(data)`, `channel_stats(images)` (with `.mean()`, `.stddev()`) and
  `standardize(images, stats)`, and the loaders `cifar10_load(files, max_count)` (the images and the classes)
  and the MNIST ones below.
* Data: `one_hot(classes, num_classes, n=0)`, `argmax(scores)`, `take(data, n)`, `as_sequence(images, n=0)`, and the MNIST
  files, `mnist_images(path, lo, hi)` and `mnist_labels(path)`.
* The data are `Tensor4` of floats and the model is float-only, so the program is built with `--float32`
  (a translation without it stops with a message at C++ compile time).
* The errors of the library are Python `RuntimeError`s with the message of the library.
* `shedskin/examples/skepu/dnn_mnist10.py` is the translation of `examples/dnn/dnn_mnist10.cpp`; its arguments name the
  directory of the MNIST files first (`examples/data/mnist`).

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
* The upstream Shed Skin test suite passes except for 16 tests that also fail with unmodified Shed Skin.

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

* `CMake/SkePUHelpers.cmake`: `skepu_add_executable` accepts absolute source paths and
  `PRECOMPILE_OPTIONS`; `skepu_add_library` (STATIC, SHARED, MODULE) shares the precompilation step.
* `src/skepu-tool`: tolerates skeleton calls not made through a variable and user functions that
  call explicit template specializations; skeleton instances in function templates with several
  instantiations are supported (selector class template; tests `tests/codegen/template_instances*.cpp`).
* `tracing.h`, `tracing.hpp`: tracing works in programs that mix precompiled and plain sources
  (`tests/codegen/tracing_multi_*.cpp`), is thread-safe, nests regions per thread, can be flushed
  (`SKEPU_TRACE_FLUSH()`), and takes file and line as arguments for non-C++ frontends.
  `SKEPU_TRACE_MODULE=name` names the trace file of a module.
* `CMakeLists.txt`: `cmake --install` finds the `skepu-tool` export files and installs `nlohmann/json.hpp`.
* `blas.hpp`, `complex.hpp`, `mapoverlap_2d.h`, `vector.hpp`, `mappairs_*`: correctness fixes found
  with SkePY programs, each with a test under `tests/`. Strided `Map` indices count positions in the
  strided sequence on every backend. `MapPairs::setInPlace(true)` is implemented on OpenCL for a
  single output of arithmetic type; CUDA reports an error.

Known SkePU issue that SkePY runs into, not fixed here: user functions named like
OpenCL C builtins (`half`, `trunc`, ...) do not compile for OpenCL.

Known, not fixed: `skepu-tool`'s output is not deterministic (the order of the
generated user-function structs varies between runs); it does not affect the
results.

## License

SkePY is a fork of [Shed Skin](https://github.com/shedskin/shedskin) and keeps its licenses where it changes its files: the compiler is
GPL-3.0-only, including our changes to it, which is why SkePY is a separate repository and not part of SkePU (it is not a submodule).
The SkePU support that is new (the `skepu` module and its C++ wrappers in `shedskin/lib`, the examples and the tests) is under the SkePU license
(`LICENSES/LicenseRef-SkePU.txt` in the SkePY repository), by Axel Hammarberg, who made the first implementation, and August Ernstsson. The programs
that SkePY generates contain neither compiler nor GPL code. The SkePU changes needed by SkePY, in this repository, are under SkePU's license.

