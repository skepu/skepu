# TODO

## skepu-tool: skeletons in function templates with several instantiations

**Status:** postponed until after the upcoming merge, which changes `src/skepu-tool/visitor.cpp` heavily.

**Problem.** A skeleton instance inside a function template (for example in `skepu::blas::scal<TX, TS>`) can only be
instantiated with one set of user functions per program. skepu-tool rewrites the declaration's source text once per
source location, and the rewrite contains a concrete kernel name and concrete types. clang visits the skeleton once per
instantiation. Current behaviour (`transformSkeletonInvocation` in `src/skepu-tool/code_gen.cpp`, `rewrittenCalls` in
`src/skepu-tool/visitor.cpp`): the first instantiation is rewritten, later ones with the same user functions are skipped,
and ones with different user functions abort with "instantiated with different user functions; not supported".
Test: `tests/codegen/template_instances.cpp`.

**Plan.** Make the rewritten text depend on the enclosing template's parameters, so one text serves every instantiation:

1. At the declaration, rewrite the skeleton to name its kernel through the enclosing `FunctionTemplateDecl`'s
   parameters, e.g. `skepu_skel_3_Kernel<TX, TS>` (class templates: include their parameters as well).
2. Before the template, emit a primary `template<typename... Ts> struct skepu_skel_3_Kernel;` and, for each
   instantiation clang reports, a specialization holding that instantiation's generated kernel source, struct
   definitions and constants such as `inPlaceSupported`. Resolve the user functions per instantiation (the visitor
   already does; today the result of the later ones is discarded).
3. Drop the "different user functions" abort.

Works the same for OpenCL and CUDA (class template specialization only).

**Things to watch.**
- All instantiations must be known before anything is emitted, so emission moves to the end of the translation unit.
- `generatedStructs` is global state and has to become per instantiation.
- Print non-type template arguments and `decltype`-based types correctly in the specialization arguments.
- Nested templates: key on the enclosing function's full argument list.
- CUDA output cannot be run here; compare the generated source by eye.

**Tests to add.** The same skeleton in a template instantiated with two different user functions or types (float and
double), on seq, cpu, openmp and opencl; a mixed `float`/`double` `skepu::blas` program; then the full ctest (232
tests; 8 failures are known and also fail on the baseline).

**Alternatives rejected.** Runtime type substitution with `-D` in OpenCL (not possible for CUDA, breaks with structs);
fixed instantiations in the library (only covers the types the library lists).

## SkePY extension modules: constant-time boundary crossing (direction B)

**Problem.** `shedskin build -e` works for SkePU programs, but `skepu.Vector` and `skepu.Matrix` cross the
boundary as lists, copied element by element. Crossing cost is O(n), which makes the modules impractical for
production use. Shed Skin has no numpy support and a Python `list[float]` is an array of boxed floats, so the copy is
the minimum for lists: the fix needs a different Python-side type.

**Direction.** Make the containers real Python objects shared by reference between Python and the module:
- One hand-written wrapper type per element type (`Vector<double>`, `Vector<float>`, `Vector<long>`, and the
  matrix equivalents), written as templates in `shedskin/lib/skepu.hpp` under `__SS_BIND`. Python allocates a container
  once, passes it between module calls (constant time) and gets containers back without a copy. A plain list passed
  where a container is expected is still converted by copying.
- Host access through the buffer protocol (`numpy.asarray(v)`, `memoryview(v)`, `array.array`), with correct
  `shape`/`strides` (Matrix: row-major, or report SkePU's actual layout after `change_layout`).
- SkePU already has `Vector(T* ptr, size, deallocEnabled)` and `Matrix(T* ptr, rows, cols, deallocEnabled)`, so
  wrapping memory it does not own is possible; not needed for the wrapper design below.

**Checked assumptions (code read, nothing built).**
- Lifetime and identity: Shed Skin's extmod already does this for user classes. A wrapper holds `_ss_object`; the global
  `_ss_proxy` dict maps C++ object to wrapper (same object in, same Python object out) and, being reachable from a
  global on the GC heap, keeps the C++ object alive until `tp_dealloc` removes the entry. Reuse the pattern for the
  builtin container templates. `_ss_proxy` is per module, so identity does not extend across modules.
- Annotations are ignored by Shed Skin (`graph.py`, "skip type annotations"). Parameter types come only from
  inference over calls in the module (the `__main__` block), so one call site means one C++ signature per exported
  function. No annotation syntax is needed for container parameters; what extmod does for a function called with two
  element types is unknown.
- numpy is not installed in the development Python 3.14; test the protocol with `array.array`/`memoryview`.

**Coherence (dirty flag).** Python writes through a raw pointer cannot be intercepted, but only "may have been written
since the last sync point" matters, and the sync points are module calls. Buffer exports delimit Python access exactly
(`bf_getbuffer`, `bf_releasebuffer`; CPython refcounting is deterministic, and numpy keeps the exporting buffer alive
as its base). Per container: `exports` (open writable exports) and `dirty` (a writable export has existed since the
last device invalidation).
- Export: `updateHost()`, `exports++`, `dirty = true`.
- Release: `exports--`; `dirty` stays set.
- Argument unwrapping at a module call: if `dirty || exports > 0`, `invalidateDeviceData()` (exists in SkePU); clear
  `dirty` when `exports == 0`. O(1) on CPU backends, one re-upload on GPU backends.
- Writes through the wrapper's own methods (`v[i] = x`) set `dirty`.
- Without views and with `dirty` clear the data stays device-resident between calls (the case the design aims for).
- Reverse direction: when a skeleton writes a container that has an open view, the module refreshes it with
  `updateHost()` when the call finishes (O(#exported) containers).
- `init`, `swap` and resizing raise `BufferError` while `exports > 0` (pointer stability, as `bytearray`).
- numpy/memoryview request buffers read-only-capable and the exporter decides writability, so a view that is only
  read still counts as writable: every call re-uploads while it lives on a GPU backend. Mitigations: a read-only view
  API (reported read-only, no flag) and a scoped `with v.host() as a:` block.
- Unsound by design: a raw address kept past release (`__array_interface__`, ctypes tricks); do not offer it.

**Not solved by this.**
- Per-call overhead: SkePY creates skeleton objects inside functions, so every call constructs them. The cost,
  especially on OpenCL, is unmeasured; hoisting skeleton construction to module level may be needed.
- Several SkePY modules each link their own SkePU, Shed Skin runtime, GC and (OpenCL) context, and their container
  types are different Python types; exchange between modules works only through the buffer protocol unless a shared
  runtime extension owns the types.
- A host view forces a download; zero-copy GPU hand-off would need DLPack or `__cuda_array_interface__` (CUDA only).

**Plan.**
1. Benchmark: list conversion against a hand-wrapped buffer, plus per-call skeleton construction cost, across sizes and
   backends.
2. Vector wrapper for `double`/`float` with buffer protocol, identity-preserving wrap/unwrap and the rules above.
3. Tests with the existing driver mechanism (`tests/skepu/*.driver.py`): write through a view and see the skeleton
   read it; write with a skeleton and see the view; no copy, no manual `updateHost`; crossing time independent of
   `n`.
4. Matrix, dtype checks (a `--float32` build rejects a float64 buffer or converts explicitly, never silently).
5. Hoisted skeleton construction if step 1 shows it matters; shared runtime for several modules last.

Research angle: the coherence rule is a protocol between the Python program, the device and SkePU's lazy transfers;
the open question is the transfer cost of each access pattern against explicit `sync()` calls. Cross-call fusion (a
recording mode feeding SkePU's fusion support) is speculative follow-on work.

## SkePY and SkeVU

**Status.** `shedskin build --trace` gives traces for SkeVU with Python file and line, the names of skeletons and
containers, scopes, snapshots, virtual data sets, external blocks and deallocations (see `SKEPY.md`, "Traces for
SkeVU"). Still open:

- **More names.** Containers are named after the variable of an assignment; containers made in other ways (returned
  from functions, in lists) and tensors are not named. `skepu.external` has the label "[external]" (a `label=`
  argument needs a numbered default in `skepu.hpp` like the other optional arguments).
- **Library calls.** The skeletons that `skepu.blas` runs are attributed to `blas.hpp`; attributing them to the
  Python call (or a region named after the BLAS function) would help.
- **Deallocation timing.** The events come from finalizers (when the garbage collector finds a container unused, and
  at exit), not from the end of the Python lifetime; `del` is not supported by Shed Skin. Escape analysis could give
  exact events for local containers that do not escape.
- **Memory.** The finalizers are only registered with `--trace`. Without them the container destructors never run,
  so the host memory (and device buffers) of every container that SkePY creates is never freed, and the collector
  does not see that memory (it has no idea how much each container holds), so collections are rare: a long running
  program or extension module that creates containers in a loop leaks. Consider registering the finalizers in every
  build and telling the collector the size (`GC_malloc` accounting, or an explicit `GC_gcollect` heuristic).
- **Extension modules.** Each module has its own tracer and writes the file when the interpreter exits, so two
  modules overwrite each other; needs a flush call and per-module file names.
- **Regions are global.** The tracer's region stack is shared by all threads; events from several threads are safe
  (locked) but their regions interleave.

## SkeVU tests

The JavaScript tests in `visualizer/tests/` are stale: `tests/helpers.js` and the older unit tests import
`../backend-js/...`, which is now `script/backend/`, so they fail at import. `vitest` also does not start on the
development machine (the native `rollup` module for this platform is missing from `node_modules`; `npm install`
fixes that). `tests/unit/test_virtual_snapshots.js` uses the current paths, but has only been run with a small
stand-in for vitest.
