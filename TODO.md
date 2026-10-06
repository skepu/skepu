# TODO

## skepu-tool: remaining limits of skeletons in templates

The selector based rewrite (see `SKEPY.md`) covers function templates at namespace scope. Not covered:
member function templates and members of class templates (the code is placed inside the class; several
instantiations with different user functions are rejected with an error), template template parameters and
unnamed template parameters in the enclosing function template (error), and template arguments that name a
type declared after the template. The CUDA output of the selector (`static constexpr auto cuN = &kernel;`)
could not be run here and was only checked by reading the generated source.

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
containers, scopes, snapshots, virtual data sets, external blocks, BLAS regions and deallocations (see `SKEPY.md`,
"Traces for SkeVU"). Still open:

- **Deallocation timing.** Only a local container that is assigned once at the top of a function and does not escape
  is freed (and its event made) when the function ends. Containers made in loops or branches, returned ones and
  module variables are freed when the collector finds them unused (`del` is not supported by Shed Skin). A loop
  body could free its containers at the end of each iteration with the same analysis.
- **Names.** Containers that are created in other places than an assignment, a list, a tuple or a return are named
  by their line; `print` with `sep=` or `end=` is not traced; the label of a skeleton that is not called through a
  variable is "Map(function)".
- **Modules.** The modules of a program that has several traced extension modules write one file each; SkeVU loads one
  trace, so they cannot be shown together.

## SkeVU tests

The JavaScript tests in `visualizer/tests/` pass (172 tests) but `vitest` does not start on the development machine:
the `rollup` module in `node_modules` is the one for x64, not arm64 (`npm install` fixes that; it needs the network).
They were run with a small stand-in for `vitest`.
