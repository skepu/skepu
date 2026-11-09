# SkePY: a Python frontend for SkePU (optional, experimental)

SkePY translates SkePU programs written in a restricted subset of Python to SkePU C++, with a modified
[Shed Skin](https://github.com/shedskin/shedskin). The translated programs are built like any other SkePU program.

SkePY is its own git repository (branch `main`, based on Shed Skin v0.9.14-1), not a submodule. Clone it into `shedskin/`, which is in `.gitignore`:

```bash
git clone <SkePY repository URL> shedskin
```

Its documentation is `shedskin/docs/skepy.md`: building, the language, the libraries, traces and the tests. The translations of SkePU's examples are in
`shedskin/examples/skepu/`.

## What SkePY needs from SkePU

A change to these needs a matching change in SkePY:

* `src/skepu-tool`: the translated code is precompiled by it. SkePY depends on skeleton calls that are not made through a variable, user functions
  that call explicit template specializations, skeleton instances in function templates (`tests/codegen/template_instances*.cpp`) and `skepu::multiple`.
* `skepu-lib/filter.hpp` and `util.hpp`: `skepu.filter` and `skepu.util` are generated from them; run `scripts/gen_skepu_filter.py` in SkePY after a change.
* `skepu-lib/blas.hpp`, `ml.hpp`, `randomize.hpp`: bound by wrappers in `shedskin/shedskin/lib/skepu/`.
* `skepu3/impl/tracing.h` and `tracing.hpp`: file and line as arguments for non-C++ frontends, `SKEPU_TRACE_MODULE`, `SKEPU_TRACE_FLUSH()`.
* `CMake/SkePUHelpers.cmake`: `skepu_add_executable` (absolute source paths, `PRECOMPILE_OPTIONS`) and `skepu_add_library`; `CMakeLists.txt` installs the
  `skepu-tool` export files and `nlohmann/json.hpp`.
* `examples/*.cpp`: the references of SkePY's example tests. After a change to an example or to a library it uses, run `shedskin/tests/skepu/reference.py -o`
  for it. `examples/lodepng.*` is copied to `shedskin/shedskin/lib/skepu/`.

## License

SkePY keeps the licenses of Shed Skin where it changes its files: the compiler is GPL-3.0-only, including our changes to it, which is why SkePY is a
separate repository. The new SkePU support (the `skepu` module and its wrappers, the examples and the tests) is under the SkePU license, by Axel
Hammarberg, who made the first implementation, and August Ernstsson. The programs that SkePY generates contain neither compiler nor GPL code. The SkePU
changes that SkePY needs, in this repository, are under SkePU's license.
