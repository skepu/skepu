# SkePU-DNN regression test

Short, fixed-seed training runs of the DNN examples (`examples/dnn/dnn_mnist10.cpp`, `dnn_signals1d.cpp`,
`dnn_cifar10.cpp`, `dnn.cpp`) on the sequential (no precompiler) path. The logs of the runs (per-batch and
per-epoch loss and accuracy, test result, confusion matrix) are compared with the golden logs in `golden/`.
It catches changes in the numerical behaviour of the library: layers, losses, optimizers, batching.

```
tests/skepu_lib/dnn_regression/run.sh                 # build, run, compare
tests/skepu_lib/dnn_regression/run.sh --strict        # require identical logs
tests/skepu_lib/dnn_regression/run.sh --update        # accept the current output as the new golden logs
```

Options: `--tol REL` relative tolerance for numbers (default `1e-3`), `--cxx COMPILER` (default `$CXX` or
`clang++`), `--build-dir DIR` (default: a temporary directory, which also holds the logs of failed cases).

Cases:

| case | example | needs |
|---|---|---|
| `mnist_mlp_sgd`, `mnist_cnn_sgd`, `mnist_conv1d_adam` | `dnn_mnist10` | `examples/data/mnist` |
| `signals_conv1d_adam` | `dnn_signals1d` | nothing (synthetic data) |
| `cifar_small_adam` | `dnn_cifar10` | `examples/data/cifar-10-batches-bin` |
| `binary_mnist_sgd` | `dnn` | `examples/data/mnist` |

Cases whose data set is missing are skipped. The data sets are not part of the repository.

## About the golden logs

* They were produced with Apple clang, `-std=c++20 -O2`, on arm64. With the same toolchain the logs are
  bit-identical (`--strict`). Other compilers or flags change floating-point rounding, which grows slowly
  during training, so numbers are compared with a tolerance by default.
* Update them (`--update`) only for an intended change of the numerical behaviour, and look at the diff:
  `git diff tests/skepu_lib/dnn_regression/golden`.
* Backends other than the sequential one are not covered. Once the precompiler is set up, the same runs
  with a different backend name (instead of `cpu`) are a natural way to compare backends against these logs.

The data-free unit tests (also model file save/load round trips) (gradient checks, Adam, predict/evaluate, a synthetic training run) are in
`../dnn.cpp`, a Catch2 test registered in `../CMakeLists.txt`.
