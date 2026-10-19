include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ------------------------------------------------
#   BLAS library
# ------------------------------------------------

# The precompiled build of this test does not compile (blas.hpp: "expected nested-name-specifier before 'complex'"). The tests below
# (blas_checked, blas_complex) are the precompiled tests of the library.
skepu_add_test(blas BACKENDS sequential)

# ------------------------------------------------
#   Complex fundamentals
# ------------------------------------------------

skepu_add_test(complex)

# ------------------------------------------------
#   Random fundamentals, using namespace and the random fills of skepu-lib/randomize.hpp
# ------------------------------------------------

# CUDA: three cases fail: MapOverlap 1D in mirror mode (not implemented for CUDA, with a large vector too), and MapPairs + MapPairsReduce
# (wrong results, see mappairsreduce). The rest passes.
skepu_add_test(random EXCLUDE cuda hybrid_cuda)


# ------------------------------------------------
#   Argument parser (skepu-lib/args.hpp)
# ------------------------------------------------

skepu_add_test(args EXCLUDE cuda hybrid_opencl hybrid_cuda)


# ------------------------------------------------
#   skepu_external
# ------------------------------------------------

skepu_add_test(skepu_external)

# ------------------------------------------------
#   DNN library (skepu-lib/dnn)
#   Gradient checks, Adam, predict/evaluate and a synthetic training run (no data files needed)
# ------------------------------------------------

# With the precompiler: the same tests on the other backends. The backend is selected with the
# environment variable SKEPU_TEST_BACKEND (the test program runs on the sequential backend otherwise).
skepu_add_test(dnn EXCLUDE cuda hybrid_opencl hybrid_cuda)
if(TEST dnn_cpu)
	set_tests_properties(dnn_cpu PROPERTIES ENVIRONMENT "SKEPU_TEST_BACKEND=CPU")
endif()
if(TEST dnn_openmp)
	set_tests_properties(dnn_openmp PROPERTIES ENVIRONMENT "SKEPU_TEST_BACKEND=OpenMP")
endif()
if(TEST dnn_opencl)
	set_tests_properties(dnn_opencl PROPERTIES ENVIRONMENT "SKEPU_TEST_BACKEND=OpenCL")
endif()

# Regression test on the logs of the MNIST / CIFAR-10 / signal examples. Cases whose data sets
# are missing from examples/data are skipped; see dnn_regression/README.md.
add_test(NAME dnn_regression
	COMMAND ${CMAKE_CURRENT_LIST_DIR}/dnn_regression/run.sh --cxx ${CMAKE_CXX_COMPILER} --tol 1e-2)

# ------------------------------------------------
#   BLAS library, checked float routines (blas_float, blas_level1, blas_level23: one program)
# ------------------------------------------------

skepu_add_test(blas_checked EXCLUDE cuda hybrid_opencl hybrid_cuda)


# ------------------------------------------------
#   BLAS library, checked complex and mixed precision routines (blas_complex, blas_mixed_precision: one program)
#   (not on OpenCL: the kernels of the complex numbers are not complete there)
# ------------------------------------------------

skepu_add_test(blas_complex BACKENDS sequential cpu openmp)


# ------------------------------------------------
#   Image filter library (skepu-lib/filter): user types in a namespace
# ------------------------------------------------

skepu_add_test(filter EXCLUDE sequential)
