include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ------------------------------------------------
#   BLAS library
# ------------------------------------------------

skepu_add_test_case_sequential(blas)
# The precompiled build of this test does not compile (blas.hpp: "expected nested-name-specifier before 'complex'"). The tests below
# (blas_checked, blas_complex) are the precompiled tests of the library.
#skepu_add_test_case_cuda(blas)
#skepu_add_test_case_hybrid_cuda(blas)

# ------------------------------------------------
#   Complex fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(complex)
skepu_add_test_case_cpu(complex)
skepu_add_test_case_openmp(complex)
skepu_add_test_case_opencl(complex)
skepu_add_test_case_cuda(complex)
skepu_add_test_case_hybrid_opencl(complex)
skepu_add_test_case_hybrid_cuda(complex)

# ------------------------------------------------
#   Random fundamentals, using namespace and the random fills of skepu-lib/randomize.hpp
#   (random, using_namespace, randomize: one program, as the build of a test program has a cost of its own; each part is in a namespace of
#   its own, and its test cases have the name of the file it came from as a prefix)
# ------------------------------------------------

skepu_add_test_case_sequential(random)
skepu_add_test_case_cpu(random)
skepu_add_test_case_openmp(random)
skepu_add_test_case_opencl(random)
#skepu_add_test_case_cuda(random)
skepu_add_test_case_hybrid_opencl(random)
#skepu_add_test_case_hybrid_cuda(random)


# ------------------------------------------------
#   skepu_external
# ------------------------------------------------

skepu_add_test_case_sequential(skepu_external)
skepu_add_test_case_cpu(skepu_external)
skepu_add_test_case_openmp(skepu_external)
skepu_add_test_case_opencl(skepu_external)
skepu_add_test_case_cuda(skepu_external)
skepu_add_test_case_hybrid_opencl(skepu_external)
skepu_add_test_case_hybrid_cuda(skepu_external)

# ------------------------------------------------
#   DNN library (skepu-lib/dnn)
#   Gradient checks, Adam, predict/evaluate and a synthetic training run (no data files needed)
# ------------------------------------------------

skepu_add_test_case_sequential(dnn)
# With the precompiler: the same tests on the other backends. The backend is selected with the
# environment variable SKEPU_TEST_BACKEND (the test program runs on the sequential backend otherwise).
skepu_add_test_case_cpu(dnn)
skepu_add_test_case_openmp(dnn)
skepu_add_test_case_opencl(dnn)
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

skepu_add_test_case_sequential(blas_checked)
skepu_add_test_case_cpu(blas_checked)
skepu_add_test_case_openmp(blas_checked)
skepu_add_test_case_opencl(blas_checked)


# ------------------------------------------------
#   BLAS library, checked complex and mixed precision routines (blas_complex, blas_mixed_precision: one program)
#   (not on OpenCL: the kernels of the complex numbers are not complete there)
# ------------------------------------------------

skepu_add_test_case_sequential(blas_complex)
skepu_add_test_case_cpu(blas_complex)
skepu_add_test_case_openmp(blas_complex)


# ------------------------------------------------
#   Image filter library (skepu-lib/filter): user types in a namespace
# ------------------------------------------------

skepu_add_test_case_cpu(filter)
skepu_add_test_case_openmp(filter)
skepu_add_test_case_opencl(filter)
skepu_add_test_case_cuda(filter)
skepu_add_test_case_hybrid_opencl(filter)
skepu_add_test_case_hybrid_cuda(filter)
