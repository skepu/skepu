include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ------------------------------------------------
#   BLAS library
# ------------------------------------------------

skepu_add_test_case_sequential(blas)
#skepu_add_test_case_cpu(blas)
#skepu_add_test_case_openmp(blas)
##skepu_add_test_case_opencl(blas)
#skepu_add_test_case_cuda(blas)
##skepu_add_test_case_hybrid_opencl(blas)
#skepu_add_test_case_hybrid_cuda(blas)

# ------------------------------------------------
#   Complex fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(complex)
skepu_add_test_case_cpu(complex)
skepu_add_test_case_openmp(complex)
#skepu_add_test_case_opencl(complex)
skepu_add_test_case_cuda(complex)
#skepu_add_test_case_hybrid_opencl(complex)
skepu_add_test_case_hybrid_cuda(complex)

# ------------------------------------------------
#   Random fundamentals
# ------------------------------------------------

#skepu_add_test_case_sequential(random)
#skepu_add_test_case_cpu(random)
#skepu_add_test_case_openmp(random)
#skepu_add_test_case_opencl(random)
#skepu_add_test_case_cuda(random)
#skepu_add_test_case_hybrid_opencl(random)
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
#   Standard library namespaces
# ------------------------------------------------

# TEST-ERROR: precompiler segfaults

# skepu_add_test_case_sequential(using_namespace)
# skepu_add_test_case_cpu(using_namespace)
# skepu_add_test_case_openmp(using_namespace)
# skepu_add_test_case_opencl(using_namespace)
# skepu_add_test_case_cuda(using_namespace)
# skepu_add_test_case_hybrid_opencl(using_namespace)
# skepu_add_test_case_hybrid_cuda(using_namespace)

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
#   BLAS library, checked float routines
# ------------------------------------------------

skepu_add_test_case_sequential(blas_float)
skepu_add_test_case_cpu(blas_float)
skepu_add_test_case_openmp(blas_float)
skepu_add_test_case_opencl(blas_float)

# ------------------------------------------------
#   BLAS library, checked level 1 routines
# ------------------------------------------------

skepu_add_test_case_sequential(blas_level1)
skepu_add_test_case_cpu(blas_level1)
skepu_add_test_case_openmp(blas_level1)
skepu_add_test_case_opencl(blas_level1)

# ------------------------------------------------
#   BLAS library, checked level 2 and 3 routines
# ------------------------------------------------

skepu_add_test_case_sequential(blas_level23)
skepu_add_test_case_cpu(blas_level23)
skepu_add_test_case_openmp(blas_level23)
skepu_add_test_case_opencl(blas_level23)

# ------------------------------------------------
#   BLAS library, checked complex routines (OpenCL: complex numbers are not tested there)
# ------------------------------------------------

skepu_add_test_case_sequential(blas_complex)
skepu_add_test_case_cpu(blas_complex)
skepu_add_test_case_openmp(blas_complex)

# ------------------------------------------------
#   BLAS library, float, double and mixed operands in one program (OpenCL: double precision)
# ------------------------------------------------

skepu_add_test_case_sequential(blas_mixed_precision)
skepu_add_test_case_cpu(blas_mixed_precision)
skepu_add_test_case_openmp(blas_mixed_precision)

# ------------------------------------------------
#   Image filter library (skepu-lib/filter): user types in a namespace
# ------------------------------------------------

skepu_add_test_case_cpu(filter)
skepu_add_test_case_openmp(filter)
skepu_add_test_case_opencl(filter)
skepu_add_test_case_cuda(filter)
skepu_add_test_case_hybrid_opencl(filter)
skepu_add_test_case_hybrid_cuda(filter)
