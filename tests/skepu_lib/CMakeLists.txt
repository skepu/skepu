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

skepu_add_test_case_sequential(random)
skepu_add_test_case_cpu(random)
skepu_add_test_case_openmp(random)
skepu_add_test_case_opencl(random)
skepu_add_test_case_cuda(random)
skepu_add_test_case_hybrid_opencl(random)
skepu_add_test_case_hybrid_cuda(random)

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
