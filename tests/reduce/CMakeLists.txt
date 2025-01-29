# ------------------------------------------------
#   Reduce fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(reduce)
skepu_add_test_case_cpu(reduce)
skepu_add_test_case_openmp(reduce)
skepu_add_test_case_opencl(reduce)
skepu_add_test_case_cuda(reduce)
skepu_add_test_case_hybrid_opencl(reduce)
skepu_add_test_case_hybrid_cuda(reduce)

# ------------------------------------------------
#   Reduce start value
# ------------------------------------------------

skepu_add_test_case_sequential(startval)
skepu_add_test_case_cpu(startval)
skepu_add_test_case_openmp(startval)
skepu_add_test_case_opencl(startval)
skepu_add_test_case_cuda(startval)
skepu_add_test_case_hybrid_opencl(startval)
skepu_add_test_case_hybrid_cuda(startval)

# ------------------------------------------------
#   Reduce with 'or' user function
# ------------------------------------------------

skepu_add_test_case_sequential(orreduce)
skepu_add_test_case_cpu(orreduce)
skepu_add_test_case_openmp(orreduce)
skepu_add_test_case_opencl(orreduce)
skepu_add_test_case_cuda(orreduce)
skepu_add_test_case_hybrid_opencl(orreduce)
skepu_add_test_case_hybrid_cuda(orreduce)
