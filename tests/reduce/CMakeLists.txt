# ------------------------------------------------
#   Reduce fundamentals
# ------------------------------------------------

# reduce: orreduce, reduce, startval (one program)
skepu_add_test_case_sequential(reduce)
skepu_add_test_case_cpu(reduce)
skepu_add_test_case_openmp(reduce)
skepu_add_test_case_opencl(reduce)
skepu_add_test_case_cuda(reduce)
skepu_add_test_case_hybrid_opencl(reduce)
skepu_add_test_case_hybrid_cuda(reduce)


