# ------------------------------------------------
#   MapPairsReduce fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mappairsreduce)
skepu_add_test_case_cpu(mappairsreduce)
skepu_add_test_case_openmp(mappairsreduce)
skepu_add_test_case_opencl(mappairsreduce)
#skepu_add_test_case_cuda(mappairsreduce)
skepu_add_test_case_hybrid_opencl(mappairsreduce)
#skepu_add_test_case_hybrid_cuda(mappairsreduce)

# ------------------------------------------------
#   MapPairsReduce with variadic return
# ------------------------------------------------

skepu_add_test_case_sequential(mpr_multi_return)
skepu_add_test_case_cpu(mpr_multi_return)
skepu_add_test_case_openmp(mpr_multi_return)
#skepu_add_test_case_opencl(mpr_multi_return)
#skepu_add_test_case_cuda(mpr_multi_return)
#skepu_add_test_case_hybrid_opencl(mpr_multi_return)
#skepu_add_test_case_hybrid_cuda(mpr_multi_return)
