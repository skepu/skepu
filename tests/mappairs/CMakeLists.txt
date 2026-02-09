# ------------------------------------------------
#   MapPairs fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mappairs)
skepu_add_test_case_cpu(mappairs)
skepu_add_test_case_openmp(mappairs)
skepu_add_test_case_opencl(mappairs)
skepu_add_test_case_cuda(mappairs)
skepu_add_test_case_hybrid_opencl(mappairs)

# Unclear why this works in OpenCL but not CUDA. The calls look similar in both.
# The problem is that only the CPU part of the call gets produced in the output container,
# the rest of the data is either 0 or rubbish.
#skepu_add_test_case_hybrid_cuda(mappairs)

# ------------------------------------------------
#   MapPairs with variadic return
# ------------------------------------------------

skepu_add_test_case_sequential(mappairs_tuple)
skepu_add_test_case_cpu(mappairs_tuple)
skepu_add_test_case_openmp(mappairs_tuple)
skepu_add_test_case_opencl(mappairs_tuple)
skepu_add_test_case_cuda(mappairs_tuple)
skepu_add_test_case_hybrid_opencl(mappairs_tuple)
skepu_add_test_case_hybrid_cuda(mappairs_tuple)
