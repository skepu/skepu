# ------------------------------------------------
#   MapPairsReduce fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mappairsreduce)
skepu_add_test_case_cpu(mappairsreduce)
skepu_add_test_case_openmp(mappairsreduce)
skepu_add_test_case_opencl(mappairsreduce)

# It's unclear why this does not work for CUDA.
# Deeper inspection required to determine cause.
#skepu_add_test_case_cuda(mappairsreduce)
skepu_add_test_case_hybrid_opencl(mappairsreduce)
#skepu_add_test_case_hybrid_cuda(mappairsreduce)

# ------------------------------------------------
#   MapPairsReduce with variadic return
# ------------------------------------------------

skepu_add_test_case_sequential(mpr_multi_return)
skepu_add_test_case_cpu(mpr_multi_return)
skepu_add_test_case_openmp(mpr_multi_return)

# This was fixed to run, i.e. it doesn't throw and error at runtime.
# However, the result is incorrect so this test will be disabled until someone fixes the implementation.
#skepu_add_test_case_opencl(mpr_multi_return)

# The OpenMP part of this works, while the OpenCL part has the same errors as above.
#skepu_add_test_case_hybrid_opencl(mpr_multi_return)


# These do not compile for the same reason as OpenCL: the implementation has not been properly configured for multi return.
# For example, the single return output container (skepu_output) is used for multi return even though they likely have their own
# output containers (likely skepu_output_0, skepu_output_1). Seeing as fixing this didn't lead to the OpenCL version working properly,
# it's unlikely to work here either. Especially since the CUDA version doesn't even work for single return.
#skepu_add_test_case_cuda(mpr_multi_return)
#skepu_add_test_case_hybrid_cuda(mpr_multi_return)
