# ------------------------------------------------
#   MapPairsReduce fundamentals
# ------------------------------------------------

# mappairsreduce: mappairsreduce, mpr_multi_return (one program)
skepu_add_test_case_sequential(mappairsreduce)
skepu_add_test_case_cpu(mappairsreduce)
skepu_add_test_case_openmp(mappairsreduce)
skepu_add_test_case_opencl(mappairsreduce)
#skepu_add_test_case_cuda(mappairsreduce)
skepu_add_test_case_hybrid_opencl(mappairsreduce)
#skepu_add_test_case_hybrid_cuda(mappairsreduce)


# It's unclear why this does not work for CUDA.
# Deeper inspection required to determine cause.

# ------------------------------------------------
#   MapPairsReduce with variadic return
# ------------------------------------------------


# The hybrid backend passes. (With these sizes the work may be done by OpenMP alone, as in the Map with stride test.)


# These do not compile for the same reason as OpenCL: the implementation has not been properly configured for multi return.
# For example, the single return output container (skepu_output) is used for multi return even though they likely have their own
# output containers (likely skepu_output_0, skepu_output_1). Seeing as fixing this didn't lead to the OpenCL version working properly,
# it's unlikely to work here either. Especially since the CUDA version doesn't even work for single return.
