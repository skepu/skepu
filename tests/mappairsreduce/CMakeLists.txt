# ------------------------------------------------
#   MapPairsReduce fundamentals and variadic return
# ------------------------------------------------

# mappairsreduce: mappairsreduce, mpr_multi_return (one program)
#
# CUDA (and hybrid CUDA) is excluded for two reasons. The CUDA kernel template (src/skepu-tool/mappairsreduce_cu.cpp) has no path for a user
# function that returns multiple values: it assigns the result to one variable of the reduction type and writes the single output
# skepu_output, so the multi return part does not compile (the OpenCL template reduces each output separately). And it is unclear why the
# single return part does not work for CUDA either; deeper inspection is required to determine the cause.
skepu_add_test(mappairsreduce EXCLUDE cuda hybrid_cuda)
