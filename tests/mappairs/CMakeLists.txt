# ------------------------------------------------
#   MapPairs fundamentals
# ------------------------------------------------

# Unclear why this works in OpenCL but not CUDA. The calls look similar in both.
# The problem is that only the CPU part of the call gets produced in the output container,
# the rest of the data is either 0 or rubbish.
skepu_add_test(mappairs EXCLUDE hybrid_cuda)


# ------------------------------------------------
#   MapPairs with variadic return
# ------------------------------------------------

skepu_add_test(mappairs_tuple)

# ------------------------------------------------
#   MapPairs in place mode
# ------------------------------------------------

skepu_add_test(mappairs_in_place EXCLUDE cuda hybrid_opencl hybrid_cuda)
