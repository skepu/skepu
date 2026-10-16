# ------------------------------------------------
#   MapPairs fundamentals
# ------------------------------------------------

skepu_add_test(mappairs EXCLUDE hybrid_cuda)


# ------------------------------------------------
#   MapPairs with variadic return
# ------------------------------------------------

skepu_add_test(mappairs_tuple)

# ------------------------------------------------
#   MapPairs in place mode
# ------------------------------------------------

# CUDA: MapPairs in place mode is not implemented (mappairs_cu.inl stops with an error).
skepu_add_test(mappairs_in_place EXCLUDE cuda hybrid_opencl hybrid_cuda)
