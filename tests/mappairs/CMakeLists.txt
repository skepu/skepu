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

skepu_add_test(mappairs_in_place EXCLUDE hybrid_opencl hybrid_cuda)
