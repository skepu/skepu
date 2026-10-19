# ------------------------------------------------
#   MapPairsReduce fundamentals and variadic return
# ------------------------------------------------

# mappairsreduce: mappairsreduce, mpr_multi_return (one program)
#
# Hybrid CUDA is excluded: it is not set up for MapPairsReduce.
skepu_add_test(mappairsreduce EXCLUDE hybrid_cuda)
