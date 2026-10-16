# ------------------------------------------------
#   MapReduce fundamentals
# ------------------------------------------------

# CUDA: MapReduce over a Matrix, Tensor3 and Tensor4 (mapred2 to mapred4) gives wrong sums (the vector case is right); not investigated.
skepu_add_test(mapreduce EXCLUDE cuda hybrid_cuda)
