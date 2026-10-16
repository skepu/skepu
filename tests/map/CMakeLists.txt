
# ------------------------------------------------
#   Map fundamentals
# ------------------------------------------------

skepu_add_test(map ALSO opencl_multi)

# ------------------------------------------------
#   Map with index parameter
# ------------------------------------------------

# map_variants: map_in_place, map_index, map_tuple, mvmult (one program)
skepu_add_test(map_variants)


# ------------------------------------------------
#   Map with stride
# ------------------------------------------------

# With these sizes the if (gpuSize < 32) passes, so only OpenMP executes in the hybrid backend: the test passes, but it does not
# exercise the GPU part of the hybrid execution.
skepu_add_test(map_stride EXCLUDE hybrid_cuda)

# ------------------------------------------------
#   Index of a strided Map
# ------------------------------------------------

skepu_add_test(map_stride_index EXCLUDE cuda hybrid_opencl hybrid_cuda)
