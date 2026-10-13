
# ------------------------------------------------
#   Map fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(map)
skepu_add_test_case_cpu(map)
skepu_add_test_case_openmp(map)
skepu_add_test_case_opencl(map)
skepu_add_test_case_opencl_multi(map)
skepu_add_test_case_cuda(map)
skepu_add_test_case_hybrid_opencl(map)
skepu_add_test_case_hybrid_cuda(map)

# ------------------------------------------------
#   Map with index parameter
# ------------------------------------------------

# map_variants: map_in_place, map_index, map_tuple, mvmult (one program)
skepu_add_test_case_sequential(map_variants)
skepu_add_test_case_cpu(map_variants)
skepu_add_test_case_openmp(map_variants)
skepu_add_test_case_opencl(map_variants)
skepu_add_test_case_cuda(map_variants)
skepu_add_test_case_hybrid_opencl(map_variants)
skepu_add_test_case_hybrid_cuda(map_variants)


# ------------------------------------------------
#   Map with stride
# ------------------------------------------------

skepu_add_test_case_sequential(map_stride)
skepu_add_test_case_cpu(map_stride)
skepu_add_test_case_openmp(map_stride)
skepu_add_test_case_opencl(map_stride)
skepu_add_test_case_cuda(map_stride)

# With these sizes the if (gpuSize < 32) passes, so only OpenMP executes in the hybrid backend: the test passes, but it does not
# exercise the GPU part of the hybrid execution.
skepu_add_test_case_hybrid_opencl(map_stride)
#skepu_add_test_case_hybrid_cuda(map_stride)
# ------------------------------------------------
#   Index of a strided Map
# ------------------------------------------------

skepu_add_test_case_sequential(map_stride_index)
skepu_add_test_case_cpu(map_stride_index)
skepu_add_test_case_openmp(map_stride_index)
skepu_add_test_case_opencl(map_stride_index)
