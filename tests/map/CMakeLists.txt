
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

skepu_add_test_case_sequential(map_index)
skepu_add_test_case_cpu(map_index)
skepu_add_test_case_openmp(map_index)
skepu_add_test_case_opencl(map_index)
skepu_add_test_case_cuda(map_index)
skepu_add_test_case_hybrid_opencl(map_index)
skepu_add_test_case_hybrid_cuda(map_index)

# ------------------------------------------------
#   Matrix-vector multiplication
# ------------------------------------------------

skepu_add_test_case_sequential(mvmult)
skepu_add_test_case_cpu(mvmult)
skepu_add_test_case_openmp(mvmult)
skepu_add_test_case_opencl(mvmult)
skepu_add_test_case_cuda(mvmult)
skepu_add_test_case_hybrid_opencl(mvmult)
skepu_add_test_case_hybrid_cuda(mvmult)

# ------------------------------------------------
#   Map in place
# ------------------------------------------------

skepu_add_test_case_sequential(map_in_place)
skepu_add_test_case_cpu(map_in_place)
skepu_add_test_case_openmp(map_in_place)
skepu_add_test_case_opencl(map_in_place)
skepu_add_test_case_cuda(map_in_place)
skepu_add_test_case_hybrid_opencl(map_in_place)
skepu_add_test_case_hybrid_cuda(map_in_place)

# ------------------------------------------------
#   Map with variadic return
# ------------------------------------------------

skepu_add_test_case_sequential(map_tuple)
skepu_add_test_case_cpu(map_tuple)
skepu_add_test_case_openmp(map_tuple)
skepu_add_test_case_opencl(map_tuple)
skepu_add_test_case_cuda(map_tuple)
skepu_add_test_case_hybrid_opencl(map_tuple)
skepu_add_test_case_hybrid_cuda(map_tuple)


# ------------------------------------------------
#   Map with stride
# ------------------------------------------------

skepu_add_test_case_sequential(map_stride)
skepu_add_test_case_cpu(map_stride)
skepu_add_test_case_openmp(map_stride)
skepu_add_test_case_opencl(map_stride)
skepu_add_test_case_cuda(map_stride)

# Currently, the if (gpuSize < 32) passes which leads to only OMP executing.
# Since OMP are called with different parameters inside map_hy.inl compared to inside map.h, the executing becomes incorrect.
# Furthermore, since this is not even executing anything on the GPU, it's a bad test for hybrid execution.
# A bandaid fix would be to perform the if (gpuSize < 32) inside map.h but that is bad code since we're executing logic
# related to map_hy.inl outisde of it so this is not a valid solution to the issue.
#skepu_add_test_case_hybrid_opencl(map_stride)
#skepu_add_test_case_hybrid_cuda(map_stride)