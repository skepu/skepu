
# ------------------------------------------------
#   Container proxy access, flat map, and the life cycle of the containers
# ------------------------------------------------

skepu_add_test_case_sequential(containers)
skepu_add_test_case_cpu(containers)


# ------------------------------------------------
#   Swapping
# ------------------------------------------------

skepu_add_test_case_sequential(swap)
skepu_add_test_case_cpu(swap)
skepu_add_test_case_openmp(swap)
skepu_add_test_case_opencl(swap)
skepu_add_test_case_opencl_multi(swap)
skepu_add_test_case_cuda(swap)
skepu_add_test_case_hybrid_opencl(swap)
skepu_add_test_case_hybrid_cuda(swap)

# ------------------------------------------------
#   skepu::external flushes the containers it reads
# ------------------------------------------------

skepu_add_test_case_sequential(external)
skepu_add_test_case_cpu(external)
