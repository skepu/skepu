
# ------------------------------------------------
#   Container proxy access
# ------------------------------------------------

skepu_add_test_case_sequential(lifecycle)
skepu_add_test_case_cpu(lifecycle)

# ------------------------------------------------
#   Container proxy access
# ------------------------------------------------

skepu_add_test_case_sequential(containers)
skepu_add_test_case_cpu(containers)

# ------------------------------------------------
#   Flat map
# ------------------------------------------------

skepu_add_test_case_sequential(flatmap)
skepu_add_test_case_cpu(flatmap)

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
