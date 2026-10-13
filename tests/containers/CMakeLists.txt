
# ------------------------------------------------
#   Container proxy access, flat map, and the life cycle of the containers (containers, flatmap, lifecycle: one program, as the build of a
#   test program has a cost of its own; each part is in a namespace of its own, and its test cases have the name of the file it came from as
#   a prefix)
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
