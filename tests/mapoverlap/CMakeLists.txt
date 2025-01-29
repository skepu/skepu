# ------------------------------------------------
#   MapOverlap 1D fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_1d)
skepu_add_test_case_cpu(mapoverlap_1d)
skepu_add_test_case_openmp(mapoverlap_1d)
skepu_add_test_case_opencl(mapoverlap_1d)
skepu_add_test_case_cuda(mapoverlap_1d)
skepu_add_test_case_hybrid_opencl(mapoverlap_1d)
skepu_add_test_case_hybrid_cuda(mapoverlap_1d)


# ------------------------------------------------
#   MapOverlap 2D fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_2d)
skepu_add_test_case_cpu(mapoverlap_2d)
skepu_add_test_case_openmp(mapoverlap_2d)
skepu_add_test_case_opencl(mapoverlap_2d)
skepu_add_test_case_cuda(mapoverlap_2d)
skepu_add_test_case_hybrid_opencl(mapoverlap_2d)
skepu_add_test_case_hybrid_cuda(mapoverlap_2d)


# ------------------------------------------------
#   MapOverlap 3D fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_3d)
skepu_add_test_case_cpu(mapoverlap_3d)
skepu_add_test_case_openmp(mapoverlap_3d)
skepu_add_test_case_opencl(mapoverlap_3d)
skepu_add_test_case_cuda(mapoverlap_3d)
skepu_add_test_case_hybrid_opencl(mapoverlap_3d)
skepu_add_test_case_hybrid_cuda(mapoverlap_3d)


# ------------------------------------------------
#   MapOverlap 4D fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_4d)
skepu_add_test_case_cpu(mapoverlap_4d)
skepu_add_test_case_openmp(mapoverlap_4d)
#skepu_add_test_case_opencl(mapoverlap_4d)
#skepu_add_test_case_cuda(mapoverlap_4d)
#skepu_add_test_case_hybrid_opencl(mapoverlap_4d)
#skepu_add_test_case_hybrid_cuda(mapoverlap_4d)


# ------------------------------------------------
#   MapOverlap in place
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_in_place)
skepu_add_test_case_cpu(mapoverlap_in_place)
skepu_add_test_case_openmp(mapoverlap_in_place)
skepu_add_test_case_opencl(mapoverlap_in_place)
skepu_add_test_case_cuda(mapoverlap_in_place)
skepu_add_test_case_hybrid_opencl(mapoverlap_in_place)
skepu_add_test_case_hybrid_cuda(mapoverlap_in_place)

# ------------------------------------------------
#   MapOverlap with variadic return
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_tuple)
skepu_add_test_case_cpu(mapoverlap_tuple)
skepu_add_test_case_openmp(mapoverlap_tuple)
skepu_add_test_case_opencl(mapoverlap_tuple)
skepu_add_test_case_cuda(mapoverlap_tuple)
skepu_add_test_case_hybrid_opencl(mapoverlap_tuple)
skepu_add_test_case_hybrid_cuda(mapoverlap_tuple)

# ------------------------------------------------
#   MapOverlap update modes
# ------------------------------------------------

skepu_add_test_case_sequential(updatemode)
skepu_add_test_case_cpu(updatemode)
skepu_add_test_case_openmp(updatemode)
skepu_add_test_case_opencl(updatemode)
skepu_add_test_case_cuda(updatemode)
skepu_add_test_case_hybrid_opencl(updatemode)
skepu_add_test_case_hybrid_cuda(updatemode)
