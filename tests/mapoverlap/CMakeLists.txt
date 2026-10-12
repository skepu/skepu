# ------------------------------------------------
#   MapOverlap default values
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_default_values)
skepu_add_test_case_cpu(mapoverlap_default_values)
skepu_add_test_case_openmp(mapoverlap_default_values)
skepu_add_test_case_opencl(mapoverlap_default_values)
skepu_add_test_case_cuda(mapoverlap_default_values)
skepu_add_test_case_hybrid_opencl(mapoverlap_default_values)
skepu_add_test_case_hybrid_cuda(mapoverlap_default_values)


# ------------------------------------------------
#   MapOverlap 1D fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_1d)
skepu_add_test_case_cpu(mapoverlap_1d)
skepu_add_test_case_openmp(mapoverlap_1d)
skepu_add_test_case_opencl(mapoverlap_1d)
skepu_add_test_case_cuda(mapoverlap_1d)
skepu_add_test_case_hybrid_opencl(mapoverlap_1d)
#skepu_add_test_case_hybrid_cuda(mapoverlap_1d)


# ------------------------------------------------
#   MapOverlap 1D rowcol fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_1d_rowcol)
skepu_add_test_case_cpu(mapoverlap_1d_rowcol)
skepu_add_test_case_openmp(mapoverlap_1d_rowcol)
skepu_add_test_case_opencl(mapoverlap_1d_rowcol)
skepu_add_test_case_cuda(mapoverlap_1d_rowcol)
skepu_add_test_case_hybrid_opencl(mapoverlap_1d_rowcol)
#skepu_add_test_case_hybrid_cuda(mapoverlap_1d_rowcol)

# ------------------------------------------------
#   MapOverlap 1D on matrices with rows and columns longer than a thread block
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_1d_wide)
skepu_add_test_case_cpu(mapoverlap_1d_wide)
skepu_add_test_case_openmp(mapoverlap_1d_wide)
skepu_add_test_case_opencl(mapoverlap_1d_wide)
skepu_add_test_case_cuda(mapoverlap_1d_wide)
# hybrid OpenCL: fails, a Matrix column access is out of bounds
#skepu_add_test_case_hybrid_opencl(mapoverlap_1d_wide)
#skepu_add_test_case_hybrid_cuda(mapoverlap_1d_wide)

# ------------------------------------------------
#   MapOverlap 2D mirror mode (edge mode None, input of the size of the output)
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_2d_mirror)
skepu_add_test_case_cpu(mapoverlap_2d_mirror)
skepu_add_test_case_openmp(mapoverlap_2d_mirror)
skepu_add_test_case_opencl(mapoverlap_2d_mirror)
# (not implemented for the CUDA and hybrid backends: the call stops with an error)
#skepu_add_test_case_cuda(mapoverlap_2d_mirror)
# hybrid OpenCL: MapOverlap2D is not implemented for the hybrid backend
#skepu_add_test_case_hybrid_opencl(mapoverlap_2d_mirror)
#skepu_add_test_case_hybrid_cuda(mapoverlap_2d_mirror)


# ------------------------------------------------
#   MapOverlap 1D, 3D and 4D mirror mode (edge mode None, input of the size of the output)
# ------------------------------------------------

# (not implemented for the CUDA and hybrid backends: the call stops with an error)
skepu_add_test_case_sequential(mapoverlap_1d_mirror)
skepu_add_test_case_cpu(mapoverlap_1d_mirror)
skepu_add_test_case_openmp(mapoverlap_1d_mirror)
skepu_add_test_case_opencl(mapoverlap_1d_mirror)

skepu_add_test_case_sequential(mapoverlap_3d_mirror)
skepu_add_test_case_cpu(mapoverlap_3d_mirror)
skepu_add_test_case_openmp(mapoverlap_3d_mirror)
skepu_add_test_case_opencl(mapoverlap_3d_mirror)

skepu_add_test_case_sequential(mapoverlap_4d_mirror)
skepu_add_test_case_cpu(mapoverlap_4d_mirror)
skepu_add_test_case_openmp(mapoverlap_4d_mirror)
skepu_add_test_case_opencl(mapoverlap_4d_mirror)


# ------------------------------------------------
#   MapOverlap 2D fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_2d)
skepu_add_test_case_cpu(mapoverlap_2d)
skepu_add_test_case_openmp(mapoverlap_2d)
skepu_add_test_case_opencl(mapoverlap_2d)
skepu_add_test_case_cuda(mapoverlap_2d)
skepu_add_test_case_hybrid_opencl(mapoverlap_2d)
#skepu_add_test_case_hybrid_cuda(mapoverlap_2d)



# ------------------------------------------------
#   MapOverlap expanding
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_2d_expanding)
skepu_add_test_case_cpu(mapoverlap_2d_expanding)
skepu_add_test_case_openmp(mapoverlap_2d_expanding)
skepu_add_test_case_opencl(mapoverlap_2d_expanding)
skepu_add_test_case_cuda(mapoverlap_2d_expanding)
skepu_add_test_case_hybrid_opencl(mapoverlap_2d_expanding)
#skepu_add_test_case_hybrid_cuda(mapoverlap_2d_expanding)


# ------------------------------------------------
#   MapOverlap 2D fundamentals new
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_2d_new)
skepu_add_test_case_cpu(mapoverlap_2d_new)
skepu_add_test_case_openmp(mapoverlap_2d_new)
skepu_add_test_case_opencl(mapoverlap_2d_new)
skepu_add_test_case_cuda(mapoverlap_2d_new)
skepu_add_test_case_hybrid_opencl(mapoverlap_2d_new)
#skepu_add_test_case_hybrid_cuda(mapoverlap_2d_new)


# ------------------------------------------------
#   MapPool 2D fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mappool_2d)
skepu_add_test_case_cpu(mappool_2d)
skepu_add_test_case_openmp(mappool_2d)
skepu_add_test_case_opencl(mappool_2d)
skepu_add_test_case_cuda(mappool_2d)
skepu_add_test_case_hybrid_opencl(mappool_2d)
#skepu_add_test_case_hybrid_cuda(mappool_2d)


# ------------------------------------------------
#   MapPool 2D small output sizes
# ------------------------------------------------

skepu_add_test_case_sequential(mappool_2d_small_output)
skepu_add_test_case_cpu(mappool_2d_small_output)
skepu_add_test_case_openmp(mappool_2d_small_output)
skepu_add_test_case_opencl(mappool_2d_small_output)
skepu_add_test_case_cuda(mappool_2d_small_output)
skepu_add_test_case_hybrid_opencl(mappool_2d_small_output)
#skepu_add_test_case_hybrid_cuda(mappool_2d_small_output)


# ------------------------------------------------
#   MapOverlap 3D fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_3d)
skepu_add_test_case_cpu(mapoverlap_3d)
skepu_add_test_case_openmp(mapoverlap_3d)
skepu_add_test_case_opencl(mapoverlap_3d)
skepu_add_test_case_cuda(mapoverlap_3d)
skepu_add_test_case_hybrid_opencl(mapoverlap_3d)
#skepu_add_test_case_hybrid_cuda(mapoverlap_3d)


# ------------------------------------------------
#   MapOverlap 3D fundamentals new
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_3d_new)
skepu_add_test_case_cpu(mapoverlap_3d_new)
skepu_add_test_case_openmp(mapoverlap_3d_new)
skepu_add_test_case_opencl(mapoverlap_3d_new)
skepu_add_test_case_cuda(mapoverlap_3d_new)
skepu_add_test_case_hybrid_opencl(mapoverlap_3d_new)
#skepu_add_test_case_hybrid_cuda(mapoverlap_3d_new)


# ------------------------------------------------
#   MapPool 3D fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mappool_3d)
skepu_add_test_case_cpu(mappool_3d)
skepu_add_test_case_openmp(mappool_3d)
skepu_add_test_case_opencl(mappool_3d)
skepu_add_test_case_cuda(mappool_3d)
skepu_add_test_case_hybrid_opencl(mappool_3d)
#skepu_add_test_case_hybrid_cuda(mappool_3d)


# ------------------------------------------------
#   MapPool 3D small output sizes
# ------------------------------------------------

skepu_add_test_case_sequential(mappool_3d_small_output)
skepu_add_test_case_cpu(mappool_3d_small_output)
skepu_add_test_case_openmp(mappool_3d_small_output)
skepu_add_test_case_opencl(mappool_3d_small_output)
skepu_add_test_case_cuda(mappool_3d_small_output)
skepu_add_test_case_hybrid_opencl(mappool_3d_small_output)
#skepu_add_test_case_hybrid_cuda(mappool_3d_small_output)


# ------------------------------------------------
#   MapOverlap 4D fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_4d)
skepu_add_test_case_cpu(mapoverlap_4d)
skepu_add_test_case_openmp(mapoverlap_4d)
skepu_add_test_case_opencl(mapoverlap_4d)
skepu_add_test_case_cuda(mapoverlap_4d)
skepu_add_test_case_hybrid_opencl(mapoverlap_4d)
#skepu_add_test_case_hybrid_cuda(mapoverlap_4d)


# ------------------------------------------------
#   MapOverlap 4D Expanding
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_4d_expanding)
skepu_add_test_case_cpu(mapoverlap_4d_expanding)
skepu_add_test_case_openmp(mapoverlap_4d_expanding)
skepu_add_test_case_opencl(mapoverlap_4d_expanding)
skepu_add_test_case_cuda(mapoverlap_4d_expanding)
skepu_add_test_case_hybrid_opencl(mapoverlap_4d_expanding)
#skepu_add_test_case_hybrid_cuda(mapoverlap_4d_expanding)


# ------------------------------------------------
#   MapOverlap 4D fundamentals new
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_4d_new)
skepu_add_test_case_cpu(mapoverlap_4d_new)
skepu_add_test_case_openmp(mapoverlap_4d_new)
skepu_add_test_case_opencl(mapoverlap_4d_new)
skepu_add_test_case_cuda(mapoverlap_4d_new)
skepu_add_test_case_hybrid_opencl(mapoverlap_4d_new)
#skepu_add_test_case_hybrid_cuda(mapoverlap_4d_new)


# ------------------------------------------------
#   MapPool 4D fundamentals
# ------------------------------------------------

skepu_add_test_case_sequential(mappool_4d)
skepu_add_test_case_cpu(mappool_4d)
skepu_add_test_case_openmp(mappool_4d)
skepu_add_test_case_opencl(mappool_4d)
skepu_add_test_case_cuda(mappool_4d)
skepu_add_test_case_hybrid_opencl(mappool_4d)
#skepu_add_test_case_hybrid_cuda(mappool_4d)


# ------------------------------------------------
#   MapPool 4D small output sizes
# ------------------------------------------------

skepu_add_test_case_sequential(mappool_4d_small_output)
skepu_add_test_case_cpu(mappool_4d_small_output)
skepu_add_test_case_openmp(mappool_4d_small_output)
skepu_add_test_case_opencl(mappool_4d_small_output)
skepu_add_test_case_cuda(mappool_4d_small_output)
skepu_add_test_case_hybrid_opencl(mappool_4d_small_output)
#skepu_add_test_case_hybrid_cuda(mappool_4d_small_output)


# ------------------------------------------------
#   MapOverlap Indexed
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_indexed)
skepu_add_test_case_cpu(mapoverlap_indexed)
skepu_add_test_case_openmp(mapoverlap_indexed)
skepu_add_test_case_opencl(mapoverlap_indexed)
skepu_add_test_case_cuda(mapoverlap_indexed)
skepu_add_test_case_hybrid_opencl(mapoverlap_indexed)
#skepu_add_test_case_hybrid_cuda(mapoverlap_indexed)


# ------------------------------------------------
#   MapOverlap in place
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_in_place)
# The hybrid backends ignore the update modes (see mapoverlap_updatemode below): the test fails on hybrid OpenCL.
skepu_add_test_case_cpu(mapoverlap_in_place)
skepu_add_test_case_openmp(mapoverlap_in_place)
skepu_add_test_case_opencl(mapoverlap_in_place)
#skepu_add_test_case_cuda(mapoverlap_in_place)
#skepu_add_test_case_hybrid_opencl(mapoverlap_in_place)
#skepu_add_test_case_hybrid_cuda(mapoverlap_in_place)

# ------------------------------------------------
#   MapOverlap with variadic return
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_tuple)
skepu_add_test_case_cpu(mapoverlap_tuple)
skepu_add_test_case_openmp(mapoverlap_tuple)
skepu_add_test_case_opencl(mapoverlap_tuple)
skepu_add_test_case_cuda(mapoverlap_tuple)
skepu_add_test_case_hybrid_opencl(mapoverlap_tuple)
#skepu_add_test_case_hybrid_cuda(mapoverlap_tuple)

# ------------------------------------------------
#   MapOverlap update modes
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_updatemode)
# The update modes work with the sequential skeleton, CPU, OpenMP and OpenCL in all the variants; the hybrid backends ignore them.
skepu_add_test_case_cpu(mapoverlap_updatemode)
skepu_add_test_case_openmp(mapoverlap_updatemode)
skepu_add_test_case_opencl(mapoverlap_updatemode)
#skepu_add_test_case_cuda(mapoverlap_updatemode)
#skepu_add_test_case_hybrid_opencl(mapoverlap_updatemode)
#skepu_add_test_case_hybrid_cuda(mapoverlap_updatemode)

# ------------------------------------------------
#   MapOverlap 2D with an edge mode, input and output of the same size
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_2d_edge_modes)
skepu_add_test_case_cpu(mapoverlap_2d_edge_modes)
skepu_add_test_case_openmp(mapoverlap_2d_edge_modes)
skepu_add_test_case_opencl(mapoverlap_2d_edge_modes)
