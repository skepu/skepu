# The MapOverlap and MapPool tests. A test program has a cost of its own to build, so the tests that run on the same backends are in one
# program (each part is in a namespace of its own, and its test cases have the name of the file it came from as a prefix).

# ------------------------------------------------
#   Fundamentals of MapOverlap 2D and 4D, with expanding edges, indexed and tuple user functions
#   (mapoverlap_2d, mapoverlap_2d_expanding, mapoverlap_4d, mapoverlap_4d_expanding, mapoverlap_indexed, mapoverlap_tuple)
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_basic)
skepu_add_test_case_cpu(mapoverlap_basic)
skepu_add_test_case_openmp(mapoverlap_basic)
skepu_add_test_case_opencl(mapoverlap_basic)
skepu_add_test_case_cuda(mapoverlap_basic)
skepu_add_test_case_hybrid_opencl(mapoverlap_basic)
#skepu_add_test_case_hybrid_cuda(mapoverlap_basic)

# ------------------------------------------------
#   MapOverlap 1D (also rows and columns), 2D, 3D and 4D with the edge modes, the overlaps and the checks of the arguments
#   (mapoverlap_1d, mapoverlap_1d_rowcol, mapoverlap_2d_new, mapoverlap_3d_new, mapoverlap_4d_new)
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_new)
skepu_add_test_case_cpu(mapoverlap_new)
skepu_add_test_case_openmp(mapoverlap_new)
skepu_add_test_case_opencl(mapoverlap_new)
skepu_add_test_case_cuda(mapoverlap_new)
skepu_add_test_case_hybrid_opencl(mapoverlap_new)
#skepu_add_test_case_hybrid_cuda(mapoverlap_new)

# ------------------------------------------------
#   The mirror mode (edge mode None, input of the size of the output) of MapOverlap 1D to 4D, the edge modes of 2D, the update modes and the
#   in place calls
#   (mapoverlap_1d_mirror, mapoverlap_2d_mirror, mapoverlap_3d_mirror, mapoverlap_4d_mirror, mapoverlap_2d_edge_modes, mapoverlap_in_place,
#   mapoverlap_updatemode)
# ------------------------------------------------

# The mirror mode is not implemented for the CUDA and hybrid backends: the call stops with an error. MapOverlap2D is not implemented for the
# hybrid backend, and the hybrid backends ignore the update modes (the test fails on hybrid OpenCL).
skepu_add_test_case_sequential(mapoverlap_edges)
skepu_add_test_case_cpu(mapoverlap_edges)
skepu_add_test_case_openmp(mapoverlap_edges)
skepu_add_test_case_opencl(mapoverlap_edges)
#skepu_add_test_case_cuda(mapoverlap_edges)
#skepu_add_test_case_hybrid_opencl(mapoverlap_edges)
#skepu_add_test_case_hybrid_cuda(mapoverlap_edges)

# ------------------------------------------------
#   MapPool 2D, 3D and 4D, also with small output sizes
#   (mappool_2d, mappool_3d, mappool_4d, mappool_2d_small_output, mappool_3d_small_output, mappool_4d_small_output)
# ------------------------------------------------

skepu_add_test_case_sequential(mappool)
skepu_add_test_case_cpu(mappool)
skepu_add_test_case_openmp(mappool)
skepu_add_test_case_opencl(mappool)
skepu_add_test_case_cuda(mappool)
skepu_add_test_case_hybrid_opencl(mappool)
#skepu_add_test_case_hybrid_cuda(mappool)

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
#   MapOverlap 3D fundamentals (a program of its own: it sets SKEPU_DEBUG, which makes the library print what it does)
# ------------------------------------------------

skepu_add_test_case_sequential(mapoverlap_3d)
skepu_add_test_case_cpu(mapoverlap_3d)
skepu_add_test_case_openmp(mapoverlap_3d)
skepu_add_test_case_opencl(mapoverlap_3d)
skepu_add_test_case_cuda(mapoverlap_3d)
skepu_add_test_case_hybrid_opencl(mapoverlap_3d)
#skepu_add_test_case_hybrid_cuda(mapoverlap_3d)
