# The MapOverlap and MapPool tests; tests that run on the same backends share a program.

# ------------------------------------------------
#   Fundamentals of MapOverlap 2D and 4D, with expanding edges, indexed and tuple user functions
# ------------------------------------------------

skepu_add_test(mapoverlap_basic)

# ------------------------------------------------
#   MapOverlap 1D (also rows and columns), 2D, 3D and 4D with the edge modes, the overlaps and the checks of the arguments
# ------------------------------------------------

skepu_add_test(mapoverlap_new)

# ------------------------------------------------
#   The mirror mode (edge mode None, input of the size of the output) of MapOverlap 1D to 4D, the edge modes of 2D, the update modes and the
#   in place calls
# ------------------------------------------------

# The mirror mode is not implemented for the hybrid backends (the call stops with an error). MapOverlap2D is not implemented for the
# hybrid backend, and the hybrid backends ignore the update modes (the test fails on hybrid OpenCL).
skepu_add_test(mapoverlap_edges EXCLUDE hybrid_opencl hybrid_cuda)

# ------------------------------------------------
#   MapPool 2D, 3D and 4D, also with small output sizes
# ------------------------------------------------

skepu_add_test(mappool)

# ------------------------------------------------
#   MapOverlap default values
# ------------------------------------------------

skepu_add_test(mapoverlap_default_values)

# ------------------------------------------------
#   MapOverlap 1D on matrices with rows and columns longer than a thread block
# ------------------------------------------------

# hybrid OpenCL: fails, a Matrix column access is out of bounds
skepu_add_test(mapoverlap_1d_wide EXCLUDE hybrid_opencl hybrid_cuda)

# ------------------------------------------------
#   MapOverlap 3D fundamentals (a program of its own: it sets SKEPU_DEBUG, which makes the library print what it does)
# ------------------------------------------------

skepu_add_test(mapoverlap_3d)
