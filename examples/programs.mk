# The SkePU example programs, their arguments and the settings of the examples. Included by examples/Makefile before make/skepu.mk.
# A Makefile that includes ../Makefile (examples/dnn/Makefile) sets PROGS first and builds only its own programs.

TRACE ?= 1
RUN_CWD = .
RUN_LOG = 1

# Linked into every program: image reading and writing (lodepng)
COMMON_SOURCES ?= $(EXAMPLES_DIR)lodepng.cpp
USER_INCLUDES  += -I $(EXAMPLES_DIR)

ifndef PROGS
PROGS += backend_switching
PROGS += blas_tour
PROGS += cellular_automaton
PROGS += cma
PROGS += conjugate_gradient
PROGS += conjugate_gradient_sparse
PROGS += coulombic
PROGS += dotproduct
PROGS += edgedetect
PROGS += fusion_patterns
PROGS += heat_diffusion # no red-black update mode or mirror mode in CUDA
PROGS += horner
PROGS += image_manip
PROGS += mandelbrot
PROGS += maxmin
PROGS += median
PROGS += miller_rabin
PROGS += mmmult
PROGS += mmmult_row_col
PROGS += montecarlo
PROGS += mvmult
PROGS += mvmult_row
PROGS += nbody
PROGS += nbody_mappairs
PROGS += ode_fisher
PROGS += ode_lorenz
PROGS += ozaki_gemm
PROGS += perlin
PROGS += ppmcc
PROGS += pso
PROGS += psnr
PROGS += riemann_sum
PROGS += taylor

# The programs that are built for the CPU backends only (the sequential version, and the parallel version for OpenMP): the sparse matrices
# are not supported by the OpenCL, CUDA and MPI backends yet
CPU_ONLY_PROGS = blas_tour conjugate_gradient_sparse

# The programs that the MPI backend supports (see examples/CMake/example-options.cmake): the others are not built in MPI mode
MPI_PROGS = cellular_automaton cma conjugate_gradient coulombic dotproduct horner mandelbrot maxmin miller_rabin \
	mmmult mmmult_row_col montecarlo mvmult mvmult_row nbody nbody_mappairs ode_fisher ode_lorenz ozaki_gemm perlin ppmcc pso psnr riemann_sum taylor
ifdef MPI_MODE
PROGS = $(filter $(MPI_PROGS),$(PROGS))
endif
endif

# Arguments of make run/<program>; RUN_BACKEND is the backend they use
RUN_BACKEND = OpenCL
SIZE_BACKEND = 2048
lookup.blas_tour           = 64 OpenMP
lookup.cellular_automaton  = 2 256 10 0.1 $(RUN_BACKEND)
lookup.cma                 = $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.conjugate_gradient  = 256 $(RUN_BACKEND)
lookup.conjugate_gradient_sparse = 4096 OpenMP
lookup.coulombic           = $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.dotproduct          = $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.edgedetect          = edge data/goose.png $(RUN_DIR)/result_edgedetect 2 $(RUN_BACKEND)
lookup.heat_diffusion      = 2 64 100 $(RUN_BACKEND) redblack
lookup.horner              = 3 1 2 3 4 $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.image_manip         = $(RUN_BACKEND)
lookup.mandelbrot          = 256 256 $(RUN_BACKEND)
lookup.maxmin              = $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.median              = data/goose.png $(RUN_DIR)/result_median 2 $(RUN_BACKEND)
lookup.miller_rabin        = $(RUN_BACKEND)
lookup.mmmult              = 128 64 32 $(RUN_BACKEND)
lookup.mmmult_row_col      = 128 64 32 $(RUN_BACKEND)
lookup.montecarlo          = $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.mvmult              = 128 128 $(RUN_BACKEND)
lookup.mvmult_row          = 128 128 $(RUN_BACKEND)
lookup.nbody               = 100 100 $(RUN_BACKEND)
lookup.nbody_mappairs      = 100 100 $(RUN_BACKEND)
lookup.ode_fisher          = 64 20 $(RUN_BACKEND)
lookup.ode_lorenz          = $(SIZE_BACKEND) 40 $(RUN_BACKEND)
lookup.ozaki_gemm          = 48 7 $(RUN_BACKEND)
lookup.perlin              = 256 256 0.5 5 4 1 $(RUN_BACKEND) --out $(RUN_DIR)/result_perlin
lookup.ppmcc               = $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.pso                 = 4096 32 200 $(RUN_BACKEND)
lookup.psnr                = 128 128 $(RUN_BACKEND)
lookup.riemann_sum         = 0 1 $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.taylor              = 0.1 $(SIZE_BACKEND) $(RUN_BACKEND)
