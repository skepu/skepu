################################################################################
#                     ____  _        ____  _   _   ____                        #
#                    / ___|| | _____|  _ \| | | | |___ \                       #
#                    \___ \| |/ / _ \ |_) | | | |   __) |                      #
#                     ___) |   <  __/  __/| |_| |  / __/                       #
#                    |____/|_|\_\___|_|    \___/  |_____|                      #
#                                                                              #
#                          ~ SkePU 2 main Makefile ~                           #
################################################################################
.DEFAULT_GOAL=all

# Default backend set for SkePU precompiler.
BACKENDS = -openmp -cuda # -opencl

SKEPU_OPTIONS =
PRECOMPILER_OPTIONS = --silent

# Backend-specific flags
BACKEND_FLAGS =

# Default OpenCL specific flags (matching a CUDA-provided installation)
OPENCL_FLAGS = -I /usr/local/cuda/include -L /usr/local/cuda/lib64/ -lOpenCL

# path to Clang source directory (repository root)
CLANG_INCLUDE = $(PWD)/../llvm/clang/lib/Headers

# path to the SkePU sources
SKEPU_INCLUDE = $(PWD)/../skepu-headers/src

CXX = g++-7

MPICXX=mpic++
MPIBACKENDS=-openmp
MPICXXFLAGS=-std=c++11 -fopenmp $$(pkg-config --cflags starpu-1.3 starpumpi-1.3)
MPILIBS=$$(pkg-config --libs starpu-1.3 starpumpi-1.3)

MPI_SRC=nbody_mpi.cpp
MPI_EXEC=nbody

DBGR = # gdb --args

CLANGTOOL_USER_FLAGS =
# ---------------------------------------------------------------------------- #
# Compilers.

# Conditionally choose either g++ or nvcc for target compiler,
# depending on whether CUDA is in the list of backends
# This will also be used for non-preprocessed compilation of SkePU programs.
ifneq (,$(findstring cuda, $(BACKENDS)))
CXX = nvcc
endif

# Location of SkePU precompiler binary.
SKEPU = $(PWD)/../build/llvm/bin/skepu-tool


# ---------------------------------------------------------------------------- #
# Test program names and target directories.

PAR_DIR = parallel
SEQ_DIR = sequential
MPI_DIR = mpi

# All SkePU 2 exmple programs (used for 'make all-[parallel|sequential]' and 'make clean').
# MapOverlap skeleton disabled until OpenCL backend is resolved.
PROGS=
#PROGS+=call
PROGS+=cma
#PROGS+=containers
PROGS+=coulombic
PROGS+=dotproduct
# PROGS+=lambda
PROGS+=mandelbrot
PROGS+=map
# PROGS+=mapoverlap
PROGS+=mapreduce
# PROGS+=median
PROGS+=mmmult
PROGS+=mvmult
PROGS+=nbody
PROGS+=ppmcc
PROGS+=psnr
PROGS+=reduce
PROGS+=scan
PROGS+=taylor
PROGS+=writeback

TESTS = $(addprefix $(SEQ_DIR)/, $(PROGS))
PTESTS = $(addprefix $(PAR_DIR)/, $(PROGS))
MPITESTS=$(addprefix $(MPI_DIR)/, $(MPI_EXEC))

# ---------------------------------------------------------------------------- #
# Compiler flags begin here.

# Flags for precompiler.
SKEPU_FLAGS = $(PRECOMPILER_OPTIONS)
SKEPU_FLAGS += -- -std=c++11 -Wno-expansion-to-defined
SKEPU_FLAGS += -I $(CLANG_INCLUDE)
SKEPU_FLAGS += -I $(SKEPU_INCLUDE)
SKEPU_FLAGS += $(CLANGTOOL_USER_FLAGS)

# Activate backend flags for CUDA backend
ifneq (,$(findstring cuda, $(BACKENDS)))
BACKEND_FLAGS += -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used"
NVCCPASS = -Xcompiler
FILETYPE = cu
else
FILETYPE = cpp
endif

# Activate backend flags for OpenMP backend
ifneq (,$(findstring openmp, $(BACKENDS)))
BACKEND_FLAGS += $(NVCCPASS) -fopenmp
endif

# Activate backend flags for OpenCL backend
ifneq (,$(findstring opencl, $(BACKENDS)))
BACKEND_FLAGS += $(OPENCL_FLAGS)
endif

# Flags for target compiler (preprocessed sources).
TARGET_FLAGS = -g -O3 -std=c++11 $(SKEPU_OPTIONS) -I $(SKEPU_INCLUDE) -I ./  $(NVCCPASS) -Wno-attributes $(BACKEND_FLAGS)

# Flags for non-preprocessed compilation.
SEQ_FLAGS = -std=c++11 -O3 -I $(SKEPU_INCLUDE) # -Wno-attributes # -fno-rtti -fno-exceptions

# ---------------------------------------------------------------------------- #
# Make recipies begin here.

# Builds all test programs, sequential implementation, witout using the precompiler.
all-sequential: $(TESTS)

# Builds all test programs, parallel implementations, using the precompiler.
all-parallel: $(PTESTS)

all-mpi: $(MPITESTS)

all: all-sequential all-parallel all-mpi

$(PAR_DIR):
	@mkdir -p $(PAR_DIR)

$(SEQ_DIR):
	@mkdir -p $(SEQ_DIR)

$(MPI_DIR):
	@mkdir -p $(MPI_DIR)

# Builds a single sequential test program.
$(SEQ_DIR)/% : %.cpp $(SEQ_DIR)
	$(DBGR) $(CXX) $(SEQ_FLAGS) $< -o $@

# Builds a single sequential test program.
$(SEQ_DIR)/median : median.cpp $(SEQ_DIR)
	$(DBGR) $(CXX) $(SEQ_FLAGS) lodepng.cpp $< -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/%: %.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name $*_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE)  $(TARGET_FLAGS) -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/median: median.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name median_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS)  -o $@

$(MPI_DIR)/%.o: %_mpi.cpp $(MPI_DIR)
	@echo "Precompiling $<"
	@$(SKEPU) $(MPIBACKENDS) -dir=$(MPI_DIR) -name $*_precompiled $< \
		$(SKEPU_FLAGS)
	@echo "Compiling $<"
	@$(MPICXX) -o $@ $(MPI_DIR)/$*_precompiled.cpp \
		$(MPICXXFLAGS) -c -g -O0 -I$(SKEPU_INCLUDE) -DSKEPU_MPI_STARPU

$(MPI_DIR)/%: $(MPI_DIR)/%.o
	@echo "Linking $*"
	@$(MPICXX) -std=c++11 -fopenmp -o $@ $< $(MPILIBS)

# Deletes all temporary files (including all precompiled sources) and binaries.
clean:
	$(RM) -R $(SEQ_DIR)
	$(RM) -R $(PAR_DIR)
	$(RM) -R $(MPI_DIR)
