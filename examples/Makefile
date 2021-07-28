#######################################################################
#                     ____  _        ____  _   _                      #
#                    / ___|| | _____|  _ \| | | |                     #
#                    \___ \| |/ / _ \ |_) | | | |                     #
#                     ___) |   <  __/  __/| |_| |                     #
#                    |____/|_|\_\___|_|    \___/                      #
#                                                                     #
#                       ~ SkePU main Makefile ~                       #
#######################################################################
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
SEQ_CXX = g++-11
#DBGR = lldb -- 

MPICXX=mpic++
MPIBACKENDS=-openmp
MPICXXFLAGS=-std=c++11 -fopenmp $$(pkg-config --cflags starpu-1.3 starpumpi-1.3)
MPILIBS=$$(pkg-config --libs starpu-1.3 starpumpi-1.3)

MPI_SRC=nbody_mpi.cpp
MPI_EXEC=nbody


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
RUN_DIR = run

# All SkePU example programs (used for 'make all-[parallel|sequential]' and 'make clean').

# MapOverlap skeleton disabled until OpenCL backend is resolved.
PROGS=
#PROGS+=call
PROGS+=cellular_automaton
PROGS+=cma
PROGS+=conjugate_gradient
PROGS+=coulombic
PROGS+=dotproduct
PROGS+=edgedetect
PROGS+=heat_diffusion
PROGS+=mandelbrot
PROGS+=maxmin
PROGS+=median
PROGS+=miller_rabin
PROGS+=mmmult
PROGS+=montecarlo
PROGS+=mvmult
PROGS+=mvmult_row
PROGS+=nbody
PROGS+=nbody_mappairs
PROGS+=ppmcc
PROGS+=psnr
PROGS+=riemann_sum
PROGS+=taylor


# if opencl isn't enabled, add more progs
ifeq (,$(findstring opencl, $(BACKENDS)))
PROGS+=mmmult_row_col # error!
endif

RUN_BACKEND = OpenCL
SIZE_BACKEND = 2048
lookup.cellular_automaton  := 2 $(SIZE_BACKEND) 10 $(RUN_BACKEND)
lookup.cma                 := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.conjugate_gradient  := 256 $(RUN_BACKEND)
lookup.coulombic           := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.dotproduct          := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.edgedetect          := blur data/test_image.png result_image 3 $(RUN_BACKEND)
lookup.heat_diffusion      := 2 $(SIZE_BACKEND) 10 $(RUN_BACKEND)
lookup.mandelbrot          := 256 256 $(RUN_BACKEND)
lookup.maxmin              := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.median              := data/test_image.png result_image 10 $(RUN_BACKEND)
lookup.miller_rabin        := $(RUN_BACKEND)
lookup.mmmult              := 128 64 32 $(RUN_BACKEND)
lookup.mmmult_row_col      := 128 64 32 $(RUN_BACKEND)
lookup.montecarlo          := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.mvmult              := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.mvmult_row          := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.nbody               := 100 100 $(RUN_BACKEND)
lookup.nbody_mappairs      := 100 100 $(RUN_BACKEND)
lookup.ppmcc               := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.psnr                := 128 128 $(RUN_BACKEND)
lookup.taylor              := 0.1 $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.riemann_sum         := 0 1 $(SIZE_BACKEND) $(RUN_BACKEND)


TESTS = $(addprefix $(SEQ_DIR)/, $(PROGS))
PTESTS = $(addprefix $(PAR_DIR)/, $(PROGS))
MPITESTS=$(addprefix $(MPI_DIR)/, $(MPI_EXEC))
RUN_TESTS = $(addprefix $(RUN_DIR)/, $(PROGS))

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
TARGET_FLAGS = -std=c++11 $(SKEPU_OPTIONS) -I $(SKEPU_INCLUDE) -I ./  $(NVCCPASS) -Wno-attributes $(BACKEND_FLAGS)

# Flags for non-preprocessed compilation.
SEQ_FLAGS = -std=c++11 -O0 -g -DSKEPU_DEBUG=3 -I $(SKEPU_INCLUDE) -Wno-attributes # -fno-rtti -fno-exceptions

# ---------------------------------------------------------------------------- #
# Make recipies begin here.

# Builds all test programs, sequential implementation, witout using the precompiler.
all-sequential: $(TESTS)

# Builds all test programs, parallel implementations, using the precompiler.
all-parallel: $(PTESTS)

all-mpi: $(MPITESTS)
	
all-run: $(RUN_TESTS)
	



capi:
	$(DBGR) $(SKEPU) $(BACKENDS) -name cskepu_precompiled $(SKEPU_INCLUDE)/cskepu.cpp  -dir ./test_build $(SKEPU_FLAGS)
	$(DBGR) $(CXX) -c ./test_build/cskepu_precompiled.$(FILETYPE)  $(TARGET_FLAGS) -o ./test_build/cskepu.o
	
	$(DBGR) $(SKEPU) $(BACKENDS) -name c_skeleton_precompiled c_skeleton.cpp  -dir ./test_build $(SKEPU_FLAGS)
	$(DBGR) $(CXX) -c ./test_build/c_skeleton_precompiled.$(FILETYPE)  $(TARGET_FLAGS) -o ./test_build/c_skeleton.o
	
	clang -c skepu_in_c.c -I $(SKEPU_INCLUDE) -o ./test_build/skepu_in_c.o
	$(CXX) ./test_build/cskepu.o ./test_build/skepu_in_c.o ./test_build/c_skeleton.o $(TARGET_FLAGS) -o ./test_build/skepu_in_c


fapi:
	$(DBGR) $(SKEPU) $(BACKENDS) -name cskepu_precompiled $(SKEPU_INCLUDE)/cskepu.cpp  -dir ./test_build $(SKEPU_FLAGS)
	$(DBGR) $(CXX) -c ./test_build/cskepu_precompiled.$(FILETYPE)  $(TARGET_FLAGS) -o ./test_build/cskepu.o
	
	$(DBGR) $(SKEPU) $(BACKENDS) -name c_skeleton_precompiled c_skeleton.cpp  -dir ./test_build $(SKEPU_FLAGS)
	$(DBGR) $(CXX) -c ./test_build/c_skeleton_precompiled.$(FILETYPE)  $(TARGET_FLAGS) -o ./test_build/c_skeleton.o
	
	gfortran -c $(SKEPU_INCLUDE)/fskepu.f90 -o ./test_build/fskepu.o
	gfortran -c f_skeleton.f90 -o ./test_build/f_skeleton.o
	
	gfortran -c skepu_in_fortran.f90 -I $(SKEPU_INCLUDE) -o ./test_build/skepu_in_fortran.o
	$(CXX) -lgfortran ./test_build/cskepu.o ./test_build/fskepu.o ./test_build/skepu_in_fortran.o ./test_build/c_skeleton.o ./test_build/f_skeleton.o $(TARGET_FLAGS) -o ./test_build/skepu_in_fortran

all: all-sequential all-parallel all-mpi

$(PAR_DIR):
	@mkdir -p $(PAR_DIR)

$(SEQ_DIR):
	@mkdir -p $(SEQ_DIR)

$(MPI_DIR):
	@mkdir -p $(MPI_DIR)

$(RUN_DIR):
	@mkdir -p $(RUN_DIR)

# Builds a single sequential test program.
$(SEQ_DIR)/% : %.cpp $(SEQ_DIR)
	$(DBGR) $(SEQ_CXX) $(SEQ_FLAGS) $< -o $@

# Builds a single sequential test program.
$(SEQ_DIR)/median : median.cpp $(SEQ_DIR)
	$(DBGR) $(SEQ_CXX) $(SEQ_FLAGS) lodepng.cpp $< -o $@

# Builds a single sequential test program.
$(SEQ_DIR)/edgedetect : edgedetect.cpp $(SEQ_DIR)
	$(DBGR) $(SEQ_CXX) $(SEQ_FLAGS) lodepng.cpp $< -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/%: %.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name $*_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE)  $(TARGET_FLAGS) -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/median: median.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name median_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS)  -o $@
	
# Precompiles and builds a single parallel test program.
$(PAR_DIR)/edgedetect: edgedetect.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name edgedetect_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS)  -o $@
	

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/multi: multi1.cpp multi2.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name multi1_precompiled multi1.cpp  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(SKEPU) $(BACKENDS) -name multi2_precompiled multi2.cpp  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) -fopenmp $(PAR_DIR)/multi1_precompiled.$(FILETYPE) $(PAR_DIR)/multi2_precompiled.$(FILETYPE)  $(TARGET_FLAGS)  -o $@

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

$(RUN_DIR)/% : $(PAR_DIR)/% $(RUN_DIR)
	$< ${lookup.$*} > $@.log 2>&1

# Deletes all temporary files (including all precompiled sources) and binaries.
clean:
	$(RM) -R $(SEQ_DIR)
	$(RM) -R $(PAR_DIR)
	$(RM) -R $(MPI_DIR)
	$(RM) -R $(RUN_DIR)
