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

# ------------------------------------------------------------------- #
# Default fundamental paths and options.
# ------------------------------------------------------------------- #

# Default path to the SkePU installation
# Override in Makefile.in
SKEPU_SRC_PATH = ../

# Default path to LLVM build directory (relative to SkePU installation path)
# Override in Makefile.in
LLVM_BUILD_REL_PATH = /build/

# Default backend set for SkePU precompiler
# Override in Makefile.in
BACKENDS = -openmp -opencl -cuda

# Default backend set for SkePU precompiler
# Override in Makefile.in
SKEPU_DEBUG_LEVEL = 0

# Default values for SkePU backend compiler options
# Override in Makefile.in
SKEPU_OPTIONS = -std=c++17 -O3 -g

# Default values for SkePU backend compiler options
# Override in Makefile.in
PRECOMPILER_OPTIONS = --no-preserve-lines #--verbose

# Backend-specific flags
# Override in Makefile.in
BACKEND_FLAGS = -DSKEPU_TRACING #-DSKEPU_SIMULATE_MULTI_GPU # -DSKEPU_SET_DEFAULT_BACKEND_CUDA #  -g -O3

# Default OpenCL specific flags (matching a CUDA-provided installation)
# Override in Makefile.in
OPENCL_FLAGS = -I /usr/local/cuda/include -L /usr/local/cuda/lib64/ -lOpenCL -Wno-deprecated-declarations

# Default compilers 
# Override in Makefile.in
CXX = clang++
SEQ_CXX = clang++

# Uncomment for launching SkePU precompiler in debugger
# DBGR = lldb --

CLUSTER_CXX      = mpic++
CLUSTER_BACKENDS = -openmp
CLUSTER_CXXFLAGS = -std=c++11 -fopenmp $$(pkg-config --cflags starpu-1.3 starpumpi-1.3) -I ./  -DSKEPU_DEBUG=3  -g -O3
CLUSTER_LIBS     = $$(pkg-config --libs starpu-1.3 starpumpi-1.3)



# ------------------------------------------------------------------- #
# Includes.
# ------------------------------------------------------------------- #

Makefile.in:
	./generate_makefile.in.sh

# Overrides default values for various paths and options
include Makefile.in


# ------------------------------------------------------------------- #
# Derived paths and option sets.
# ------------------------------------------------------------------- #

# Path to Clang source directory (repository root)
CLANG_INCLUDE_PATH = $(SKEPU_SRC_PATH)/llvm/clang/lib/Headers/

# Default path to the SkePU sources
SKEPU_INCLUDE_PATH = $(SKEPU_SRC_PATH)/skepu-headers/src/

# Location for LLVM build directory
LLVM_BUILD_PATH = $(SKEPU_SRC_PATH)/$(LLVM_BUILD_REL_PATH)

# Location of SkePU precompiler binary
SKEPU_TOOL_PATH = $(LLVM_BUILD_PATH)/llvm/bin/skepu-tool

# Add SkePU-specific debug level to options set
SKEPU_OPTIONS += -DSKEPU_DEBUG=$(SKEPU_DEBUG_LEVEL)


# ---------------------------------------------------------------------------- #
# Test program names and target directories.

PAR_DIR     = parallel
SEQ_DIR     = sequential
CLUSTER_DIR = cluster
RUN_DIR     = run

# All SkePU example programs (used for 'make all-[parallel|sequential|cluster]' and 'make clean').

PROGS=
PROGS += cellular_automaton
PROGS += cma
#PROGS += conjugate_gradient # Error related to repeated instantiations of different types in SkePU BLAS
PROGS += coulombic
PROGS += dotproduct
PROGS += edgedetect
PROGS += heat_diffusion # no red-black update mode in CUDA
PROGS += horner
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
PROGS += ppmcc
PROGS += psnr
PROGS += riemann_sum
PROGS += taylor

RUN_BACKEND = OpenCL
SIZE_BACKEND = 2048
lookup.cellular_automaton  := 2 $(SIZE_BACKEND) 10 $(RUN_BACKEND)
lookup.cma                 := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.conjugate_gradient  := 256 $(RUN_BACKEND)
lookup.coulombic           := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.dotproduct          := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.heat_diffusion      := 2 $(SIZE_BACKEND) 10 $(RUN_BACKEND)
lookup.horner              := 3 1 2 3 4 $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.image_manip         := data/test_image.png result_image blur 3 $(RUN_BACKEND)
lookup.mandelbrot          := 256 256 $(RUN_BACKEND)
lookup.maxmin              := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.miller_rabin        := $(RUN_BACKEND)
lookup.mmmult              := 128 64 32 $(RUN_BACKEND)
lookup.mmmult_row_col      := 128 64 32 $(RUN_BACKEND)
lookup.montecarlo          := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.mvmult              := 128 128 $(RUN_BACKEND)
lookup.mvmult_row          := 128 128 $(RUN_BACKEND)
lookup.nbody               := 100 100 $(RUN_BACKEND)
lookup.nbody_mappairs      := 100 100 $(RUN_BACKEND)
lookup.ppmcc               := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.psnr                := 128 128 $(RUN_BACKEND)
lookup.taylor              := 0.1 $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.riemann_sum         := 0 1 $(SIZE_BACKEND) $(RUN_BACKEND)


TESTS          = $(addprefix $(SEQ_DIR)/,      $(PROGS))
PTESTS         = $(addprefix $(PAR_DIR)/,      $(PROGS))
CLUSTER_TESTS  = $(addprefix $(CLUSTER_DIR)/,  $(CLUSTER_PROGS))
RUN_TESTS      = $(addprefix $(RUN_DIR)/,      $(PROGS))

# ---------------------------------------------------------------------------- #
# Compiler flags begin here.

# Flags for precompiler.
SKEPU_FLAGS = $(PRECOMPILER_OPTIONS)
SKEPU_FLAGS += -- 
SKEPU_FLAGS += $(SKEPU_TOOL_USER_FLAGS)
SKEPU_FLAGS += -I $(CLANG_INCLUDE_PATH)
SKEPU_FLAGS += -I $(SKEPU_INCLUDE_PATH)

# Activate backend flags for CUDA backend
ifneq (,$(findstring cuda, $(BACKENDS)))
BACKEND_FLAGS += -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used"
NVCCPASS = -Xcompiler
FILETYPE = cu
CXX = nvcc
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
TARGET_FLAGS = $(SKEPU_OPTIONS) -I $(SKEPU_INCLUDE_PATH) -I ./  $(NVCCPASS) -Wno-attributes $(BACKEND_FLAGS)

# Flags for non-preprocessed compilation.
SEQ_FLAGS    = $(SKEPU_OPTIONS) -I $(SKEPU_INCLUDE_PATH) -Wno-attributes # -fno-rtti -fno-exceptions


# ---------------------------------------------------------------------------- #
# Make recipies begin here.

# Builds all test programs, sequential implementation, witout using the precompiler.
all-sequential: $(TESTS)

# Builds all test programs, parallel implementations, using the precompiler.
all-parallel: $(PTESTS)

all-cluster: $(CLUSTER_TESTS)

all-run: $(RUN_TESTS)




capi:
	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name cskepu_precompiled $(SKEPU_INCLUDE_PATH)/cskepu.cpp  -dir ./test_build $(SKEPU_FLAGS)
	$(DBGR) $(CXX) -c ./test_build/cskepu_precompiled.$(FILETYPE)  $(TARGET_FLAGS) -o ./test_build/cskepu.o

	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name c_skeleton_precompiled c_skeleton.cpp  -dir ./test_build $(SKEPU_FLAGS)
	$(DBGR) $(CXX) -c ./test_build/c_skeleton_precompiled.$(FILETYPE)  $(TARGET_FLAGS) -o ./test_build/c_skeleton.o

	clang -c skepu_in_c.c -I $(SKEPU_INCLUDE_PATH) -o ./test_build/skepu_in_c.o
	$(CXX) ./test_build/cskepu.o ./test_build/skepu_in_c.o ./test_build/c_skeleton.o $(TARGET_FLAGS) -o ./test_build/skepu_in_c


fapi:
	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name cskepu_precompiled $(SKEPU_INCLUDE_PATH)/cskepu.cpp  -dir ./test_build $(SKEPU_FLAGS)
	$(DBGR) $(CXX) -c ./test_build/cskepu_precompiled.$(FILETYPE)  $(TARGET_FLAGS) -o ./test_build/cskepu.o

	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name c_skeleton_precompiled c_skeleton.cpp  -dir ./test_build $(SKEPU_FLAGS)
	$(DBGR) $(CXX) -c ./test_build/c_skeleton_precompiled.$(FILETYPE)  $(TARGET_FLAGS) -o ./test_build/c_skeleton.o

	gfortran -c $(SKEPU_INCLUDE_PATH)/fskepu.f90 -o ./test_build/fskepu.o
	gfortran -c f_skeleton.f90 -o ./test_build/f_skeleton.o

	gfortran -c skepu_in_fortran.f90 -I $(SKEPU_INCLUDE_PATH) -o ./test_build/skepu_in_fortran.o
	$(CXX) -lgfortran ./test_build/cskepu.o ./test_build/fskepu.o ./test_build/skepu_in_fortran.o ./test_build/c_skeleton.o ./test_build/f_skeleton.o $(TARGET_FLAGS) -o ./test_build/skepu_in_fortran

all: Makefile.in all-sequential all-parallel all-cluster

$(PAR_DIR):
	@mkdir -p $(PAR_DIR)

$(SEQ_DIR):
	@mkdir -p $(SEQ_DIR)

$(CLUSTER_DIR):
	@mkdir -p $(CLUSTER_DIR)

$(RUN_DIR):
	@mkdir -p $(RUN_DIR)

# Builds a single sequential test program.
$(SEQ_DIR)/% : %.cpp $(SEQ_DIR)
	$(DBGR) $(SEQ_CXX) $(SEQ_FLAGS) lodepng.cpp $< -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/%: %.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name $*_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS) -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/image_manip: image_manip.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name image_manip_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS)  -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/demo: demo.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name demo_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS)  -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/fusion_eval: fusion_eval.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name fusion_eval_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS)  -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/perlin: perlin.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name perlin_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS)  -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/uniformnoise: uniformnoise.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name uniformnoise_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS)  -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/multi: multi1.cpp multi2.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name multi1_precompiled multi1.cpp  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name multi2_precompiled multi2.cpp  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) -fopenmp $(PAR_DIR)/multi1_precompiled.$(FILETYPE) $(PAR_DIR)/multi2_precompiled.$(FILETYPE)  $(TARGET_FLAGS)  -o $@

$(CLUSTER_DIR)/%.o: %.cpp $(CLUSTER_DIR)
	@echo "Precompiling $<"
	$(DBGR) $(SKEPU_TOOL_PATH) $(CLUSTER_BACKENDS) -dir=$(CLUSTER_DIR) -name $*_precompiled $< $(SKEPU_FLAGS)
	@echo "Compiling $<"
	$(CLUSTER_CXX) -o $@ $(CLUSTER_DIR)/$*_precompiled.cpp $(CLUSTER_CXXFLAGS) -c -I$(SKEPU_INCLUDE_PATH) -DSKEPU_STARPU_MPI
	# NOTE: -O0 generates undefined referece to proxy_tags

$(CLUSTER_DIR)/%: $(CLUSTER_DIR)/%.o
	@echo "Linking $*"
	@$(CLUSTER_CXX) -std=c++11 -fopenmp -o $@ $< $(CLUSTER_LIBS)

lodeppng.o:
	@$(CLUSTER_CXX) -std=c++11 -fopenmp -o $@ lodepng.cpp $(CLUSTER_LIBS)

$(CLUSTER_DIR)/image_manip: $(CLUSTER_DIR)/image_manip.o lodepng.o
	@echo "Linking $*"
	@$(CLUSTER_CXX) -std=c++11 -fopenmp -o $@ $< lodepng.o $(CLUSTER_LIBS)


$(RUN_DIR)/% : $(PAR_DIR)/% $(RUN_DIR)
	$< ${lookup.$*} > $@.log 2>&1


# Deletes all temporary files (including all precompiled sources) and binaries.
clean:
	$(RM) -R $(SEQ_DIR)
	$(RM) -R $(PAR_DIR)
	$(RM) -R $(CLUSTER_DIR)
	$(RM) -R $(RUN_DIR)
