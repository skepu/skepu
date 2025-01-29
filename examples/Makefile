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

# Init Cmake
# cmake -DCMAKE_BUILD_TYPE=Release ..



# Default backend set for SkePU precompiler.
BACKENDS = -openmp -opencl #-cuda

SKEPU_OPTIONS = -std=c++17 -O3 #-g -DSKEPU_DEBUG=3 #-DSKEPU_DEBUG_PRNG
#SKEPU_OPTIONS +=

PRECOMPILER_OPTIONS = --no-preserve-lines #--verbose

# Backend-specific flags
BACKEND_FLAGS = -DSKEPU_TRACING #-DSKEPU_SIMULATE_MULTI_GPU # -DSKEPU_SET_DEFAULT_BACKEND_CUDA #  -g -O3

# Default OpenCL specific flags (matching a CUDA-provided installation)
OPENCL_FLAGS = -I /usr/local/cuda/include -L /usr/local/cuda/lib64/ -lOpenCL

# path to Clang source directory (repository root)
CLANG_INCLUDE = $(PWD)/../llvm/clang/lib/Headers

# path to the SkePU sources
SKEPU_INCLUDE = $(PWD)/../skepu-headers/src

CXX = g++-14
SEQ_CXX = clang++
#DBGR = lldb --

CLUSTER_CXX      = mpic++
CLUSTER_BACKENDS = -openmp
CLUSTER_CXXFLAGS = -std=c++11 -fopenmp $$(pkg-config --cflags starpu-1.3 starpumpi-1.3) -I ./  -DSKEPU_DEBUG=3  -g -O3
CLUSTER_LIBS     = $$(pkg-config --libs starpu-1.3 starpumpi-1.3)


OPENCL_FLAGS = -framework OpenCL

#CLANGTOOL_USER_FLAGS = -I /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/ -I /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/  -I /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/ -I /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1/ -Wno-macro-redefined

#CLANGTOOL_USER_FLAGS  = -I /usr/local/include
#CLANGTOOL_USER_FLAGS += -I /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1/
#CLANGTOOL_USER_FLAGS += -I /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/15.0.0/include/
#CLANGTOOL_USER_FLAGS += -I /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/
#CLANGTOOL_USER_FLAGS += -I /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include
#CLANGTOOL_USER_FLAGS += -I /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks

# echo | g++-14 -E -x c++ -Wp,-v -
CLANGTOOL_USER_FLAGS  = -I /usr/local/Cellar/gcc/14.2.0_1/bin/../lib/gcc/current/gcc/x86_64-apple-darwin23/14/../../../../../../include/c++/14
CLANGTOOL_USER_FLAGS += -I /usr/local/Cellar/gcc/14.2.0_1/bin/../lib/gcc/current/gcc/x86_64-apple-darwin23/14/../../../../../../include/c++/14/x86_64-apple-darwin23
#CLANGTOOL_USER_FLAGS += -I /usr/local/Cellar/gcc/13.2.0/bin/../lib/gcc/current/gcc/x86_64-apple-darwin23/13/../../../../../../include/c++/13/backward
#CLANGTOOL_USER_FLAGS += -I /usr/local/Cellar/gcc/13.2.0/bin/../lib/gcc/current/gcc/x86_64-apple-darwin23/13/include
#CLANGTOOL_USER_FLAGS += -I /usr/local/Cellar/gcc/13.2.0/bin/../lib/gcc/current/gcc/x86_64-apple-darwin23/13/include-fixed
CLANGTOOL_USER_FLAGS += -I /Library/Developer/CommandLineTools/SDKs/MacOSX14.sdk/usr/include
#CLANGTOOL_USER_FLAGS += -I /Library/Developer/CommandLineTools/SDKs/MacOSX14.sdk/System/Library/Frameworks

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

PAR_DIR     = parallel
SEQ_DIR     = sequential
CLUSTER_DIR = cluster
RUN_DIR     = run

# All SkePU example programs (used for 'make all-[parallel|sequential|cluster]' and 'make clean').

PROGS=
PROGS += cellular_automaton
PROGS += cma
PROGS += conjugate_gradient
PROGS += coulombic
PROGS += dotproduct
PROGS += heat_diffusion # no red-black update mode in CUDA
PROGS += horner
#PROGS += image_manip
PROGS += mandelbrot
PROGS += maxmin
PROGS += miller_rabin
PROGS += mmmult
PROGS += montecarlo
PROGS += mvmult
PROGS += mvmult_row
PROGS += mmmult_row_col
PROGS += nbody
PROGS += nbody_mappairs
PROGS += ppmcc
PROGS += psnr
PROGS += riemann_sum
PROGS += taylor


CLUSTER_PROGS =
#CLUSTER_PROGS += cellular_automaton
CLUSTER_PROGS += cma
#CLUSTER_PROGS += conjugate_gradient
CLUSTER_PROGS += coulombic
CLUSTER_PROGS += dotproduct
#CLUSTER_PROGS += heat_diffusion # no red-black update mode
CLUSTER_PROGS += mandelbrot
CLUSTER_PROGS += maxmin
#CLUSTER_PROGS += miller_rabin # no skepu random
CLUSTER_PROGS += mmmult
#CLUSTER_PROGS += montecarlo # no skepu random
CLUSTER_PROGS += mvmult
CLUSTER_PROGS += mvmult_row
#CLUSTER_PROGS += mmmult_row_col # no matcol
CLUSTER_PROGS += nbody
CLUSTER_PROGS += nbody_mappairs
CLUSTER_PROGS += ppmcc
CLUSTER_PROGS += psnr
CLUSTER_PROGS += riemann_sum
CLUSTER_PROGS += taylor


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
SKEPU_FLAGS += -- #$(SKEPU_OPTIONS) -Wno-expansion-to-defined
SKEPU_FLAGS += $(CLANGTOOL_USER_FLAGS)
SKEPU_FLAGS += -I $(CLANG_INCLUDE)
SKEPU_FLAGS += -I $(SKEPU_INCLUDE)

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
TARGET_FLAGS = $(SKEPU_OPTIONS) -I $(SKEPU_INCLUDE) -I ./  $(NVCCPASS) -Wno-attributes $(BACKEND_FLAGS)

# Flags for non-preprocessed compilation.
SEQ_FLAGS    = $(SKEPU_OPTIONS) -I $(SKEPU_INCLUDE) -Wno-attributes # -fno-rtti -fno-exceptions

# ---------------------------------------------------------------------------- #
# Make recipies begin here.

# Builds all test programs, sequential implementation, witout using the precompiler.
all-sequential: $(TESTS)

# Builds all test programs, parallel implementations, using the precompiler.
all-parallel: $(PTESTS)

all-cluster: $(CLUSTER_TESTS)

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

all: all-sequential all-parallel all-cluster

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
	$(DBGR) $(SEQ_CXX) $(SEQ_FLAGS) $< -o $@

# Builds a single sequential test program.
$(SEQ_DIR)/image_manip : image_manip.cpp $(SEQ_DIR)
	$(DBGR) $(SEQ_CXX) $(SEQ_FLAGS) lodepng.cpp $< -o $@

$(SEQ_DIR)/sde : sde.cpp $(SEQ_DIR)
	$(DBGR) $(SEQ_CXX) $(SEQ_FLAGS) lodepng.cpp $< -o $@

# Builds a single sequential test program.
$(SEQ_DIR)/fusion_eval : fusion_eval.cpp $(SEQ_DIR)
	$(DBGR) $(SEQ_CXX) $(SEQ_FLAGS) lodepng.cpp $< -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/%: %.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name $*_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS) -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/image_manip: image_manip.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name image_manip_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS)  -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/demo: demo.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name demo_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS)  -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/fusion_eval: fusion_eval.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name fusion_eval_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS)  -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/perlin: perlin.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name perlin_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS)  -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/uniformnoise: uniformnoise.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name uniformnoise_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS)  -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/multi: multi1.cpp multi2.cpp $(PAR_DIR)
	$(DBGR) $(SKEPU) $(BACKENDS) -name multi1_precompiled multi1.cpp  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(SKEPU) $(BACKENDS) -name multi2_precompiled multi2.cpp  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) -fopenmp $(PAR_DIR)/multi1_precompiled.$(FILETYPE) $(PAR_DIR)/multi2_precompiled.$(FILETYPE)  $(TARGET_FLAGS)  -o $@

$(CLUSTER_DIR)/%.o: %.cpp $(CLUSTER_DIR)
	@echo "Precompiling $<"
	$(DBGR) $(SKEPU) $(CLUSTER_BACKENDS) -dir=$(CLUSTER_DIR) -name $*_precompiled $< $(SKEPU_FLAGS)
	@echo "Compiling $<"
	$(CLUSTER_CXX) -o $@ $(CLUSTER_DIR)/$*_precompiled.cpp $(CLUSTER_CXXFLAGS) -c -I$(SKEPU_INCLUDE) -DSKEPU_STARPU_MPI
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
