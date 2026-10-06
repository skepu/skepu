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
SKEPU_SRC_PATH = $(dir $(lastword $(MAKEFILE_LIST)))/../

# Default path to LLVM build directory (relative to SkePU installation path)
# Override in Makefile.in
LLVM_BUILD_REL_PATH = /build/

# Default backend set for SkePU precompiler
# Override in Makefile.in
BACKENDS = -openmp -opencl -cuda

# Default debug level of the SkePU backends (-DSKEPU_DEBUG)
# Override in Makefile.in
SKEPU_DEBUG_LEVEL = 0

# Default values for SkePU backend compiler options
# Override in Makefile.in
SKEPU_OPTIONS = -std=c++17 -O3 -g

# Default options for the SkePU precompiler
# Override in Makefile.in
PRECOMPILER_OPTIONS = #--no-preserve-lines #--verbose

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

# System include paths for the SkePU precompiler. skepu-tool is a libclang tool and does not find the
# system headers of the machine by itself on macOS, so they are derived from the active toolchain
# (Xcode or the Command Line Tools). Set SKEPU_TOOL_USER_FLAGS in Makefile.in to use other paths.
ifeq ($(shell uname -s),Darwin)
SKEPU_SDK          := $(shell xcrun --show-sdk-path 2>/dev/null)
SKEPU_CLANG_RESDIR := $(shell xcrun clang++ -print-resource-dir 2>/dev/null)
SKEPU_DEVDIR       := $(shell xcode-select -p 2>/dev/null)
ifneq ($(and $(SKEPU_SDK),$(SKEPU_CLANG_RESDIR)),)
SKEPU_TOOL_USER_FLAGS ?= -I /usr/local/include \
	-I $(SKEPU_SDK)/usr/include/c++/v1 -I $(SKEPU_CLANG_RESDIR)/include \
	-I $(SKEPU_SDK)/usr/include -I $(SKEPU_DEVDIR)/usr/include
endif
endif

# Uncomment for launching SkePU precompiler in debugger
# DBGR = lldb --


# ------------------------------------------------------------------- #
# Includes.
# ------------------------------------------------------------------- #

Makefile.in:
	./generate_makefile.in.sh

# Overrides default values for various paths and options
include $(dir $(lastword $(MAKEFILE_LIST)))Makefile.in


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

# What every program depends on besides its own source: the SkePU headers, the precompiler binary and
# the build configuration. Deliberately coarse (no per-program dependency tracking): a change to any
# of them rebuilds all programs, which is always correct and, for the examples, cheap enough.
SKEPU_HEADERS = $(shell find $(SKEPU_INCLUDE_PATH) -type f ! -name '.*')
SEQ_DEPS      = $(SKEPU_HEADERS) $(MAKEFILE_LIST)
PAR_DEPS      = $(SEQ_DEPS) $(SKEPU_TOOL_PATH)

# Add SkePU-specific debug level to options set
SKEPU_OPTIONS += -DSKEPU_DEBUG=$(SKEPU_DEBUG_LEVEL)


# ---------------------------------------------------------------------------- #
# Test program names and target directories.

PAR_DIR     = parallel
SEQ_DIR     = sequential
RUN_DIR     = run
SNAPSHOT_DIR ?= snapshots
TRACE_DIR    ?= trace

# All SkePU example programs (used for 'make all-[parallel|sequential]' and 'make clean').

PROGS=
PROGS += cellular_automaton
PROGS += cma
PROGS += conjugate_gradient
PROGS += coulombic
PROGS += dotproduct
PROGS += edgedetect
PROGS += heat_diffusion # no red-black update mode in CUDA
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
PROGS += ppmcc
PROGS += psnr
PROGS += riemann_sum
PROGS += taylor

RUN_BACKEND = OpenCL
SIZE_BACKEND = 2048
lookup.cellular_automaton  := 2 256 10 0.1 $(RUN_BACKEND)
lookup.cma                 := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.conjugate_gradient  := 256 $(RUN_BACKEND)
lookup.coulombic           := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.dotproduct          := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.edgedetect          := edge data/goose.png run/result_edgedetect 2 $(RUN_BACKEND)
lookup.heat_diffusion      := 2 64 10 $(RUN_BACKEND)
lookup.horner              := 3 1 2 3 4 $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.image_manip         := $(RUN_BACKEND)
lookup.mandelbrot          := 256 256 $(RUN_BACKEND)
lookup.maxmin              := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.median              := data/goose.png run/result_median 2 $(RUN_BACKEND)
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
lookup.riemann_sum         := 0 1 $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.taylor              := 0.1 $(SIZE_BACKEND) $(RUN_BACKEND)


TESTS          = $(addprefix $(SEQ_DIR)/,      $(PROGS))
PTESTS         = $(addprefix $(PAR_DIR)/,      $(PROGS))
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
# Make recipes begin here.

# Builds all test programs, sequential implementation, without using the precompiler.
all-sequential: $(TESTS)

# Builds all test programs, parallel implementations, using the precompiler.
all-parallel: $(PTESTS)

all-run: $(RUN_TESTS)

all: Makefile.in all-sequential all-parallel

.PHONY: all all-sequential all-parallel all-run pack-snapshots clean

$(PAR_DIR):
	@mkdir -p $(PAR_DIR)

$(SEQ_DIR):
	@mkdir -p $(SEQ_DIR)

$(RUN_DIR):
	@mkdir -p $(RUN_DIR)

$(SNAPSHOT_DIR):
	@mkdir -p $(SNAPSHOT_DIR)

# The trace files of the programs (SkeVU input); created when first needed.
$(TRACE_DIR):
	@mkdir -p $(TRACE_DIR)

# Builds a single sequential test program.
$(SEQ_DIR)/% : %.cpp $(SEQ_DEPS) | $(SEQ_DIR)
	$(DBGR) $(SEQ_CXX) $(SEQ_FLAGS) lodepng.cpp $< -o $@

# Precompiles and builds a single parallel test program.
$(PAR_DIR)/%: %.cpp $(PAR_DEPS) | $(PAR_DIR) $(TRACE_DIR)
	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name $*_precompiled $<  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) $@_precompiled.$(FILETYPE) lodepng.cpp $(TARGET_FLAGS) -DSKEPU_TRACE_FILE="$(TRACE_DIR)/trace_$*.json" -DSKEPU_TRACE_SNAPSHOT_DIR="$(SNAPSHOT_DIR)" -o $@

# Precompiles and builds the two-file test program.
$(PAR_DIR)/multifile: multifile_1.cpp multifile_2.cpp $(PAR_DEPS) | $(PAR_DIR)
	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name multifile_1_precompiled multifile_1.cpp  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name multifile_2_precompiled multifile_2.cpp  -dir $(PAR_DIR) $(SKEPU_FLAGS)
	$(DBGR) $(CXX) -fopenmp $(PAR_DIR)/multifile_1_precompiled.$(FILETYPE) $(PAR_DIR)/multifile_2_precompiled.$(FILETYPE)  $(TARGET_FLAGS)  -o $@

# Run an example program while collecting log file
$(RUN_DIR)/% : $(PAR_DIR)/% | $(RUN_DIR) $(SNAPSHOT_DIR) $(TRACE_DIR)
	$< $(lookup.$*) > $@.log 2>&1
	$(MAKE) pack-snapshots ARCHIVE_TITLE=snapshots_$*

ARCHIVE_TITLE ?= snapshots
pack-snapshots:
	@cd $(SNAPSHOT_DIR) && \
		if ls *.txt 1>/dev/null 2>&1; then \
		  zip -qj $(CURDIR)/$(ARCHIVE_TITLE).zip *.txt && \
		  rm *.txt && \
		  echo "$(ARCHIVE_TITLE).zip created from $(SNAPSHOT_DIR)/*.txt (existing files removed)"; \
		else \
		  echo "No snapshot files in $(SNAPSHOT_DIR)."; \
		fi

# Deletes all temporary files (including all precompiled sources), binaries and the
# products of running them (logs, traces, snapshots); all of it can be regenerated.
clean:
	$(RM) -R $(SEQ_DIR)
	$(RM) -R $(PAR_DIR)
	$(RM) -R $(RUN_DIR)
	$(RM) -R $(TRACE_DIR) $(SNAPSHOT_DIR)
	$(RM) snapshots_*.zip
