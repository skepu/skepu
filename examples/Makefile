#######################################################################
#                     ____  _        ____  _   _                      #
#                    / ___|| | _____|  _ \| | | |                     #
#                    \___ \| |/ / _ \ |_) | | | |                     #
#                     ___) |   <  __/  __/| |_| |                     #
#                    |____/|_|\_\___|_|    \___/                      #
#                                                                     #
#                       ~ SkePU main Makefile ~                       #
#######################################################################
# The directory of this Makefile. A program in a subdirectory of the examples (or anywhere else) is built with this
# Makefile by a Makefile of its own that lists its programs and includes this one:
#     PROGS = mnist
#     include ../Makefile

EXAMPLES_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

# The MPI backend (distributed containers over the node-level backends; programs run SPMD-style, one process per rank):
#     make MPI=1 parallel-mpi/dotproduct         builds it for MPI (with the other backends of BACKENDS, but CUDA)
#     make MPI=1 run-mpi/dotproduct              runs it on MPI_RANKS processes (the log is run-mpi/dotproduct.log)
#     make MPI=1 MPI_RANKS=2 all-run             all the programs that the MPI backend supports, on 2 processes
# or put -mpi in BACKENDS in Makefile.in. The MPI compiler, the program that starts the processes and its options are
# set below (MPI_CXX, MPI_BASE_CXX, MPIRUN, MPIRUN_OPTIONS, MPI_RANKS).

.DEFAULT_GOAL=all

# ------------------------------------------------------------------- #
# Default fundamental paths and options.
# ------------------------------------------------------------------- #

# Default path to the SkePU installation
# Override in Makefile.in
SKEPU_SRC_PATH = $(EXAMPLES_DIR)/../

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

# The MPI backend. MPI=1 on the command line adds -mpi to BACKENDS (and takes -cuda out: the MPI backend has no CUDA).
# Override in Makefile.in
MPI_CXX = mpicxx
# The compiler that the MPI compiler wrapper runs (OMPI_CXX for Open MPI, MPICH_CXX for MPICH); the default is CXX
# MPI_BASE_CXX =
# The program that starts the processes of a run, its options (for example --oversubscribe) and the number of processes
MPIRUN = mpirun
MPIRUN_OPTIONS =
MPI_RANKS = 4
# What a run of a program is prefixed with (environment variables for the processes: RUN_ENV = OMP_NUM_THREADS=2)
RUN_ENV =

# Uncomment for launching SkePU precompiler in debugger
# DBGR = lldb --


# ------------------------------------------------------------------- #
# Includes.
# ------------------------------------------------------------------- #

$(EXAMPLES_DIR)Makefile.in:
	cd $(EXAMPLES_DIR) && ./generate_makefile.in.sh

# Overrides default values for various paths and options
include $(EXAMPLES_DIR)Makefile.in

# MPI mode: -mpi among the backends (MPI=1 puts it there). The programs are built with the MPI compiler, in directories of their own
# (parallel-mpi, run-mpi, trace-mpi) so that the builds, logs and traces of the two ways do not replace each other.
ifdef MPI
ifeq (,$(filter -mpi,$(BACKENDS)))
BACKENDS := $(filter-out -cuda,$(BACKENDS)) -mpi
endif
endif
MPI_MODE := $(filter -mpi,$(BACKENDS))
MPI_SUFFIX := $(if $(MPI_MODE),-mpi)

ifdef MPI_MODE
ifneq (,$(filter -cuda,$(BACKENDS)))
$(error The MPI backend cannot be built together with CUDA: remove -cuda from BACKENDS)
endif
ifndef MPI_BASE_CXX
MPI_BASE_CXX := $(CXX)
endif
export OMPI_CXX  := $(MPI_BASE_CXX)
export MPICH_CXX := $(MPI_BASE_CXX)
CXX := $(MPI_CXX)
endif


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

PAR_BASE    = parallel
RUN_BASE    = run
TRACE_BASE  = trace
PAR_DIR     = $(PAR_BASE)$(MPI_SUFFIX)
SEQ_DIR     = sequential
RUN_DIR     = $(RUN_BASE)$(MPI_SUFFIX)
SNAPSHOT_DIR ?= snapshots

# Snapshots of containers (SKEPU_TRACE_SNAPSHOT in the programs, shown as heat maps in SkeVU) are only taken if the programs are built
# with 'make TRACE_SNAPSHOTS=1 ...' (-DSKEPU_TRACING_SNAPSHOTS, in addition to tracing). Each snapshot is a file with the whole
# container, so they can fill the disk and slow the program down.
TRACE_SNAPSHOTS ?= 0
ifeq ($(TRACE_SNAPSHOTS),1)
BACKEND_FLAGS += -DSKEPU_TRACING_SNAPSHOTS
endif
TRACE_DIR    ?= $(TRACE_BASE)$(MPI_SUFFIX)

# All SkePU example programs (unless the Makefile that includes this one has set PROGS; used for 'make all-[parallel|sequential]' and 'make clean').

ifndef PROGS
PROGS += cellular_automaton
PROGS += cma
PROGS += conjugate_gradient
PROGS += coulombic
PROGS += dotproduct
PROGS += edgedetect
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
PROGS += ppmcc
PROGS += psnr
PROGS += riemann_sum
PROGS += taylor

# The programs that the MPI backend supports (see examples/CMake/example-options.cmake): the others are not built in MPI mode
MPI_PROGS = cellular_automaton cma conjugate_gradient coulombic dotproduct horner mandelbrot maxmin miller_rabin \
	mmmult mmmult_row_col montecarlo mvmult mvmult_row nbody nbody_mappairs ode_fisher ode_lorenz ppmcc psnr riemann_sum taylor
ifdef MPI_MODE
PROGS := $(filter $(MPI_PROGS),$(PROGS))
endif
endif

RUN_BACKEND = OpenCL
SIZE_BACKEND = 2048
lookup.cellular_automaton  := 2 256 10 0.1 $(RUN_BACKEND)
lookup.cma                 := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.conjugate_gradient  := 256 $(RUN_BACKEND)
lookup.coulombic           := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.dotproduct          := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.edgedetect          := edge data/goose.png run/result_edgedetect 2 $(RUN_BACKEND)
lookup.heat_diffusion      := 2 64 100 $(RUN_BACKEND) redblack
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
lookup.ode_fisher          := 64 20 $(RUN_BACKEND)
lookup.ode_lorenz          := $(SIZE_BACKEND) 40 $(RUN_BACKEND)
lookup.ppmcc               := $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.psnr                := 128 128 $(RUN_BACKEND)
lookup.riemann_sum         := 0 1 $(SIZE_BACKEND) $(RUN_BACKEND)
lookup.taylor              := 0.1 $(SIZE_BACKEND) $(RUN_BACKEND)


# The programs that have more than one source file list them: sources.<program> := a.cpp b.cpp
# (the default is <program>.cpp). Each of them is precompiled and the results are built into one program.
# (see multifile/Makefile)
srcs = $(or $(sources.$(1)),$(1).cpp)

# Sources that are built into every program (the image reading and writing of the examples).
COMMON_SOURCES ?= $(EXAMPLES_DIR)lodepng.cpp

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
TARGET_FLAGS = $(SKEPU_OPTIONS) -I $(SKEPU_INCLUDE_PATH) -I ./ -I $(EXAMPLES_DIR) $(NVCCPASS) -Wno-attributes $(BACKEND_FLAGS)

# Flags for non-preprocessed compilation.
SEQ_FLAGS    = $(SKEPU_OPTIONS) -I $(SKEPU_INCLUDE_PATH) -I $(EXAMPLES_DIR) -Wno-attributes # -fno-rtti -fno-exceptions


# ---------------------------------------------------------------------------- #
# Make recipes begin here.

# Builds all test programs, sequential implementation, without using the precompiler.
all-sequential: $(TESTS)

# Builds all test programs, parallel implementations, using the precompiler.
all-parallel: $(PTESTS)

all-run: $(RUN_TESTS)

all: $(EXAMPLES_DIR)Makefile.in all-sequential all-parallel

.PHONY: all all-sequential all-parallel all-run pack-snapshots clean compile-commands

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

# A compilation database (compile_commands.json) for editors and language servers (VS Code, Zed, clangd): one entry for each
# example source, with the flags of the sequential build, which is the source as it is written (the precompiled sources of
# the parallel builds are generated files). Written by make itself, no extra tool needed. Regenerated when the flags change.
COMPILE_COMMANDS = compile_commands.json
COMPILE_SOURCES  = $(sort $(foreach p,$(PROGS),$(call srcs,$(p))) $(COMMON_SOURCES))
json_entry = {"directory": "$(CURDIR)", "file": "$(1)", "command": "$(SEQ_CXX) $(SEQ_FLAGS) -c $(1)"}

compile-commands: $(COMPILE_COMMANDS)

$(COMPILE_COMMANDS): $(MAKEFILE_LIST)
	@printf '[' > $@
	@sep='\n'; $(foreach f,$(COMPILE_SOURCES),printf "$$sep%s" '$(call json_entry,$(f))' >> $@; sep=',\n';) true
	@printf '\n]\n' >> $@
	@echo "$@ written for $(words $(COMPILE_SOURCES)) sources"

# (the prerequisites of the rules below name the sources of the program: second expansion)
.SECONDEXPANSION:

# Builds a single sequential test program.
$(SEQ_DIR)/% : $$(call srcs,$$*) $(SEQ_DEPS) | $(SEQ_DIR)
	$(DBGR) $(SEQ_CXX) $(SEQ_FLAGS) $(COMMON_SOURCES) $(call srcs,$*) -o $@

# Precompiles and builds a single parallel test program (the precompiled source of each of its sources is
# $(PAR_DIR)/<source>_precompiled.$(FILETYPE)).
$(PAR_DIR)/%: $$(call srcs,$$*) $(PAR_DEPS) | $(PAR_DIR) $(TRACE_DIR)
	set -e; $(foreach s,$(call srcs,$*),$(DBGR) $(SKEPU_TOOL_PATH) $(BACKENDS) -name $(basename $(notdir $(s)))_precompiled $(s) -dir $(PAR_DIR) $(SKEPU_FLAGS);) true
	$(DBGR) $(CXX) $(foreach s,$(call srcs,$*),$(PAR_DIR)/$(basename $(notdir $(s)))_precompiled.$(FILETYPE)) $(COMMON_SOURCES) $(TARGET_FLAGS) -DSKEPU_TRACE_FILE="$(TRACE_DIR)/trace_$*.json" -DSKEPU_TRACE_SNAPSHOT_DIR="$(SNAPSHOT_DIR)" -o $@

# The start of a run: the processes of an MPI program are started by MPIRUN, one for each rank (SPMD)
RUN_PREFIX = $(RUN_ENV) $(if $(MPI_MODE),$(MPIRUN) $(MPIRUN_OPTIONS) -n $(MPI_RANKS))

# Run an example program while collecting log file
$(RUN_DIR)/% : $(PAR_DIR)/% | $(RUN_DIR) $(SNAPSHOT_DIR) $(TRACE_DIR)
	$(RUN_PREFIX) $< $(lookup.$*) > $@.log 2>&1
	$(MAKE) pack-snapshots ARCHIVE_TITLE=snapshots_$*

ARCHIVE_TITLE ?= snapshots
pack-snapshots:
	@cd $(SNAPSHOT_DIR) && \
		if ls *.txt 1>/dev/null 2>&1; then \
		  zip -qj $(CURDIR)/$(ARCHIVE_TITLE).zip *.txt && \
		  rm *.txt && \
		  echo "$(ARCHIVE_TITLE).zip created from $(SNAPSHOT_DIR)/*.txt (existing files removed)"; \
		else \
		  echo "No snapshot files in $(SNAPSHOT_DIR) (snapshots are taken only with TRACE_SNAPSHOTS=1)."; \
		fi

# Deletes all temporary files (including all precompiled sources), binaries and the
# products of running them (logs, traces, snapshots); all of it can be regenerated.
clean:
	$(RM) -R $(SEQ_DIR)
	$(RM) -R $(PAR_BASE) $(PAR_BASE)-mpi
	$(RM) -R $(RUN_BASE) $(RUN_BASE)-mpi
	$(RM) -R $(TRACE_BASE) $(TRACE_BASE)-mpi $(SNAPSHOT_DIR)
	$(RM) snapshots_*.zip $(COMPILE_COMMANDS)
