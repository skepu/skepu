# SkePU build rules: precompile programs with skepu-tool, build, run and trace them. Shared by packages (installed as skepu.mk in the package root), the
# examples (examples/Makefile) and any other Makefile that builds SkePU programs.
#
# A Makefile lists its programs and includes this file:
#
#     PROGS = program1 program2
#     include ../skepu.mk
#
#     make                        build all programs into bin/
#     make bin/program1           build one program
#     make run/program1           build if out of date, then run with lookup.program1 or ARGS
#     make sequential/program1    build without the precompiler (sequential implementation)
#     make wasm/program1.js       build for the browser with Emscripten (all-wasm: all programs)
#     make compile-commands       write compile_commands.json
#     make clean-skepu            delete build products (clean-run: run output; clean-wasm: browser builds)
#
# Paths: SKEPU_TOOL, SKEPU_INCLUDE and CLANG_HEADERS default to the package layout (relative to this file). A Makefile may set them before including this
# file; make/repo.mk does so for a source tree.
#
# Settings: local.mk next to this file (copy make/local.mk.template), or the command line (make BACKENDS="-openmp"). Precedence, lowest first: the
# defaults here, the including Makefile, local.mk, the command line. A program is built again when a setting that its build commands use changes (bin/.settings).
#   BACKENDS            backends to build: -openmp -opencl -cuda (default); -mpi: see MPI
#   CXX                 compiler (g++; nvcc if CUDA is a backend)
#   SEQ_CXX             compiler for sequential builds and compile_commands.json (g++)
#   OPT, SKEPU_OPTIONS  optimization level (-O3); compiler options (-std=c++17 -g $(OPT))
#   TRACE               1: programs write trace/trace_<program>.json under RUN_CWD; 0: no tracing (for example to measure its cost)
#   TRACE_SNAPSHOTS     1: also write container snapshots (snapshots/); requires TRACE=1
#   COMMON_SKEPU_SOURCES   further SkePU sources, precompiled and linked into every program
#   COMMON_SOURCES         further plain C++ sources, linked into every program
#   USER_INCLUDES       further include directories for the precompiler and the compiler (-I dir)
#   sources.<program>   source files of a multi-file program (default: <program>.cpp)
#   lookup.<program>    arguments of make run/<program>; ARGS="..." on the command line overrides them
#   RUN_CWD             directory that programs run from and write traces to (package: the directory of this file; examples: the current directory)
#   RUN_LOG             1: write run output to run/<program>.log instead of the terminal
#   RUN_ENV             environment for runs (OMP_NUM_THREADS=2)
#   CPU_ONLY_PROGS      programs built for the CPU backends only
#   CUDA_HOME           CUDA installation (/usr/local/cuda), the source of the OpenCL headers
#   OPENCL_FLAGS        OpenCL libraries and headers (detected on a Mac)
#   EM_CXX, EM_FLAGS, WASM_FLAGS, WASM_BACKENDS, WASM_DIR
#                       Emscripten compiler and options, extra definitions (tracing to a file in the module's virtual file system), precompiler
#                       backends (none), output directory (wasm)
#   PRECOMPILER_OPTIONS, SKEPU_TOOL_USER_FLAGS, SKEPU_DEBUG_LEVEL
#                       precompiler options, headers for the precompiler (detected on a Mac), SkePU debug level
#   DBGR                prefix that runs the precompiler in a debugger (lldb --)
#
# MPI (SPMD: one process per rank):
#     make MPI=1 run-mpi/dotproduct   build in bin-mpi/ and run on MPI_RANKS processes; MPI=1 is -mpi in BACKENDS and removes -cuda
#   MPI_CXX (mpicxx), MPI_BASE_CXX (compiler behind the wrapper; CXX), MPIRUN (mpirun), MPIRUN_OPTIONS (--oversubscribe), MPI_RANKS (4)
#   MPI_PROGS           programs that support MPI; the others are skipped in MPI mode (default: all)

SKEPU_MK_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
SKEPU_ROOT   ?= $(SKEPU_MK_DIR)

-include $(SKEPU_MK_DIR)/local.mk

BACKENDS ?= -openmp -opencl -cuda
OPT      ?= -O3
SKEPU_OPTIONS ?= -std=c++17 -g $(OPT)
SKEPU_DEBUG_LEVEL ?= 0
CUDA_HOME ?= /usr/local/cuda
ifeq ($(origin CXX),default)
CXX = g++
endif
SEQ_CXX  ?= g++
TRACE    ?= 0
TRACE_SNAPSHOTS ?= 0
SNAPSHOT_DIR ?= snapshots
RUN_CWD  ?= $(SKEPU_ROOT)
RUN_ENV  ?=
BACKEND_FLAGS ?=

SKEPU_TOOL    ?= $(SKEPU_ROOT)/bin/skepu-tool
SKEPU_INCLUDE ?= $(SKEPU_ROOT)/include
CLANG_HEADERS ?= $(SKEPU_ROOT)/lib/SkePU/clang-headers

# On a Mac the precompiler does not find the system headers: use those of the active toolchain (Xcode or the Command Line Tools).
# Clang's headers go between the C++ library headers and the SDK headers (the SDK's stddef.h needs Clang's first).
ifeq ($(shell uname -s),Darwin)
SKEPU_SDK          := $(shell xcrun --show-sdk-path 2>/dev/null)
SKEPU_DEVDIR       := $(shell xcode-select -p 2>/dev/null)
ifneq ($(SKEPU_SDK),)
SKEPU_TOOL_USER_FLAGS ?= -I /usr/local/include \
	-I $(SKEPU_SDK)/usr/include/c++/v1 -I $(CLANG_HEADERS) \
	-I $(SKEPU_SDK)/usr/include -I $(SKEPU_DEVDIR)/usr/include
OPENCL_FLAGS ?= -framework OpenCL
endif
endif
OPENCL_FLAGS ?= -lOpenCL $(if $(wildcard $(CUDA_HOME)/include),-I $(CUDA_HOME)/include) $(if $(wildcard $(CUDA_HOME)/lib64),-L $(CUDA_HOME)/lib64)

# ------------------------------------------------------------------- #
# MPI mode: separate output directories (bin-mpi, run-mpi, trace-mpi) keep MPI and normal builds apart.
# ------------------------------------------------------------------- #

MPI_CXX ?= mpicxx
MPIRUN ?= mpirun
MPIRUN_OPTIONS ?=
MPI_RANKS ?= 4
ifdef MPI
ifeq (,$(filter -mpi,$(BACKENDS)))
BACKENDS := $(filter-out -cuda,$(BACKENDS)) -mpi
endif
endif
MPI_MODE   := $(filter -mpi,$(BACKENDS))
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
ifdef MPI_PROGS
PROGS := $(filter $(MPI_PROGS),$(PROGS))
endif
endif

# ------------------------------------------------------------------- #
# Directories. bin: programs and precompiler output; sequential: programs built without the precompiler. Relative to RUN_CWD: run (logs), trace, snapshots.
# ------------------------------------------------------------------- #

OUT_BASE   ?= bin
RUN_BASE   ?= run
TRACE_BASE ?= trace
OUT_DIR    ?= $(OUT_BASE)$(MPI_SUFFIX)
SEQ_DIR    ?= sequential
RUN_DIR    ?= $(RUN_BASE)$(MPI_SUFFIX)
TRACE_DIR  ?= $(TRACE_BASE)$(MPI_SUFFIX)

WASM_DIR   ?= wasm
WASM_BUILD  = $(OUT_BASE)-wasm

# ------------------------------------------------------------------- #
# Flags.
# ------------------------------------------------------------------- #

SKEPU_FLAGS = $(PRECOMPILER_OPTIONS) -- -std=c++17 -Wno-expansion-to-defined $(SKEPU_TOOL_USER_FLAGS) -I $(CLANG_HEADERS) -I $(SKEPU_INCLUDE) $(USER_INCLUDES)
SKEPU_OPTIONS += -DSKEPU_DEBUG=$(SKEPU_DEBUG_LEVEL)

# Compiler of the precompiled programs: nvcc if CUDA is a backend (host compiler options are passed with -Xcompiler)
ifneq (,$(findstring cuda,$(BACKENDS)))
CXX := nvcc
FILETYPE = cu
NVCCPASS = -Xcompiler
BACKEND_FLAGS += -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used --diag_suppress=2803"
else
FILETYPE = cpp
endif
ifneq (,$(findstring openmp,$(BACKENDS)))
BACKEND_FLAGS += $(NVCCPASS) -fopenmp
endif
ifneq (,$(findstring opencl,$(BACKENDS)))
BACKEND_FLAGS += -DCL_TARGET_OPENCL_VERSION=210 -Wno-deprecated-declarations $(OPENCL_FLAGS)
endif
# Tracing: the build fixes the trace file name (trace/trace_<program>.json, relative to the run directory)
ifeq ($(TRACE),1)
BACKEND_FLAGS += -DSKEPU_TRACING
ifeq ($(TRACE_SNAPSHOTS),1)
BACKEND_FLAGS += -DSKEPU_TRACING_SNAPSHOTS
endif
endif
trace_defs = $(if $(filter 1,$(TRACE)),-DSKEPU_TRACE_FILE="$(TRACE_DIR)/trace_$(1).json" $(if $(filter 1,$(TRACE_SNAPSHOTS)),-DSKEPU_TRACE_SNAPSHOT_DIR="$(SNAPSHOT_DIR)"))

TARGET_FLAGS = $(SKEPU_OPTIONS) -I $(SKEPU_INCLUDE) -I . $(USER_INCLUDES) $(NVCCPASS) -Wno-attributes $(BACKEND_FLAGS)
SEQ_FLAGS    = $(SKEPU_OPTIONS) -I $(SKEPU_INCLUDE) -I . $(USER_INCLUDES) -Wno-attributes

# ------------------------------------------------------------------- #
# Programs and sources.
# ------------------------------------------------------------------- #

srcs = $(or $(sources.$(1)),$(1).cpp)

program_backends = $(if $(filter $(1),$(CPU_ONLY_PROGS)),$(filter-out -opencl -cuda -mpi,$(BACKENDS)),$(BACKENDS))
program_filetype = $(if $(filter $(1),$(CPU_ONLY_PROGS)),cpp,$(FILETYPE))

TESTS     = $(addprefix $(SEQ_DIR)/,$(PROGS))
PTESTS    = $(addprefix $(OUT_DIR)/,$(PROGS))
RUN_TESTS = $(addprefix $(RUN_DIR)/,$(PROGS))
COMMON_PRECOMPILED = $(patsubst %.cpp,$(OUT_DIR)/%_precompiled.$(FILETYPE),$(COMMON_SKEPU_SOURCES))

# Coarse dependencies (no per-program tracking): any change to the headers, the precompiler, a Makefile or the settings below rebuilds all programs.
SKEPU_HEADERS = $(shell find $(SKEPU_INCLUDE)/ -type f ! -name '.*')
SEQ_DEPS      = $(SKEPU_HEADERS) $(MAKEFILE_LIST)
PAR_DEPS      = $(SEQ_DEPS) $(SKEPU_TOOL) $(BUILD_STAMP)

# The settings that the build commands are made of, in a file that is written only when they change: a program is built again after another TRACE, BACKENDS,
# CXX, or flags, which are no files that it depends on.
BUILD_STAMP    = $(OUT_DIR)/.settings
BUILD_SETTINGS = $(CXX)|$(BACKENDS)|$(SKEPU_FLAGS)|$(TARGET_FLAGS)|$(TRACE)|$(TRACE_SNAPSHOTS)|$(TRACE_DIR)|$(SNAPSHOT_DIR)|$(COMMON_SKEPU_SOURCES)|$(COMMON_SOURCES)|$(CPU_ONLY_PROGS)
quote_shell    = '$(subst ','\'',$(1))'

# ------------------------------------------------------------------- #
# Rules.
# ------------------------------------------------------------------- #

.DEFAULT_GOAL = all
.SECONDARY:
.SECONDEXPANSION:

all: all-parallel
all-parallel: $(PTESTS)
all-sequential: $(TESTS)
all-run: $(RUN_TESTS)

.PHONY: all all-parallel all-sequential all-run all-wasm pack-snapshots compile-commands clean-skepu clean-run clean-wasm

RUN_PATHS = $(RUN_CWD)/$(TRACE_DIR) $(if $(RUN_LOG),$(RUN_CWD)/$(RUN_DIR)) $(if $(filter 1,$(TRACE_SNAPSHOTS)),$(RUN_CWD)/$(SNAPSHOT_DIR))
$(OUT_DIR) $(SEQ_DIR) $(WASM_DIR) $(WASM_BUILD) $(RUN_CWD)/$(TRACE_DIR) $(RUN_CWD)/$(RUN_DIR) $(RUN_CWD)/$(SNAPSHOT_DIR):
	@mkdir -p $@

.PHONY: FORCE
$(BUILD_STAMP): FORCE | $(OUT_DIR)
	@printf '%s\n' $(call quote_shell,$(BUILD_SETTINGS)) | cmp -s - $@ || printf '%s\n' $(call quote_shell,$(BUILD_SETTINGS)) > $@

$(OUT_DIR)/%_precompiled.$(FILETYPE): %.cpp $(PAR_DEPS) | $(OUT_DIR)
	$(DBGR) $(SKEPU_TOOL) $(BACKENDS) -name $*_precompiled $< -dir $(OUT_DIR) $(SKEPU_FLAGS)

$(OUT_DIR)/%: $$(call srcs,$$*) $(COMMON_PRECOMPILED) $(COMMON_SOURCES) $(PAR_DEPS) | $(OUT_DIR) $(if $(filter 1,$(TRACE)),$(RUN_CWD)/$(TRACE_DIR))
	set -e; $(foreach s,$(call srcs,$*),$(DBGR) $(SKEPU_TOOL) $(call program_backends,$*) -name $(basename $(notdir $(s)))_precompiled $(s) -dir $(OUT_DIR) $(SKEPU_FLAGS);) true
	$(DBGR) $(CXX) $(foreach s,$(call srcs,$*),$(OUT_DIR)/$(basename $(notdir $(s)))_precompiled.$(call program_filetype,$*)) $(COMMON_PRECOMPILED) $(COMMON_SOURCES) $(TARGET_FLAGS) $(call trace_defs,$*) -o $@

$(SEQ_DIR)/%: $$(call srcs,$$*) $(COMMON_SKEPU_SOURCES) $(COMMON_SOURCES) $(SEQ_DEPS) | $(SEQ_DIR)
	$(DBGR) $(SEQ_CXX) $(SEQ_FLAGS) $(COMMON_SKEPU_SOURCES) $(COMMON_SOURCES) $(call srcs,$*) -o $@

# WebAssembly: precompile for the sequential implementation (no backends), then build with Emscripten into WASM_DIR/<program>.js and .wasm
# (see user-guide/demos). The trace is a file in the module's virtual file system.
EM_CXX        ?= em++
WASM_BACKENDS ?=
WASM_FLAGS    ?= -DSKEPU_TRACING -DSKEPU_TRACE_FILE="trace.json"
EM_FLAGS      ?= -std=c++17 -I $(SKEPU_INCLUDE) -s MODULARIZE=1 -s EXPORT_NAME='createSkePUModule' -s INVOKE_RUN=0 -s EXIT_RUNTIME=1 \
                 -s EXPORTED_RUNTIME_METHODS='["callMain","FS"]' -s ALLOW_MEMORY_GROWTH=1 -O2
all-wasm: $(addprefix $(WASM_DIR)/,$(addsuffix .js,$(PROGS)))

$(WASM_DIR)/%.js: $$(call srcs,$$*) $(COMMON_SOURCES) $(PAR_DEPS) | $(WASM_DIR) $(WASM_BUILD)
	set -e; $(foreach s,$(call srcs,$*),$(DBGR) $(SKEPU_TOOL) $(WASM_BACKENDS) -name $(basename $(notdir $(s)))_precompiled $(s) -dir $(WASM_BUILD) $(SKEPU_FLAGS);) true
	$(EM_CXX) $(foreach s,$(call srcs,$*),$(WASM_BUILD)/$(basename $(notdir $(s)))_precompiled.cpp) $(COMMON_SOURCES) -o $@ $(EM_FLAGS) $(WASM_FLAGS)

RUN_PREFIX = $(RUN_ENV) $(if $(MPI_MODE),$(MPIRUN) $(MPIRUN_OPTIONS) -n $(MPI_RANKS))
run_args   = $(if $(filter undefined,$(origin ARGS)),$(lookup.$(1)),$(ARGS))

$(RUN_DIR)/%: $(OUT_DIR)/% | $$(RUN_PATHS)
	cd $(RUN_CWD) && $(RUN_PREFIX) $(abspath $(OUT_DIR))/$* $(call run_args,$*)$(if $(RUN_LOG), > $(RUN_DIR)/$*.log 2>&1)
	$(if $(filter 1,$(TRACE_SNAPSHOTS)),$(MAKE) pack-snapshots ARCHIVE_TITLE=snapshots_$*)

# Pack the snapshot files of a run into a .zip (the form SkeVU loads) and delete them
ARCHIVE_TITLE ?= snapshots
pack-snapshots:
	@cd $(RUN_CWD)/$(SNAPSHOT_DIR) && \
		if ls *.txt 1>/dev/null 2>&1; then \
		  zip -qj $(abspath $(RUN_CWD))/$(ARCHIVE_TITLE).zip *.txt && \
		  rm *.txt && \
		  echo "$(ARCHIVE_TITLE).zip created from $(SNAPSHOT_DIR)/*.txt (existing files removed)"; \
		else \
		  echo "No snapshot files in $(SNAPSHOT_DIR) (snapshots are taken only with TRACE_SNAPSHOTS=1)."; \
		fi

# compile_commands.json for the language server: one entry per source, with the flags of the sequential build (the precompiled sources are generated).
# Rewritten when the Makefiles change.
COMPILE_SOURCES = $(sort $(foreach p,$(PROGS),$(call srcs,$(p))) $(COMMON_SKEPU_SOURCES) $(COMMON_SOURCES))
json_entry = {"directory": "$(CURDIR)", "file": "$(1)", "command": "$(SEQ_CXX) $(SEQ_FLAGS) -c $(1)"}

compile-commands: compile_commands.json

compile_commands.json: $(MAKEFILE_LIST)
	@printf '[' > $@
	@sep='\n'; $(foreach f,$(COMPILE_SOURCES),printf "$$sep%s" '$(call json_entry,$(f))' >> $@; sep=',\n';) true
	@printf '\n]\n' >> $@
	@echo "$@ written for $(words $(COMPILE_SOURCES)) sources"

clean-skepu:
	-$(RM) $(foreach d,$(OUT_BASE) $(OUT_BASE)-mpi,$(d)/*_precompiled.cpp $(d)/*_precompiled.cu $(d)/*_cl_source.inl $(addprefix $(d)/,$(PROGS)))
	-$(RM) -r $(OUT_BASE)/*.dSYM $(OUT_BASE)-mpi/*.dSYM
	-$(RM) $(TESTS)
	-rmdir $(OUT_BASE) $(OUT_BASE)-mpi $(SEQ_DIR) 2>/dev/null; true

clean-run:
	$(RM) -R $(foreach d,$(RUN_BASE) $(TRACE_BASE),$(RUN_CWD)/$(d) $(RUN_CWD)/$(d)-mpi) $(RUN_CWD)/$(SNAPSHOT_DIR)
	$(RM) $(RUN_CWD)/snapshots_*.zip

clean-wasm:
	$(RM) -R $(WASM_DIR) $(WASM_BUILD)
