# SkePU for a package: how a program is precompiled with skepu-tool and built. The Makefile of a course, lab or tutorial lists its programs and includes this:
#
#     PROGS = program1 program2
#     include skepu.mk       (include ../skepu.mk, from a subdirectory of the package)
#
# and has `all: $(PTESTS)`. `make run/<program>` builds a program and runs it (see below). Everything here is relative to the directory of this file, so the package can be unpacked anywhere.
#
# Settings (in local.mk, which is read first, or on the command line: make BACKENDS="-openmp"):
#   BACKENDS            -openmp -cuda -opencl (the default); take out what the computer does not have
#   CXX                 the compiler (g++; nvcc when CUDA is a backend)
#   OPT                 optimization (-O3)
#   TRACE=1             the programs write a trace that SkeVU shows (skepu_trace.json, or $SKEPU_TRACE_FILE; trace_<program>.json by `make run/<program>`);
#                       TRACE_SNAPSHOTS=1 also the snapshots (in snapshots/)
#   COMMON_SKEPU_SOURCES   other source files of the programs that use SkePU (they are precompiled too)
#   COMMON_SOURCES         other source files of the programs (plain C++)
#   USER_INCLUDES       more include directories of the programs, for the precompiler and the compiler (-I extras)
#   lookup.<program>    the arguments of `make run/<program>` (set in the Makefile: lookup.median = data/small.png output/median.png 3 OpenMP)
#   CUDA_HOME           where CUDA is (/usr/local/cuda), for the OpenCL headers that come with it

SKEPU_ROOT := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

-include $(SKEPU_ROOT)/local.mk

BACKENDS ?= -openmp -cuda -opencl
OPT      ?= -O3
OUT_DIR  ?= bin
TRACE    ?= 0
TRACE_SNAPSHOTS ?= 0
SNAPSHOT_DIR ?= snapshots
CUDA_HOME ?= /usr/local/cuda
SEQ_CXX  ?= g++

SKEPU_TOOL    = $(SKEPU_ROOT)/bin/skepu-tool
SKEPU_INCLUDE = $(SKEPU_ROOT)/include
CLANG_HEADERS = $(SKEPU_ROOT)/lib/SkePU/clang-headers

# skepu-tool does not find the system headers of a Mac by itself: they are those of the active toolchain.
# Clang's headers go between the C++ library headers and the SDK headers (the SDK's stddef.h relies on Clang's being found first).
ifeq ($(shell uname -s),Darwin)
SKEPU_SDK          := $(shell xcrun --show-sdk-path 2>/dev/null)
SKEPU_DEVDIR       := $(shell xcode-select -p 2>/dev/null)
ifneq ($(SKEPU_SDK),)
SKEPU_TOOL_USER_FLAGS ?= -I /usr/local/include \
	-I $(SKEPU_SDK)/usr/include/c++/v1 -I $(CLANG_HEADERS) \
	-I $(SKEPU_SDK)/usr/include -I $(SKEPU_DEVDIR)/usr/include
OPENCL_LIBS ?= -framework OpenCL
endif
endif
OPENCL_LIBS ?= -lOpenCL

# The precompiler
SKEPU_FLAGS = $(BACKENDS) $(PRECOMPILER_OPTIONS) -- -std=c++17 -Wno-expansion-to-defined $(SKEPU_TOOL_USER_FLAGS) -I $(CLANG_HEADERS) -I $(SKEPU_INCLUDE) $(USER_INCLUDES)

# The compiler of the precompiled programs: nvcc, if CUDA is a backend (it then passes what is for the host compiler on with -Xcompiler)
BACKEND_FLAGS =
ifneq (,$(findstring cuda,$(BACKENDS)))
CXX := nvcc
FILETYPE = cu
NVCCPASS = -Xcompiler
BACKEND_FLAGS += -Xcudafe "--diag_suppress=declared_but_not_referenced --diag_suppress=set_but_not_used"
else
CXX ?= g++
FILETYPE = cpp
endif
ifneq (,$(findstring openmp,$(BACKENDS)))
BACKEND_FLAGS += $(NVCCPASS) -fopenmp
endif
ifneq (,$(findstring opencl,$(BACKENDS)))
BACKEND_FLAGS += -DCL_TARGET_OPENCL_VERSION=210 -Wno-deprecated-declarations $(OPENCL_LIBS) $(if $(wildcard $(CUDA_HOME)/include),-I $(CUDA_HOME)/include)
endif
ifeq ($(TRACE),1)
BACKEND_FLAGS += -DSKEPU_TRACING
ifeq ($(TRACE_SNAPSHOTS),1)
BACKEND_FLAGS += -DSKEPU_TRACING_SNAPSHOTS -DSKEPU_TRACE_SNAPSHOT_DIR=$(SNAPSHOT_DIR)
endif
endif
TARGET_FLAGS = -g $(OPT) -std=c++17 -I $(SKEPU_INCLUDE) -I . $(USER_INCLUDES) $(NVCCPASS) -Wno-attributes $(BACKEND_FLAGS)

PTESTS = $(addprefix $(OUT_DIR)/,$(PROGS))
COMMON_PRECOMPILED = $(patsubst %.cpp,$(OUT_DIR)/%_precompiled.$(FILETYPE),$(COMMON_SKEPU_SOURCES))

.DEFAULT_GOAL = all
.SECONDARY:

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# A source file that uses SkePU is precompiled: skepu-tool writes the code for the backends
$(OUT_DIR)/%_precompiled.$(FILETYPE): %.cpp $(SKEPU_TOOL) $(MAKEFILE_LIST) | $(OUT_DIR)
	$(SKEPU_TOOL) -name $*_precompiled $< -dir $(OUT_DIR) $(SKEPU_FLAGS)

$(OUT_DIR)/%: $(OUT_DIR)/%_precompiled.$(FILETYPE) $(COMMON_PRECOMPILED) $(COMMON_SOURCES)
	$(CXX) $< $(COMMON_PRECOMPILED) $(COMMON_SOURCES) -o $@ $(TARGET_FLAGS)

# Runs a program (built first, if it is out of date) from RUN_DIR, the directory of this file, where the data of the programs and what they write are
# kept: `make run/median`. Its arguments are `lookup.median` of the Makefile, or ARGS (make run/median ARGS="data/small.png out.png 2 CPU"); with
# TRACE=1, the trace is RUN_DIR/trace_median.json.
RUN_DIR ?= $(SKEPU_ROOT)
run_args = $(if $(filter undefined,$(origin ARGS)),$(lookup.$(1)),$(ARGS))

run/%: $(OUT_DIR)/%
	@$(if $(filter 1,$(TRACE_SNAPSHOTS)),mkdir -p $(RUN_DIR)/$(SNAPSHOT_DIR))
	cd $(RUN_DIR) && SKEPU_TRACE_FILE=trace_$*.json $(abspath $(OUT_DIR))/$* $(call run_args,$*)

# compile_commands.json, for the language server (clangd): an entry for each source, as the editor should read it
COMPILE_SOURCES = $(sort $(foreach p,$(PROGS),$(p).cpp) $(COMMON_SKEPU_SOURCES) $(COMMON_SOURCES))
json_entry = {"directory": "$(CURDIR)", "file": "$(1)", "command": "$(SEQ_CXX) -std=c++17 -I $(SKEPU_INCLUDE) -I . $(USER_INCLUDES) -Wno-attributes -c $(1)"}

compile-commands: compile_commands.json

compile_commands.json: $(MAKEFILE_LIST)
	@printf '[' > $@
	@sep='\n'; $(foreach f,$(COMPILE_SOURCES),printf "$$sep%s" '$(call json_entry,$(f))' >> $@; sep=',\n';) true
	@printf '\n]\n' >> $@
	@echo "$@ written for $(words $(COMPILE_SOURCES)) sources"

clean-skepu:
	-$(RM) $(OUT_DIR)/*_precompiled.cpp $(OUT_DIR)/*_precompiled.cu $(OUT_DIR)/*_cl_source.inl $(PTESTS)
	-$(RM) -r $(OUT_DIR)/*.dSYM

.PHONY: compile-commands clean-skepu
