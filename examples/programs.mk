# The SkePU example programs, their arguments and the settings of the examples. Included by examples/Makefile before make/skepu.mk.
# A Makefile that includes ../Makefile (examples/dnn/Makefile) sets PROGRAMS_TABLE to its own table and builds only its programs.

TRACE ?= 1
RUN_CWD = .
RUN_LOG = 1

# Linked into every program: image reading and writing (lodepng)
COMMON_SOURCES ?= $(EXAMPLES_DIR)lodepng.cpp
USER_INCLUDES  += -I $(EXAMPLES_DIR)

# Programs.txt rows (also read by CMake): name, flags, arguments (the rest). make/skepu.mk filters PROGS by MPI_PROGS in MPI mode.
PROGRAMS_TABLE ?= $(EXAMPLES_DIR)programs.txt
HASH := \#
table_rows := !/^[ \t]*($(HASH)|$$)/
table_names = $(shell awk '$(table_rows) { print $$1 }' $(PROGRAMS_TABLE))
table_flagged = $(shell awk '$(table_rows) && $$2 ~ /(^|,)$(1)(,|$$)/ { print $$1 }' $(PROGRAMS_TABLE))
table_args = $(shell awk -v p=$(1) '$(table_rows) && $$1 == p { $$1 = ""; $$2 = ""; sub(/^ +/, ""); print }' $(PROGRAMS_TABLE))

ifndef PROGS
PROGS := $(table_names)
CPU_ONLY_PROGS := $(call table_flagged,cpu-only)
MPI_PROGS := $(call table_flagged,mpi)
endif

# Arguments of make run/<program>; RUN_BACKEND is the backend they use
RUN_BACKEND = OpenCL
SIZE_BACKEND = 2048
$(foreach program,$(table_names),$(eval lookup.$(program) = $(call table_args,$(program))))
