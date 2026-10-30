#######################################################################
#                     ____  _        ____  _   _                      #
#                    / ___|| | _____|  _ \| | | |                     #
#                    \___ \| |/ / _ \ |_) | | | |                     #
#                     ___) |   <  __/  __/| |_| |                     #
#                    |____/|_|\_\___|_|    \___/                      #
#                                                                     #
#                       ~ SkePU main Makefile ~                       #
#######################################################################
# The SkePU example programs. Rules and settings: make/skepu.mk. Paths: make/repo.mk. Programs and their arguments: programs.mk.
#
#     make run/dotproduct            build and run (log: run/dotproduct.log; trace: trace/trace_dotproduct.json)
#     make all-run                   run all programs
#     make MPI=1 run-mpi/dotproduct  the same with MPI, on MPI_RANKS processes (programs in MPI_PROGS only)
#
# Settings for this computer (BACKENDS, CXX, ...): make/local.mk, copied from make/local.mk.template.
#
# A Makefile in a subdirectory (or elsewhere) lists its programs and includes this one:
#     PROGS = mnist
#     include ../Makefile

EXAMPLES_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

ifneq ($(wildcard $(EXAMPLES_DIR)Makefile.in),)
$(warning examples/Makefile.in is no longer read: move its settings to make/local.mk (see make/local.mk.template))
endif
include $(EXAMPLES_DIR)programs.mk
include $(EXAMPLES_DIR)../make/repo.mk
include $(EXAMPLES_DIR)../make/skepu.mk

# The examples are built with and without the precompiler
all: all-sequential

clean: clean-skepu clean-run
	$(RM) compile_commands.json
.PHONY: clean
