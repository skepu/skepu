# A Makefile for programs that use SkePU: list them, include skepu.mk (here it is one directory up), and `make` builds them into bin/.
# Copy this directory to start a program of your own. skepu.mk documents the settings (BACKENDS, TRACE, ...).
PROGS = hello

# The arguments of `make run/hello` (the last one is usually the backend: CPU, OpenMP, OpenCL or CUDA, as the program decides)
lookup.hello = OpenMP

include ../skepu.mk

all: $(PTESTS)

clean: clean-skepu
.PHONY: all clean
