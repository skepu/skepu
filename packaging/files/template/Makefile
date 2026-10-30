# A Makefile for SkePU programs: list them, include skepu.mk (one directory up here); `make` builds them into bin/.
# Copy this directory to start a program. skepu.mk documents the settings (BACKENDS, TRACE, ...).
PROGS = hello

# Arguments of make run/hello (the last is usually the backend: CPU, OpenMP, OpenCL or CUDA)
lookup.hello = OpenMP

include ../skepu.mk

all: $(PTESTS)

clean: clean-skepu
.PHONY: all clean
