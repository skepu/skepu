# A Makefile for programs that use SkePU: list them, include skepu.mk (here it is one directory up), and `make` builds them into bin/.
# Copy this directory to start a program of your own. skepu.mk documents the settings (BACKENDS, TRACE, ...).
PROGS = hello

include ../skepu.mk

all: $(PTESTS)

clean: clean-skepu
.PHONY: all clean
