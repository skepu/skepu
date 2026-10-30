# Demos of the user guide, built for the browser: `make` (all-wasm) or `make wasm/<program>.js`. Add demos to PROGS.
PROGS = map_squaring
COMMON_SOURCES =

include $(dir $(lastword $(MAKEFILE_LIST)))../../examples/Makefile

.DEFAULT_GOAL = all-wasm

clean: clean-wasm
