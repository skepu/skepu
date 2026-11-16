# Builds the listings of the guide that are marked to run (scripts/build_listings.py stages them here as <hash>.cpp): `make wasm/<hash>.js`.
PROGS =
COMMON_SOURCES =

include $(dir $(lastword $(MAKEFILE_LIST)))../../examples/Makefile

# the tools of the build, as the build system has them: scripts/build_listings.py checks that they are there
print-tools:
	@echo "$(SKEPU_TOOL)"; echo "$(SKEPU_INCLUDE)"; echo "$(CLANG_HEADERS)"; echo "$(EM_CXX)"
