# Paths of a built source tree (cmake --build build): the precompiler and Clang's headers in the build, SkePU's headers in the sources. Included before
# skepu.mk by the Makefiles of the repository (examples/Makefile). Packages do not use it; their layout is the default of skepu.mk.
#
# Settings (in make/local.mk, which is read before these paths are used): SKEPU_SRC_PATH and LLVM_BUILD_REL_PATH, or SKEPU_TOOL, SKEPU_INCLUDE and
# CLANG_HEADERS directly.

REPO_MK_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

SKEPU_SRC_PATH ?= $(REPO_MK_DIR)/..

LLVM_BUILD_REL_PATH ?= /build/
LLVM_BUILD_PATH = $(SKEPU_SRC_PATH)/$(LLVM_BUILD_REL_PATH)

SKEPU_INCLUDE ?= $(SKEPU_SRC_PATH)/skepu-headers/src
SKEPU_TOOL    ?= $(LLVM_BUILD_PATH)/llvm/bin/skepu-tool

# Clang headers for the precompiler: the resource directory of the LLVM build (llvm/clang/lib/Headers lacks the generated headers). The only Clang header
# directory passed; the version is that of LLVM in this repository.
CLANG_VERSION_MAJOR = $(shell sed -n 's/.*set(LLVM_VERSION_MAJOR \([0-9]*\)).*/\1/p' $(SKEPU_SRC_PATH)/llvm/cmake/Modules/LLVMVersion.cmake)
CLANG_HEADERS ?= $(LLVM_BUILD_PATH)/llvm/lib/clang/$(CLANG_VERSION_MAJOR)/include
