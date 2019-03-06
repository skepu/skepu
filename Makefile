.DEFAULT_GOAL := all

build:
	@mkdir build 2> /dev/null || true

build/src: build
	rsync -a llvm/ build/src
	rsync -a clang build/src/tools

skepu-tool-src: build/src
	rsync -ad clang_precompiler/ build/src/tools/clang/tools/skepu-tool

clean-all:
	@rm -rf build

clean:
	make -C build clean

cmake: skepu-tool-src
	@cd build && cmake -G "Unix Makefiles" src -DCMAKE_BUILD_TYPE=Release;

skepu-tool: cmake skepu-tool-src
	$(MAKE) -C build skepu-tool

all: skepu-tool
