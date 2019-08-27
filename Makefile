.DEFAULT_GOAL := all
PKG_DIR=skepu

build:
	@mkdir build 2> /dev/null || true

build/src: build
	rsync -a llvm/ build/src
	rsync -a clang build/src/tools

skepu-tool-src: build/src
	rsync -ad clang_precompiler/ build/src/tools/clang/tools/skepu-tool

clean:
	@rm -rf build
	@rm -Rf $(PKG_DIR)
	@rm -f skepu.tgz

cmake: skepu-tool-src
	@cd build && cmake -G "Unix Makefiles" src -DCMAKE_BUILD_TYPE=Release;

skepu-tool: cmake skepu-tool-src
	$(MAKE) -C build skepu-tool

all: skepu-tool

pkg:
	install -D build/bin/skepu-tool $(PKG_DIR)/bin/skepu-tool
	cp -R clang/lib/Headers $(PKG_DIR)/
	cd $(PKG_DIR) && mv Headers clang_headers
	cp -R include $(PKG_DIR)
	install -D PKG_README.in $(PKG_DIR)/README
	tar cpzf skepu.tgz $(PKG_DIR)
