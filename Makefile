.DEFAULT_GOAL := all
PKG_DIR=skepu

build:
	@mkdir -p build

build/src: build
	rsync -a llvm/ build/src
	rsync -a clang build/src/tools

skepu-tool-src: build/src
	rsync -ad clang_precompiler/ build/src/tools/clang/tools/skepu-tool

cmake: skepu-tool-src
	@cd build && cmake -G "Unix Makefiles" src -DCMAKE_BUILD_TYPE=Release;

skepu-tool: cmake skepu-tool-src
	$(MAKE) -C build skepu-tool

all: skepu-tool

skepu-pkg-dir:
	install -D build/bin/skepu-tool $(PKG_DIR)/bin/skepu-tool
	cp -R clang/lib/Headers $(PKG_DIR)/
	cd $(PKG_DIR) && mv Headers clang_headers
	cp -R include $(PKG_DIR)
	install -D PKG_README.in $(PKG_DIR)/README

skepu.tgz: skepu-pkg-dir
	tar cpzf skepu.tgz $(PKG_DIR)

pkg: skepu.tgz

clean-skepu-pkg-dir:
	rm -Rf $(PKG_DIR)

clean-pkg:
	rm -f skepu.tgz

clean-dist: clean-skepu-pkg-dir clean-pkg

clean-build:
	rm -rf build

clean: clean-build clean-dist
