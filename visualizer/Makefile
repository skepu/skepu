# Makefile — visualizer maintenance targets.
#
# Targets:
#   skepu-lib        Copy SkePU library headers into static/skepu-lib/ and
#                    generate static/skepu-lib/index.json.  Requires that
#                    the skepu-headers repository is checked out at the
#                    same level as this one (../skepu-headers/).
#
#   clean-skepu-lib  Remove the generated static/skepu-lib/ directory.
#
#   vsix             Build the VS Code extension as skepu-visualizer-<version>.vsix (npm run package, which builds the bundles and
#                    compiles the extension), with the library headers in it, and the user guide: the simple edition of ../user-guide, which
#                    needs Quarto. Without Quarto, or if the guide cannot be built, a warning is printed and the extension is built without
#                    the guide; GUIDE=0 leaves it out on purpose, QUARTO=/path/to/quarto names Quarto. What the guide is built in is removed
#                    afterwards. Needs npm install first (@vscode/vsce is a dev dependency).
#                    Install it with: codium --install-extension skepu-visualizer-<version>.vsix  (or code --install-extension ...)
#
#   guide            The simple edition of the user guide, in guide/ (what `make vsix` does first; for trying the extension from this directory).
#
#   clean-guide      Remove guide/ and what the guide was built in.

SKEPU_LIB_SRC := ../skepu-headers/src/skepu-lib
SKEPU_LIB_DST := static/skepu-lib

GUIDE_SRC := ../user-guide
GUIDE_DIR := guide
QUARTO    ?= quarto

.PHONY: skepu-lib clean-skepu-lib vsix guide clean-guide

## ── skepu-lib ────────────────────────────────────────────────────────────────
## Copies every .hpp/.h file from $(SKEPU_LIB_SRC) and its dnn/ subdirectory
## into $(SKEPU_LIB_DST), then generates an index.json manifest so the
## visualizer worker knows which files are available without probing 404s.
##
## The destination lives inside the visualizer tree so it is unambiguously a
## copy — not a symlink to live sources.  Add static/skepu-lib/ to .gitignore
## if you do not want to commit the headers alongside the visualizer.

skepu-lib: $(SKEPU_LIB_DST)/index.json

$(SKEPU_LIB_DST)/index.json: $(wildcard $(SKEPU_LIB_SRC)/*.hpp) \
                              $(wildcard $(SKEPU_LIB_SRC)/*.h)   \
                              $(wildcard $(SKEPU_LIB_SRC)/dnn/*.hpp) \
                              $(wildcard $(SKEPU_LIB_SRC)/dnn/*.h)
	@echo "skepu-lib: copying headers from $(SKEPU_LIB_SRC) ..."
	mkdir -p $(SKEPU_LIB_DST)/dnn
	cp $(SKEPU_LIB_SRC)/*.hpp $(SKEPU_LIB_DST)/
	@if ls $(SKEPU_LIB_SRC)/*.h 1>/dev/null 2>&1; then \
	    cp $(SKEPU_LIB_SRC)/*.h $(SKEPU_LIB_DST)/; \
	fi
	cp $(SKEPU_LIB_SRC)/dnn/*.hpp $(SKEPU_LIB_DST)/dnn/
	@if ls $(SKEPU_LIB_SRC)/dnn/*.h 1>/dev/null 2>&1; then \
	    cp $(SKEPU_LIB_SRC)/dnn/*.h $(SKEPU_LIB_DST)/dnn/; \
	fi
	python3 scripts/gen_lib_index.py $(SKEPU_LIB_DST) > $@
	@echo "skepu-lib: done. Index written to $@"

clean-skepu-lib:
	rm -rf $(SKEPU_LIB_DST)
	@echo "skepu-lib: removed $(SKEPU_LIB_DST)"

## ── The extension for VS Code ────────────────────────────────────────────────

# The user guide is the simple edition of ../user-guide (make simple there: the profile `simple`, in _book-simple/), copied to guide/, which the package has
# (.vscodeignore). If Quarto is not there or the build fails there is a warning, and no guide in the package: the extension says so when it is asked for the guide.
guide:
	@rm -rf $(GUIDE_DIR)
	@if ! command -v $(QUARTO) >/dev/null 2>&1; then \
	    echo ""; \
	    echo "WARNING: '$(QUARTO)' (Quarto) was not found, so the user guide cannot be built: the extension will be built WITHOUT the user guide."; \
	    echo "         Install Quarto (https://quarto.org), or name it with QUARTO=/path/to/quarto, and build again."; \
	    echo ""; \
	elif $(MAKE) --no-print-directory -C $(GUIDE_SRC) simple QUARTO=$(QUARTO) && cp -R $(GUIDE_SRC)/_book-simple $(GUIDE_DIR); then \
	    echo "guide: the simple edition of $(GUIDE_SRC) is in $(GUIDE_DIR)/"; \
	else \
	    rm -rf $(GUIDE_DIR); \
	    echo ""; \
	    echo "WARNING: the user guide could not be built (the output of Quarto is above): the extension will be built WITHOUT the user guide."; \
	    echo ""; \
	fi

clean-guide:
	rm -rf $(GUIDE_DIR) $(GUIDE_SRC)/_book-simple

# (the guide is built for the package only: guide/, the book that it is made in and Quarto's cache, if there was none before, are removed after it)
vsix: skepu-lib
	@had_cache=0; test -d $(GUIDE_SRC)/.quarto && had_cache=1; \
	if [ "$(GUIDE)" = 0 ]; then echo "guide: GUIDE=0: the extension is built without the user guide"; rm -rf $(GUIDE_DIR); else $(MAKE) --no-print-directory guide; fi; \
	npm run package; status=$$?; \
	rm -rf $(GUIDE_DIR) $(GUIDE_SRC)/_book-simple; \
	[ $$had_cache = 1 ] || rm -rf $(GUIDE_SRC)/.quarto; \
	exit $$status
