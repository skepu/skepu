# Makefile — visualizer maintenance targets.
#
# Targets:
#   skepu-lib        Copy SkePU library headers into static/skepu-lib/ and
#                    generate static/skepu-lib/index.json.  Requires that
#                    the skepu-headers repository is checked out at the
#                    same level as this one (../skepu-headers/).
#
#   clean-skepu-lib  Remove the generated static/skepu-lib/ directory.

SKEPU_LIB_SRC := ../skepu-headers/src/skepu-lib
SKEPU_LIB_DST := static/skepu-lib

.PHONY: skepu-lib clean-skepu-lib

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
