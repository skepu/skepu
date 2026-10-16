# The visualizer panels in the book are served from _book/viz/, installed by
# scripts/install_viz.py (a Quarto post-render step; `make viz` runs it by hand).

# `make simple` is the edition for the SkeVU extension: no panels, no WebAssembly, no copy of the visualizer, in _book-simple/ (_quarto-simple.yml).

QUARTO ?= quarto

.PHONY: all html simple pdf demos viz preview clean

all: html

html:
	$(QUARTO) render --to html

simple:
	QUARTO_PROFILE=simple $(QUARTO) render --to html

pdf:
	$(QUARTO) render --to pdf --output-dir _pdf

demos:
	$(MAKE) -C demos all-wasm

viz:
	python3 scripts/install_viz.py

preview:
	$(QUARTO) preview

clean:
	rm -rf _book _book-simple _pdf .quarto
