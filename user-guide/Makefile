# The visualizer panels in the book are served from _book/viz/, installed by
# scripts/install_viz.py (a Quarto post-render step; `make viz` runs it by hand).

.PHONY: all html pdf demos viz preview clean

all: html

html:
	quarto render --to html

pdf:
	quarto render --to pdf --output-dir _pdf

demos:
	$(MAKE) -C demos all-wasm

viz:
	python3 scripts/install_viz.py

preview:
	quarto preview

clean:
	rm -rf _book _pdf .quarto
