#!/usr/bin/env python3
"""Install the visualizer (../visualizer) into the rendered book as <output>/viz/.

The embedded SkeVU panels load the visualizer's own scripts, workers, vendored
libraries and example traces at run time, so the book serves an unmodified copy
of those directories under viz/.  Nothing from the visualizer is kept in the
user-guide sources; the visualizer does not know about the book.

Runs as a Quarto post-render script (QUARTO_PROJECT_DIR and QUARTO_PROJECT_OUTPUT_DIR
are set), or by hand from anywhere:  python3 scripts/install_viz.py [OUTPUT_DIR]
"""
import os
import shutil
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("QUARTO_PROJECT_DIR", Path(__file__).resolve().parent.parent))
OUTPUT = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", PROJECT / "_book"))
if not OUTPUT.is_absolute():
	OUTPUT = PROJECT / OUTPUT
VISUALIZER = PROJECT.parent / "visualizer"

# The simple edition of the book (the profile `simple`) has no panels: no copy of the visualizer, and no WebAssembly programs either (resources: demos/).
if "simple" in os.environ.get("QUARTO_PROFILE", "").split(","):
	shutil.rmtree(OUTPUT / "demos", ignore_errors=True)
	print(f"install_viz: simple edition: no viz/, no demos/ in {OUTPUT}")
	sys.exit(0)

# Only the HTML book has panels; Quarto lists the files of this render when it runs us.
rendered = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES")
if rendered is not None and not any(f.endswith(".html") for f in rendered.split()):
	sys.exit(0)

# What the panels use; everything else in the visualizer (node_modules, tests, the stand-alone app) stays out.
PARTS = ["script", "static", "examples"]
# Large and used only by the stand-alone app's style sheet.
SKIP = shutil.ignore_patterns("cmu-font", ".DS_Store")

if not VISUALIZER.is_dir():
	sys.exit(f"install_viz: {VISUALIZER} not found; the user guide is built from within the SkePU repository")

target = OUTPUT / "viz"
shutil.rmtree(target, ignore_errors=True)
for part in PARTS:
	shutil.copytree(VISUALIZER / part, target / part, ignore=SKIP)
print(f"install_viz: {', '.join(PARTS)} -> {target}")
