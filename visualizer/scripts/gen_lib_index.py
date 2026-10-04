#!/usr/bin/env python3
"""Writes the manifest of the bundled SkePU library headers to stdout.

usage: gen_lib_index.py DIR > DIR/index.json

DIR is the directory the headers were copied to (static/skepu-lib). The manifest lists the header
files (.hpp, .h) in DIR and in DIR/dnn, so that the visualizer knows which library sources it can
fetch without probing for 404s:

    {"root": ["blas.hpp", ...], "dnn": ["conv2d.hpp", ...]}

Called by `make skepu-lib`.
"""
import json
import os
import sys


def headers(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory)
                  if f.endswith((".hpp", ".h")) and os.path.isfile(os.path.join(directory, f)))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    json.dump({"root": headers(root), "dnn": headers(os.path.join(root, "dnn"))}, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
