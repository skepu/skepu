#!/usr/bin/env python3
"""Compares a log of a SkePU-DNN example run with its golden log.

Lines must match exactly except for numbers, which are compared with a relative tolerance
(and a small absolute one for values near zero). With --strict, lines must be identical.

usage: compare.py GOLDEN ACTUAL [--tol REL] [--strict]
Exit status: 0 if the logs match, 1 otherwise.
"""
import argparse
import re
import sys

NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|nan|inf)")


def split(line):
    """Splits a line into (text parts, numbers)."""
    return NUMBER.split(line), [float(x) for x in NUMBER.findall(line)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("actual")
    ap.add_argument("--tol", type=float, default=1e-3, help="relative tolerance for numbers")
    ap.add_argument("--abs-tol", type=float, default=1e-6, help="absolute tolerance for numbers")
    ap.add_argument("--strict", action="store_true", help="require identical lines")
    args = ap.parse_args()

    golden = open(args.golden).read().splitlines()
    actual = open(args.actual).read().splitlines()
    problems = []

    if len(golden) != len(actual):
        problems.append(f"line count differs: golden {len(golden)}, actual {len(actual)}")

    for n, (g, a) in enumerate(zip(golden, actual), 1):
        if g == a:
            continue
        if args.strict:
            problems.append(f"line {n}: expected '{g}', got '{a}'")
            continue
        gtext, gnum = split(g)
        atext, anum = split(a)
        if gtext != atext or len(gnum) != len(anum):
            problems.append(f"line {n}: expected '{g}', got '{a}'")
            continue
        for x, y in zip(gnum, anum):
            if abs(x - y) > args.abs_tol + args.tol * max(abs(x), abs(y)):
                problems.append(f"line {n}: expected '{g}', got '{a}' ({x} vs {y})")
                break

    if problems:
        for p in problems[:10]:
            print("  " + p)
        if len(problems) > 10:
            print(f"  ... and {len(problems) - 10} more differences")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
