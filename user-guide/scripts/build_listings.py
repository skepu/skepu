#!/usr/bin/env python3
"""Builds the listings marked to run (```{.cpp .run}) as WebAssembly programs; see README.md.

A build that fails is the report that the listing no longer compiles. A quarto pre-render step, or by hand:
  python3 scripts/build_listings.py [--annotate] [--keep-going] [--jobs N]
    --annotate    mark the complete programs that print something, in the sources
    --keep-going  exit with 0 when listings fail (the failures are reported either way)
"""
import argparse
import concurrent.futures as cf
import difflib
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("QUARTO_PROJECT_DIR", Path(__file__).resolve().parent.parent))
STAGE = PROJECT / "listings"
OUT = STAGE / "wasm"

# a fenced code block: the opening line, the text, the closing fence
BLOCK = re.compile(r"^(?P<fence>`{3,})(?P<info>[^\n`]*)\n(?P<text>.*?)^(?P=fence)[ \t]*$", re.S | re.M)


def listing_hash(text):
    """Must be what listings.lua computes: pandoc.sha1 of the text of the code block, which has no final newline and its tabs expanded (to 4)."""
    return hashlib.sha1(text.rstrip("\n").expandtabs(4).encode()).hexdigest()[:10]


def sources():
    files = [PROJECT / "index.qmd"] + sorted((PROJECT / "src").rglob("*.qmd"))
    return [f for f in files if f.is_file()]


def info_classes(info):
    """The classes of the opening line of a block: `cpp`, or {.cpp .run argv="10"}."""
    info = info.strip()
    if not info.startswith("{"):
        return set(info.split()[:1])
    return set(re.findall(r"\.([\w+-]+)", re.sub(r'[\w-]+="[^"]*"', "", info)))


def marked_listings():
    """(file, line, text, expected output) of each listing marked to run; the expected output is that of the block ```{.text .output} after it, or None."""
    found = []
    for f in sources():
        text = f.read_text()
        blocks = list(BLOCK.finditer(text))
        for k, m in enumerate(blocks):
            classes = info_classes(m.group("info"))
            if "run" in classes and "cpp" in classes:
                expected = None
                if k + 1 < len(blocks) and "output" in info_classes(blocks[k + 1].group("info")) and not text[m.end():blocks[k + 1].start()].strip():
                    expected = blocks[k + 1].group("text")
                found.append((f, text[:m.start()].count("\n") + 2, m.group("text"), expected))
    return found


def annotate():
    """Mark the complete programs that print: the ones where running is of interest."""
    count = 0
    for f in sources():
        text = f.read_text()

        def mark(m):
            nonlocal count
            info = m.group("info").strip()
            body = m.group("text")
            if info not in ("cpp", "{.cpp}") or "int main" not in body or not re.search(r"cout|printf", body):
                return m.group(0)
            count += 1
            return f"{m.group('fence')}{{.cpp .run}}\n{body}{m.group('fence')}"
        new = BLOCK.sub(mark, text)
        if new != text:
            f.write_text(new)
    print(f"build_listings: {count} listings marked")


def missing_tools():
    """What the build needs and is not there: the SkePU build system has to be configured (precompiler, headers) and Emscripten installed."""
    p = subprocess.run(["make", "-s", "-C", str(STAGE), "print-tools"], capture_output=True, text=True)
    if p.returncode or len(p.stdout.split("\n")) < 4:
        return ["the SkePU build system (make failed in listings/)"]
    # make prints paths relative to its directory
    tool, include, clang_headers = (STAGE / x for x in p.stdout.split("\n")[:3])
    em = p.stdout.split("\n")[3]
    missing = []
    if not os.access(tool, os.X_OK):
        missing.append(f"the SkePU precompiler ({tool})")
    if not Path(include).is_dir():
        missing.append(f"the SkePU headers ({include})")
    if not Path(clang_headers).is_dir():
        missing.append(f"the Clang headers ({clang_headers})")
    if shutil.which(em) is None:
        missing.append(f"Emscripten ({em})")
    return missing


def build(job):
    f, line, text, _ = job
    h = listing_hash(text)
    (STAGE / f"{h}.cpp").write_text(text)
    p = subprocess.run(["make", "-C", str(STAGE), f"wasm/{h}.js"], capture_output=True, text=True)
    return job, h, p.returncode, p.stdout + p.stderr


def first_errors(output, n=3):
    lines = [l for l in output.splitlines() if re.search(r"error|Error", l)]
    return lines[:n] or output.splitlines()[-n:]


def normalize(text):
    return [l.rstrip() for l in text.strip().splitlines()]


def compare_outputs(jobs):
    """Warn where the output of a built listing is not the block that is shown for it (updating the block is up to the author)."""
    checked = [j for j in jobs if j[3] is not None]
    if not checked:
        return
    if shutil.which("node") is None:
        print("WARNING build_listings: node not found: the outputs of the listings are not compared with the blocks that show them", file=sys.stderr)
        return
    differ = 0
    for f, line, text, expected in checked:
        js = OUT / f"{listing_hash(text)}.js"
        if not js.is_file():
            continue
        p = subprocess.run(["node", str(PROJECT / "scripts" / "run_wasm.js"), str(js)], capture_output=True, text=True, encoding="utf-8")
        where = f"{f.relative_to(PROJECT)}:{line}"
        if p.returncode:
            print(f"WARNING {where}: the program failed: {p.stderr.strip()[:200]}", file=sys.stderr)
            differ += 1
        elif normalize(p.stdout) != normalize(expected):
            differ += 1
            print(f"WARNING {where}: the output differs from the block that shows it", file=sys.stderr)
            for l in list(difflib.unified_diff(normalize(expected), normalize(p.stdout), "shown", "actual", lineterm="", n=0))[2:10]:
                print(f"    {l}", file=sys.stderr)
    print(f"build_listings: outputs compared: {len(checked)} listings, {differ} differ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    if args.annotate:
        annotate()
        return 0
    # runners are for the full HTML book: not the simple edition (profile simple), not print (make pdf sets SKEPU_LISTINGS=off)
    if "simple" in os.environ.get("QUARTO_PROFILE", "").split(",") or os.environ.get("SKEPU_LISTINGS") == "off":
        print("build_listings: skipped: no runners in this edition")
        return 0

    jobs = marked_listings()
    wanted = {listing_hash(j[2]) for j in jobs}
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in list(OUT.iterdir()) + list(STAGE.glob("*.cpp")):
        if stale.stem not in wanted:
            stale.unlink()
    todo = [j for j in jobs if not (OUT / f"{listing_hash(j[2])}.js").is_file()]
    todo = list({listing_hash(j[2]): j for j in todo}.values())
    if not todo:
        print(f"build_listings: {len(jobs)} listings, all built")
        compare_outputs(jobs)
        return 0
    missing = missing_tools()
    if missing:
        print(f"WARNING build_listings: {'; '.join(missing)} not available: {len(todo)} listings are not built or checked, and the book has no runners for them", file=sys.stderr)
        return 0

    failed = 0
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for (f, line, _, _), h, rc, output in ex.map(build, todo):
            where = f"{f.relative_to(PROJECT)}:{line}"
            if rc:
                failed += 1
                print(f"FAIL {where}", file=sys.stderr)
                for l in first_errors(output):
                    print(f"    {l}", file=sys.stderr)
    print(f"build_listings: {len(jobs)} listings, {len(todo)} built, {failed} failed")
    compare_outputs([j for j in jobs if (OUT / f"{listing_hash(j[2])}.js").is_file()])
    return 0 if args.keep_going or not failed else 1


if __name__ == "__main__":
    sys.exit(main())
