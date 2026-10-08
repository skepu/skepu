#!/usr/bin/env python3
"""skepy_compare.py DIR NAME: build NAME.py with SkePY and NAME_ref.cpp with SkePU (both sequential, header-only), run both,
and compare what they print (numbers with a relative tolerance; the other words must be equal, unless --numbers is given).
Needs the SkePY repository (shedskin/, a Shed Skin fork) and Python 3.14, as tests/skepu/run_tests.py of SkePY does.
Exit code: 0 the same, 1 different, 2 the C++ does not build or run, 3 the Python does not."""
import pathlib, re, subprocess, sys, types

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'shedskin' / 'tests' / 'skepu'))
import run_tests as rt          # (the build of SkePY's own tests)

args = [a for a in sys.argv[1:] if not a.startswith('--')]
numbers_only = '--numbers' in sys.argv
work = pathlib.Path(args[0]).resolve()
name = args[1]
cfg = rt.Config(types.SimpleNamespace(skepu_prefix=None, cxx=None, cc=None, seq_cxx='c++', skepu_headers=None, timeout=300, rtol=1e-5))

r = subprocess.run(['c++', '-std=c++17', '-O1', '-w', '-I', str(cfg.headers), f'{name}_ref.cpp', '-o', f'{name}_cpp'], cwd=work, capture_output=True, text=True)
if r.returncode:
    print(f'{name}: the C++ does not compile\n' + r.stderr[-1500:]); sys.exit(2)
cpp = subprocess.run([f'./{name}_cpp', 'cpu'], cwd=work, capture_output=True, text=True, timeout=300)
if cpp.returncode:
    print(f'{name}: the C++ exits with {cpp.returncode}\n' + (cpp.stdout + cpp.stderr)[-600:]); sys.exit(2)
exe, err = rt.build_seq(cfg, name, work)
if err:
    print(f'{name}: SkePY: {err}'); sys.exit(3)
py = subprocess.run([str(exe), 'cpu'], cwd=work, capture_output=True, text=True, timeout=300)
if py.returncode:
    print(f'{name}: the Python exits with {py.returncode}\n' + (py.stdout + py.stderr)[-600:]); sys.exit(3)

def words(text):
    # (SkePU prints a tensor with a header and SkePY with another layout: only the numbers of a tensor are compared)
    text = re.sub(r'Tensor[34]: \([0-9 x]+\)', '', text)
    ws = " ".join(l for l in text.splitlines() if not l.startswith(rt.NOISE)).split()
    return [w for w in ws if re.fullmatch(r'[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?', w)] if numbers_only else ws

a, b = words(cpp.stdout), words(py.stdout)
same = len(a) == len(b) and all(rt.same_word(x, y, 1e-5) for x, y in zip(a, b))
print(f'{name}: {"the same" if same else "DIFFERENT"} ({len(a)} words from C++, {len(b)} from Python)')
if not same:
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if not rt.same_word(x, y, 1e-5)), min(len(a), len(b)))
    print(f'  first difference at word {i}: C++ {a[max(0, i - 3):i + 4]}, Python {b[max(0, i - 3):i + 4]}')
    sys.exit(1)
