#!/usr/bin/env python3
"""Writes the two synthetic SkeVU examples: the source files and the traces that go with them.

    backend_switching/backend_switching.cpp  trace_backend_switching_mixed.json
    fusion_patterns/fusion_patterns.cpp      trace_fusion_patterns_synthetic.json

The programs are illustrative and have not been compiled; the traces are made up by a small model of
what SkePU does (a transfer when a call on the OpenCL backend needs data that only the host has, or the
other way round; the time of a call from its backend and size), with the line numbers of the source
files in this script, so the source view and the trace agree. Run it again after editing this file:

    python3 examples/synthetic_traces.py
"""
import json
import os
import random
from contextlib import contextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = '/Users/august/Developer/skepu-alpha/examples'   # as the other examples, shortened when loaded


class Prog:
    def __init__(self, name, header, seed):
        self.name, self.rng = name, random.Random(seed)
        self.lines, self.events = [], []
        self.t, self.next_id = 100, 1000
        self.regions = []
        self._rec = self._play = None
        self.indent = 0
        self.state = {}                # container id -> dict(label, n, host, dev)
        self.file = f'{SRC_DIR}/{name}.cpp'
        self.src(header)

    # -- source ------------------------------------------------------------
    def src(self, text):
        """Append lines of source; the number of the last. In the second and later passes of loop(),
        the lines were written by the first, and this is the line that the same call of src() got."""
        if self._play is not None:
            return next(self._play)
        for l in text.strip('\n').split('\n'):
            self.lines.append('\t' * self.indent + l if l else l)
        if self._rec is not None:
            self._rec.append(len(self.lines))
        return len(self.lines)

    def loop(self, n):
        """for it in p.loop(3): the body is source once, and three times in the trace."""
        log = []
        for it in range(n):
            self._rec, self._play = (log, None) if it == 0 else (None, iter(log))
            yield it
        self._rec = self._play = None

    def blank(self):
        self.lines.append('')

    # -- events ------------------------------------------------------------
    def id(self):
        self.next_id += 1
        return self.next_id - 1

    def emit(self, e, **kw):
        if self.regions: e['region'] = self.regions[-1]
        e.update(kw)
        self.events.append(e)
        return e

    def gap(self, lo=3, hi=12):
        self.t += self.rng.randint(lo, hi)

    @contextmanager
    def region(self, label, text):
        """SKEPU_TRACE_SCOPE(label) with the line of source `text`."""
        line = self.src(text)
        parent = self.regions[-1] if self.regions else None
        self.gap()
        e = {'type': 'region', 'label': label, 'line': line, 'file': self.file,
             'region_depth': len(self.regions) + 1, 'time': self.t}
        if parent: e['region'] = parent
        self.events.append(e)
        self.regions.append(label)
        yield
        self.regions.pop()

    # -- containers --------------------------------------------------------
    def vec(self, label, n, text, host_data=False):
        """A container, with the line of source that declares it."""
        self.src(text)
        cid = self.id()
        self.gap(1, 6)
        self.emit({'type': 'allocation', 'label': label, 'object_id': cid, 'line': 246, 'time': self.t})
        self.state[cid] = dict(label=label, n=n, host=True, dev=False)
        return cid

    def free(self, *cids):
        for c in cids:
            self.gap(1, 4)
            self.emit({'type': 'deallocation', 'label': self.state[c]['label'], 'object_id': c, 'line': 270, 'time': self.t})

    def _transfer(self, cid, direction, label=None, n=None, start=None, end=None):
        s = self.state.get(cid)
        n = n if n is not None else s['n']
        d = int(40 + n * 4 / 8000 * self.rng.uniform(0.95, 1.05))
        self.gap(1, 4)
        a = self.t if start is None else start
        b = a + d if end is None else end
        self.t = max(self.t, b)
        self.events.append(dict({'type': 'transfer', 'backend': 'OpenCL', 'direction': direction, 'elements': n,
                                 'label': label or s['label'], 'object_id': cid, 'line': 258 if direction == 'host-to-device' else 306,
                                 'start': a, 'end': b, 'time': b}, **({'region': self.regions[-1]} if self.regions else {})))

    def _to_host(self, cid):
        s = self.state[cid]
        if not s['host']:
            self._transfer(cid, 'device-to-host')
            s['host'] = True

    def _to_device(self, cid):
        s = self.state[cid]
        if not s['dev']:
            self._transfer(cid, 'host-to-device')
            s['dev'] = True

    # -- skeletons ---------------------------------------------------------
    def skel(self, label, pattern, text):
        """A skeleton instance, with the line of the end of its definition."""
        return dict(label=label, pattern=pattern, id=self.id(), line=self.src(text))

    def duration(self, pattern, backend, n):
        base = pattern.split()[0]
        k = {'CPU': (5.0, 4), 'OpenMP': (0.9, 25), 'OpenCL': (0.25, 70)}[backend]
        w = {'Map': 1, 'Reduce': 1, 'MapReduce': 1.2, 'Scan': 3, 'MapOverlap': 4, 'MapPool': 2.5, 'MapPairs': 1.5, 'MapPairsReduce': 2}.get(base, 1)
        return int((k[1] + n * k[0] * w / 1000) * self.rng.uniform(0.93, 1.08))

    def call(self, sk, text, *, backend, n=None, elements=None, ins=(), proxy=(), scalars=(), outs=(), prng=None, size=None, params=None):
        """A call of skeleton `sk` (a line of source `text` makes it) on `backend`.
        Transfers happen before it if a container is on the wrong side."""
        self.src(text)
        elements = elements or [n]
        cost_n = size or 1
        if size is None:
            for x in elements: cost_n *= x
        cl = backend == 'OpenCL'
        for c in [*ins, *proxy]:
            (self._to_device if cl else self._to_host)(c)
        for c in scalars:
            self._to_host(c)
        self.gap(2, 8)
        d = self.duration(sk['pattern'], backend, cost_n)
        start, end = self.t, self.t + d
        if cl and 'Reduce' in sk['pattern']:         # the partial results come back to the host
            self._transfer(self.id(), 'device-to-host', 'Internal (reduce-temp)', 1, start + d // 2, end)
        self.t = end
        scalar = sk['pattern'].split()[0] in ('Reduce', 'MapReduce')
        for c in outs:
            s = self.state[c]
            s['host'], s['dev'] = (not cl or scalar), (cl and not scalar)
        e = {'type': 'skeleton_call', 'label': sk['label'], 'pattern': sk['pattern'], 'backend': backend,
             'object_id': sk['id'], 'file': self.file, 'line': sk['line'], 'start': start, 'end': end,
             'elements': elements, 'elwise_inputs': list(ins), 'proxy_inputs': list(proxy), 'scalar_inputs': list(scalars),
             'outputs': list(outs), 'threads': 1 if backend == 'CPU' else 8}
        if prng is not None: e['prng'] = prng
        if params: e['parameters'] = params     # (only what is not the default, as the tracer does)
        self.emit(e)

    # -- the host ----------------------------------------------------------
    def external(self, label, text, reads=(), writes=(), dur=40):
        """Code of the host that reads and writes containers."""
        line = self.src(text)
        for c in reads: self._to_host(c)
        self.gap(2, 8)
        a = self.t
        self.t += dur
        for c in writes:
            self.state[c]['host'], self.state[c]['dev'] = True, False
        self.emit({'type': 'external', 'label': label, 'file': self.file, 'line': line, 'start': a, 'end': self.t,
                   'elwise_inputs': list(reads), 'proxy_inputs': [], 'scalar_inputs': [], 'outputs': list(writes)})

    def copy(self, source, text):
        """skepu::Vector<float> copy(source): a new container with the elements of the source (it has no label yet,
        and no source location: the copy constructor has no place for one)."""
        self.src(text)
        s = self.state[source]
        cid = self.id()
        self.gap(1, 6)
        label = f'Vector #{cid}'
        self.emit({'type': 'allocation', 'label': label, 'object_id': cid, 'line': 246, 'time': self.t})
        self.state[cid] = dict(label=label, n=s['n'], host=True, dev=False)
        start = self.t
        self._to_host(source)
        self.gap(1, 3)
        self.t += int(2 + s['n'] * 0.0005)
        self.events.append(dict({'type': 'copy', 'source': source, 'object_id': cid, 'label': label, 'elements': s['n'],
                                 'start': start, 'end': self.t}, **({'region': self.regions[-1]} if self.regions else {})))
        return cid

    def element(self, cid, index, text, write=False):
        """container[index]: the host gets the container; a write also makes the copy on the device stale."""
        line = self.src(text)
        self._to_host(cid)
        self.gap(1, 4)
        self.emit({'type': 'element_access', 'label': self.state[cid]['label'], 'object_id': self.id(), 'source': cid,
                   'index': index, 'file': self.file, 'line': line, 'time': self.t})
        if write: self.state[cid]['dev'] = False

    def write(self, path):
        with open(os.path.join(HERE, path), 'w') as f:
            json.dump(self.events, f, indent=1, sort_keys=True)
            f.write('\n')
        with open(os.path.join(HERE, self.name.split('/')[0], self.name.split('/')[-1] + '.cpp'), 'w') as f:
            f.write('\n'.join(self.lines) + '\n')


# ═════════════════════════════════════════════════════════════════════════════
#  Backend switching
# ═════════════════════════════════════════════════════════════════════════════

def backend_switching():
    p = Prog('backend_switching', '''
// Synthetic example for SkeVU: one program that uses the sequential, OpenMP and OpenCL backends, and the
// host, on the same containers. A trace of it shows when the data has to cross between host and device.
// (Illustrative: the trace was made by examples/synthetic_traces.py, this program has not been run.)
#include <iostream>
#include <skepu>
#include <skepu-lib/io.hpp>

using skepu::Backend;
using skepu::BackendSpec;

constexpr size_t N = 1 << 20;

int main()
{''', seed=7)
    P = p.src
    CPU, OMP, CL = 'CPU', 'OpenMP', 'OpenCL'
    p.src('''
	auto generate = skepu::Map<0>([](skepu::Index1D i) { return std::sin(i.i * 0.001f); });
	auto scale    = skepu::Map<1>([](float x) { return 2.0f * x; });
	auto offset   = skepu::Map<1>([](float x) { return x + 1.0f; });
	auto square   = skepu::Map<1>([](float x) { return x * x; });
	auto total    = skepu::Reduce([](float a, float b) { return a + b; });''')
    # the five skeletons above get their lines from the statements: one skeleton per line
    base = len(p.lines) - 5
    sk = {name: dict(label=name, pattern=pat, id=p.id(), line=base + i + 1)
          for i, (name, pat) in enumerate([('generate', 'Map'), ('scale', 'Map'), ('offset', 'Map'), ('square', 'Map'), ('total', 'Reduce')])}
    p.blank()

    # 1. Sequential, then OpenMP: the host memory is shared, nothing moves.
    signal = p.vec('Signal', 1 << 20, '\tskepu::Vector<float> signal(N, "Signal");')
    scaled = p.vec('Scaled', 1 << 20, '\tskepu::Vector<float> scaled(N, "Scaled");')
    with p.region('Sequential then OpenMP', '\t{ SKEPU_TRACE_SCOPE("Sequential then OpenMP");'):
        p.call(sk['generate'], '\t\tgenerate.setBackend(BackendSpec(Backend::Type::CPU)); generate(signal);', backend=CPU, n=1 << 20, outs=[signal])
        p.call(sk['scale'],    '\t\tscale.setBackend(BackendSpec(Backend::Type::OpenMP));     scale(scaled, signal);', backend=OMP, n=1 << 20, ins=[signal], outs=[scaled])
        p.src('\t}')
    p.blank()

    # 2. Several calls on the device in a row: one transfer in, none between.
    with p.region('OpenCL back to back', '\t{ SKEPU_TRACE_SCOPE("OpenCL back to back");'):
        stage = p.vec('Stage', 1 << 20, '\t\tskepu::Vector<float> stage(N, "Stage");')
        sumv = p.vec('Sum', 1, '\t\tskepu::Scalar<float> sum("Sum");')
        p.call(sk['offset'], '\t\toffset.setBackend(BackendSpec(Backend::Type::OpenCL)); offset(stage, scaled);', backend=CL, n=1 << 20, ins=[scaled], outs=[stage])
        p.call(sk['square'], '\t\tsquare(stage, stage);', backend=CL, n=1 << 20, ins=[stage], outs=[stage])
        p.call(sk['scale'],  '\t\tscale.setBackend(BackendSpec(Backend::Type::OpenCL)); scale(stage, stage);', backend=CL, n=1 << 20, ins=[stage], outs=[stage])
        p.src('\t\ttotal.setBackend(BackendSpec(Backend::Type::OpenCL));')
        p.call(sk['total'], '\t\ttotal(sum, stage);', backend=CL, n=1 << 20, ins=[stage], outs=[sumv])
        p.external('Print', '\t\tstd::cout << "sum " << sum << "\\n";', reads=[sumv], dur=15)
        p.src('\t}')
    p.blank()

    # 3. Alternating between OpenCL and OpenMP on one container: it crosses at every switch.
    p.src('''
	// Alternating backends: every switch moves the vector to the other side.
	auto step_a = skepu::Map<1>([](float x) { return x * 0.5f + 1.0f; });
	auto step_b = skepu::Map<1>([](float x) { return x - std::floor(x); });''')
    sa = dict(label='step_a', pattern='Map', id=p.id(), line=len(p.lines) - 1)
    sb = dict(label='step_b', pattern='Map', id=p.id(), line=len(p.lines))
    with p.region('Alternating backends', '\t{ SKEPU_TRACE_SCOPE("Alternating backends");'):
        state = p.vec('State', 1 << 20, '\t\tskepu::Vector<float> state(N, "State", 0.5f);')
        p.src('\t\tstep_a.setBackend(BackendSpec(Backend::Type::OpenCL));\n\t\tstep_b.setBackend(BackendSpec(Backend::Type::OpenMP));')
        p.src('\t\tfor (int i = 0; i < 3; ++i)\n\t\t{')
        for _ in p.loop(3):
            with p.region('Iteration', '\t\t\tSKEPU_TRACE_SCOPE("Iteration");'):
                p.call(sa, '\t\t\tstep_a(state, state);', backend=CL, n=1 << 20, ins=[state], outs=[state])
                p.call(sb, '\t\t\tstep_b(state, state);', backend=OMP, n=1 << 20, ins=[state], outs=[state])
        p.src('\t\t}\n\t}')
    p.blank()

    # 4. The same work all on one backend.
    with p.region('One backend', '\t{ SKEPU_TRACE_SCOPE("One backend");'):
        state2 = p.vec('State (OpenCL only)', 1 << 20, '\t\tskepu::Vector<float> state2(N, "State (OpenCL only)", 0.5f);')
        p.src('\t\tstep_b.setBackend(BackendSpec(Backend::Type::OpenCL));\n\t\tfor (int i = 0; i < 3; ++i)\n\t\t{')
        for _ in p.loop(3):
            with p.region('Iteration', '\t\t\tSKEPU_TRACE_SCOPE("Iteration");'):
                p.call(sa, '\t\t\tstep_a(state2, state2);', backend=CL, n=1 << 20, ins=[state2], outs=[state2])
                p.call(sb, '\t\t\tstep_b(state2, state2);', backend=CL, n=1 << 20, ins=[state2], outs=[state2])
        p.src('\t\t}\n\t}')
    p.blank()

    # 5. Code of the host between two calls on the device.
    p.src('''
	auto blur = skepu::Map<1>([](float x) { return 0.9f * x; });''')
    bl = dict(label='blur', pattern='Map', id=p.id(), line=len(p.lines))
    with p.region('Host code between kernels', '\t{ SKEPU_TRACE_SCOPE("Host code between kernels");'):
        img = p.vec('Image', 1 << 20, '\t\tskepu::Vector<float> image(N, "Image", 1.0f);')
        p.src('\t\tblur.setBackend(BackendSpec(Backend::Type::OpenCL));')
        p.call(bl, '\t\tblur(image, image);', backend=CL, n=1 << 20, ins=[img], outs=[img])
        p.external('Host filter', '\t\thost_filter(image);   // plain C++ on the elements, through image[i]', reads=[img], writes=[img], dur=2600)
        p.call(bl, '\t\tblur(image, image);', backend=CL, n=1 << 20, ins=[img], outs=[img])
        p.src('\t}')
    p.blank()

    # 6. One element read, and one element written, of a container that is on the device.
    with p.region('Single elements', '\t{ SKEPU_TRACE_SCOPE("Single elements");'):
        res = p.vec('Result', 1 << 20, '\t\tskepu::Vector<float> result(N, "Result");')
        p.call(sk['generate'], '\t\tgenerate.setBackend(BackendSpec(Backend::Type::OpenCL)); generate(result);', backend=CL, n=1 << 20, outs=[res])
        p.element(res, 0, '\t\tfloat first = result[0];       // reads one element: the whole vector comes back, the device copy stays good')
        p.call(sk['offset'], '\t\toffset(result, result);       // no transfer in', backend=CL, n=1 << 20, ins=[res], outs=[res])
        p.element(res, 0, '\t\tresult[0] = 0.0f;               // writes one element: the whole vector comes back, the device copy is stale', write=True)
        p.call(sk['offset'], '\t\toffset(result, result);       // the whole vector goes out again', backend=CL, n=1 << 20, ins=[res], outs=[res])
        p.src('\t}')
    p.blank()

    # 7. A copy of a container that is on the device: it is made on the host.
    with p.region('Copy of a container', '\t{ SKEPU_TRACE_SCOPE("Copy of a container");'):
        res2 = p.vec('Field', 1 << 20, '\t\tskepu::Vector<float> field(N, "Field");')
        p.call(sk['generate'], '\t\tgenerate(field);', backend=CL, n=1 << 20, outs=[res2])
        backup = p.copy(res2, '\t\tskepu::Vector<float> backup(field);   // the copy is made on the host: field comes back, and backup is on the host')
        p.call(sk['offset'], '\t\toffset(field, field);', backend=CL, n=1 << 20, ins=[res2], outs=[res2])
        p.call(sk['scale'], '\t\tscale(backup, backup);   // and goes to the device when it is used there', backend=CL, n=1 << 20, ins=[backup], outs=[backup])
        p.free(backup, res2)
        p.src('\t}')
    p.blank()

    # 8. The backend chosen by the size of the data.
    p.src('''
	// The backend chosen by the size: sequential for small data, OpenMP for medium, OpenCL for large.
	auto choose = [](size_t n) { return n < (1 << 12) ? Backend::Type::CPU : n < (1 << 18) ? Backend::Type::OpenMP : Backend::Type::OpenCL; };''')
    with p.region('Backend by size', '\t{ SKEPU_TRACE_SCOPE("Backend by size");'):
        for n, be, nm in [(1 << 10, CPU, 'Small'), (1 << 16, OMP, 'Medium'), (1 << 22, CL, 'Large')]:
            c = p.vec(nm, n, f'\t\tskepu::Vector<float> {nm.lower()}({n}, "{nm}", 1.0f);')
            p.src(f'\t\tscale.setBackend(BackendSpec(choose({nm.lower()}.size())));')
            p.call(sk['scale'], f'\t\tscale({nm.lower()}, {nm.lower()});', backend=be, n=n, ins=[c], outs=[c])
        p.src('\t}')
    p.blank()
    p.src('\treturn 0;\n}')
    p.write('backend_switching/trace_backend_switching_mixed.json')


# ═════════════════════════════════════════════════════════════════════════════
#  Fusion patterns
# ═════════════════════════════════════════════════════════════════════════════

def fusion_patterns():
    p = Prog('fusion_patterns', """
// Synthetic example for SkeVU: calls that can be fused (skepu::Map(f) >> skepu::Map(g), a || b, see
// examples/fusion.cpp), and calls that look as if they could but cannot. Every block is one region; turn on
// Advanced > Fusion analysis to see the groups, and the reason why the other calls are not in one.
// (Illustrative: the trace was made by examples/synthetic_traces.py, this program has not been run.)
#include <skepu>
#include <skepu-lib/io.hpp>

int main()
{""", seed=11)
    OMP, CL = 'OpenMP', 'OpenCL'
    count = [0]

    def size():
        count[0] += 1
        return 65536 + 4096 * count[0]       # a size of its own for each block, so that blocks do not fuse with each other

    def vec(var, label, n):
        return p.vec(label, n, f'\t\tskepu::Vector<float> {var}(n, "{label}");')

    @contextmanager
    def block(title, doc, n):
        p.src(doc)
        with p.region(title, f'\t{{ SKEPU_TRACE_SCOPE("{title}");'):
            p.src(f'\t\tsize_t n = {n};')
            yield

    def end(*cs):
        p.free(*cs)
        p.src('\t}')
        p.blank()

    def two(name, pat, body):
        return p.skel(name, pat, f'\t\tauto {name} = skepu::{pat}({body});')

    def close_group():
        p.indent = 0
        p.src('\t}')
        p.blank()
        group.__exit__(None, None, None)

    SQ, ADD = '[](float a) { return a * a; }', '[](float a, float b) { return a + b; }'

    # ── Fusible ──────────────────────────────────────────────────────────────
    group = p.region('Fusible', '\t{ SKEPU_TRACE_SCOPE("Fusible");')
    group.__enter__(); p.indent = 1

    n = size()
    with block('Map, Map, Map', '\t// Three Maps in a row, each reading only the result of the one before.\n'
               '\t//   fused:  auto f = scale >> shift >> clamp;  f(y, x);', n):
        x, t1, t2, y = vec('x', 'X', n), vec('t1', 'T1', n), vec('t2', 'T2', n), vec('y', 'Y', n)
        a = two('scale', 'Map', '[](float a) { return 2.0f * a; }')
        b = two('shift', 'Map', '[](float a) { return a + 1.0f; }')
        c = two('clamp', 'Map', '[](float a) { return fminf(a, 10.0f); }')
        p.call(a, '\t\tscale(t1, x);', backend=OMP, n=n, ins=[x], outs=[t1])
        p.call(b, '\t\tshift(t2, t1);', backend=OMP, n=n, ins=[t1], outs=[t2])
        p.call(c, '\t\tclamp(y, t2);', backend=OMP, n=n, ins=[t2], outs=[y])
        end(t2, t1, y, x)

    n = size()
    with block('Map into Reduce', '\t// A Map and a Reduce of its result: a MapReduce, the squares are never stored.\n'
               '\t//   fused:  auto sum_of_squares = skepu::Map(square) >> skepu::Reduce(add);', n):
        x, t, s = vec('x', 'X', n), vec('t', 'Squares', n), p.vec('Sum', 1, '\t\tskepu::Scalar<float> sum("Sum");')
        a = two('square_sum', 'Map', SQ)
        b = two('add_all', 'Reduce', ADD)
        p.call(a, '\t\tsquare_sum(t, x);', backend=OMP, n=n, ins=[x], outs=[t])
        p.call(b, '\t\tadd_all(sum, t);', backend=OMP, n=n, ins=[t], outs=[s])
        end(t, x, s)

    n = size()
    with block('Map, Map, Map side by side', '\t// Three Maps over the same elements that do not depend on each other: one launch, X read once.\n'
               '\t//   fused:  auto f = skepu::Map(square) || skepu::Map(negate) || skepu::Map(half);', n):
        x, u, v, w = vec('x', 'X', n), vec('u', 'Squares', n), vec('v', 'Negated', n), vec('w', 'Halves', n)
        a = two('sq', 'Map', SQ)
        b = two('neg', 'Map', '[](float a) { return -a; }')
        c = two('halve', 'Map', '[](float a) { return 0.5f * a; }')
        p.call(a, '\t\tsq(u, x);', backend=OMP, n=n, ins=[x], outs=[u])
        p.call(b, '\t\tneg(v, x);', backend=OMP, n=n, ins=[x], outs=[v])
        p.call(c, '\t\thalve(w, x);', backend=OMP, n=n, ins=[x], outs=[w])
        end(w, v, u, x)

    n = size()
    with block('Two Maps into one', '\t// Two Maps feed a third: a tree, which fuses into one call (each of U and V is read by one call only).\n'
               '\t//   fused:  auto f = (skepu::Map(square) || skepu::Map(negate)) >> skepu::Map(add);', n):
        x, z, u, v, w = vec('x', 'X', n), vec('z', 'Z', n), vec('u', 'U', n), vec('v', 'V', n), vec('w', 'W', n)
        a = two('square_u', 'Map', SQ)
        b = two('negate_v', 'Map', '[](float a) { return -a; }')
        c = two('add_uv', 'Map', ADD)
        p.call(a, '\t\tsquare_u(u, x);', backend=OMP, n=n, ins=[x], outs=[u])
        p.call(b, '\t\tnegate_v(v, z);', backend=OMP, n=n, ins=[z], outs=[v])
        p.call(c, '\t\tadd_uv(w, u, v);', backend=OMP, n=n, ins=[u, v], outs=[w])
        end(w, v, u, z, x)

    n = size()
    with block('Reduce and Reduce', '\t// Two Reduces over the same vector, in one pass.\n'
               '\t//   fused:  auto minmax = skepu::Reduce(max) || skepu::Reduce(min);  auto [hi, lo] = minmax(x, x);', n):
        x, hi, lo = vec('x', 'X', n), p.vec('Max', 1, '\t\tskepu::Scalar<float> hi("Max");'), p.vec('Min', 1, '\t\tskepu::Scalar<float> lo("Min");')
        a = two('max_of', 'Reduce', '[](float a, float b) { return fmaxf(a, b); }')
        b = two('min_of', 'Reduce', '[](float a, float b) { return fminf(a, b); }')
        p.call(a, '\t\tmax_of(hi, x);', backend=OMP, n=n, ins=[x], outs=[hi])
        p.call(b, '\t\tmin_of(lo, x);', backend=OMP, n=n, ins=[x], outs=[lo])
        end(lo, hi, x)

    n = size()
    h = 32 + count[0] * 2
    with block('Map into MapPool', '\t// A Map (ReLU) into a MapPool with windows that do not overlap (stride = window): no element is\n'
               '\t// computed twice, so this is as cheap as a Map into a Map. With overlapping windows it would recompute.', n):
        x = p.vec('X', 8 * h * h * 8, f'\t\tskepu::Tensor4<float> x(8, {h}, {h}, 8, "X");')
        r = p.vec('Activations', 8 * h * h * 8, f'\t\tskepu::Tensor4<float> relu_out(8, {h}, {h}, 8, "Activations");')
        o = p.vec('Pooled', 8 * (h // 2) ** 2 * 8, f'\t\tskepu::Tensor4<float> pooled(8, {h // 2}, {h // 2}, 8, "Pooled");')
        a = two('relu', 'Map', '[](float a) { return fmaxf(a, 0.0f); }')
        b = p.skel('max_pool', 'MapPool 4D', '\t\tauto max_pool = skepu::MapPool(max_pooling_uf);   // pool size and stride set below\n\t\tmax_pool.setPoolSizeAndStride(1, 2, 2, 1);')
        p.call(a, '\t\trelu(relu_out, x);', backend=OMP, elements=[8, h, h, 8], ins=[x], outs=[r])
        p.call(b, '\t\tmax_pool(pooled, relu_out);', backend=OMP, elements=[8, h // 2, h // 2, 8], ins=[r], outs=[o],
               params={'pool': True, 'pool_size': [1, 2, 2, 1], 'stride': [1, 2, 2, 1]})
        end(o, r, x)

    n = size()
    side = 200 + count[0] * 4
    with block('Map into MapOverlap (recomputes)', '\t// A Map (brightness) into a 3x3 stencil. Fusing is possible, but the stencil reads each element nine times:\n'
               '\t// the fused call would compute the Map nine times for each element, trading memory traffic for work.', n):
        m = side * side
        img = p.vec('Image', m, f'\t\tskepu::Matrix<float> image({side}, {side}, "Image");')
        br = p.vec('Brightened', m, f'\t\tskepu::Matrix<float> bright({side}, {side}, "Brightened");')
        out = p.vec('Edges', m, f'\t\tskepu::Matrix<float> edges({side}, {side}, "Edges");')
        a = two('brighten', 'Map', '[](float a) { return 1.2f * a + 0.1f; }')
        b = p.skel('laplace', 'MapOverlap 2D', '\t\tauto laplace = skepu::MapOverlap(laplace_uf);\n\t\tlaplace.setOverlap(1, 1);   // 3x3 stencil')
        p.call(a, '\t\tbrighten(bright, image);', backend=OMP, elements=[side, side], ins=[img], outs=[br])
        p.call(b, '\t\tlaplace(edges, bright);', backend=OMP, elements=[side, side], ins=[br], outs=[out], params={'overlap': [1, 1]})
        end(out, br, img)

    n = size()
    hs, vs = 512, 128 + count[0] * 4
    with block('Map into MapPairs (recomputes)', '\t// A Map over one of the vectors of a MapPairs. MapPairs reads each element of it for every element of the other\n'
               '\t// vector, so the fused call computes the Map that many times.', n):
        a_in, a_n = p.vec('Raw A', hs, f'\t\tskepu::Vector<float> raw_a({hs}, "Raw A");'), p.vec('A', hs, f'\t\tskepu::Vector<float> a({hs}, "A");')
        b_in = p.vec('B', vs, f'\t\tskepu::Vector<float> b({vs}, "B");')
        d = p.vec('Distances', hs * vs, f'\t\tskepu::Matrix<float> dist({vs}, {hs}, "Distances");')
        a = two('normalize', 'Map', '[](float a) { return a / 255.0f; }')
        b = p.skel('distance', 'MapPairs', '\t\tauto distance = skepu::MapPairs<1, 1>([](float a, float b) { return fabsf(a - b); });')
        p.call(a, '\t\tnormalize(a, raw_a);', backend=OMP, n=hs, ins=[a_in], outs=[a_n])
        p.call(b, '\t\tdistance(dist, a, b);', backend=OMP, elements=[hs, vs], ins=[a_n, b_in], outs=[d])
        end(d, b_in, a_n, a_in)

    n = size()
    hs, vs = 512, 96 + count[0] * 4
    with block('MapPairs into Map', '\t// The matrix of a MapPairs, scaled element by element: the matrix is never stored.\n'
               '\t//   fused:  auto f = skepu::MapPairs<1, 1>(add) >> skepu::Map(reciprocal);', n):
        u, v = p.vec('U', hs, f'\t\tskepu::Vector<float> u({hs}, "U");'), p.vec('V', vs, f'\t\tskepu::Vector<float> v({vs}, "V");')
        s = p.vec('Sums', hs * vs, f'\t\tskepu::Matrix<float> sums({vs}, {hs}, "Sums");')
        r = p.vec('Reciprocals', hs * vs, f'\t\tskepu::Matrix<float> recip({vs}, {hs}, "Reciprocals");')
        a = p.skel('pair_sum', 'MapPairs', f'\t\tauto pair_sum = skepu::MapPairs<1, 1>({ADD});')
        b = two('reciprocal', 'Map', '[](float a) { return 1.0f / a; }')
        p.call(a, '\t\tpair_sum(sums, u, v);', backend=OMP, elements=[hs, vs], ins=[u, v], outs=[s])
        p.call(b, '\t\treciprocal(recip, sums);', backend=OMP, elements=[vs, hs], ins=[s], outs=[r])
        end(r, s, v, u)

    close_group()
    # ── Not fusible ──────────────────────────────────────────────────────────
    group = p.region('Not fusible', '\t{ SKEPU_TRACE_SCOPE("Not fusible");')
    group.__enter__(); p.indent = 1

    n = size()
    with block('Map into Scan', '\t// A Map into a Scan: there is no MapScan, the Scan cannot take the Map\'s function.', n):
        x, t, y = vec('x', 'X', n), vec('t', 'T', n), vec('y', 'Prefix sums', n)
        a = two('double_x', 'Map', '[](float a) { return 2.0f * a; }')
        b = two('prefix_sum', 'Scan', ADD)
        p.call(a, '\t\tdouble_x(t, x);', backend=OMP, n=n, ins=[x], outs=[t])
        p.call(b, '\t\tprefix_sum(y, t);', backend=OMP, n=n, ins=[t], outs=[y])
        end(y, t, x)

    n = size()
    with block('Two readers', '\t// T is read by two calls: it has to be stored. The two readers are independent of each other,\n'
               '\t// though, and can be fused side by side (g || h).', n):
        x, t, ya, yb = vec('x', 'X', n), vec('t', 'T', n), vec('ya', 'Ya', n), vec('yb', 'Yb', n)
        a = two('plus_one', 'Map', '[](float a) { return a + 1.0f; }')
        b = two('times_two', 'Map', '[](float a) { return 2.0f * a; }')
        c = two('negate_it', 'Map', '[](float a) { return -a; }')
        p.call(a, '\t\tplus_one(t, x);', backend=OMP, n=n, ins=[x], outs=[t])
        p.call(b, '\t\ttimes_two(ya, t);', backend=OMP, n=n, ins=[t], outs=[ya])
        p.call(c, '\t\tnegate_it(yb, t);', backend=OMP, n=n, ins=[t], outs=[yb])
        end(yb, ya, t, x)

    n = size()
    with block('The host looks in between', '\t// The host reads an element of T between the two calls, so T has to exist.', n):
        x, t, y = vec('x', 'X', n), vec('t', 'T', n), vec('y', 'Y', n)
        a = two('add_one', 'Map', '[](float a) { return a + 1.0f; }')
        b = two('scale_two', 'Map', '[](float a) { return 2.0f * a; }')
        p.call(a, '\t\tadd_one(t, x);', backend=OMP, n=n, ins=[x], outs=[t])
        p.element(t, 0, '\t\tfloat first = t[0];')
        p.call(b, '\t\tscale_two(y, t);', backend=OMP, n=n, ins=[t], outs=[y])
        end(y, t, x)

    n = size()
    with block('A copy in between', '\t// The first call writes T, which is copied before it is read: the copy reads T, so T has to exist.', n):
        x, t, y = vec('x', 'X', n), vec('t', 'T', n), vec('y', 'Y', n)
        a = two('add_c', 'Map', '[](float a) { return a + 1.0f; }')
        b = two('double_c', 'Map', '[](float a) { return 2.0f * a; }')
        p.call(a, '\t\tadd_c(t, x);', backend=OMP, n=n, ins=[x], outs=[t])
        t2 = p.copy(t, '\t\tskepu::Vector<float> t_copy(t);')
        p.call(b, '\t\tdouble_c(y, t_copy);', backend=OMP, n=n, ins=[t2], outs=[y])
        end(y, t2, t, x)

    n = size()
    with block('Random access', '\t// The second call reads T at any index (a proxy, skepu::Vec), not element by element:\n'
               '\t// it needs all of T finished before it starts.', n):
        x, w, t, y = vec('x', 'X', n), vec('w', 'Indices', n), vec('t', 'T', n), vec('y', 'Y', n)
        a = two('bump', 'Map', '[](float a) { return a + 1.0f; }')
        b = two('lookup', 'Map', '[](float i, skepu::Vec<float> table) { return table[(size_t)i % table.size]; }')
        p.call(a, '\t\tbump(t, x);', backend=OMP, n=n, ins=[x], outs=[t])
        p.call(b, '\t\tlookup(y, w, t);', backend=OMP, n=n, ins=[w], proxy=[t], outs=[y])
        end(y, t, w, x)

    n = size()
    side = 256
    with block('Other shapes', '\t// The same data, read as a vector by the first call and as a matrix by the second:\n'
               '\t// the calls do not run over the same elements.', n):
        x, t, y = vec('x', 'X', n), vec('t', 'T', n), vec('y', 'Y', n)
        a = two('shift_vec', 'Map', '[](float a) { return a + 1.0f; }')
        b = two('scale_mat', 'Map', '[](float a) { return 2.0f * a; }')
        p.call(a, '\t\tshift_vec(t, x);', backend=OMP, n=n, ins=[x], outs=[t])
        p.call(b, f'\t\tscale_mat(y_matrix, t_matrix);   // the same memory, as a {side} x {n // side} matrix', backend=OMP, elements=[side, n // side], ins=[t], outs=[y])
        end(y, t, x)

    n = size()
    with block('A stride', '\t// The second Map has a stride: it reads every other element, not one to one.', n):
        x, t, y = vec('x', 'X', n), vec('t', 'T', n), vec('y', 'Y', n // 2)
        a = two('step_f', 'Map', '[](float a) { return a + 1.0f; }')
        b = two('step_g', 'Map', '[](float a) { return 2.0f * a; }')
        p.src('\t\tstep_g.setStride(2, 2);')
        p.call(a, '\t\tstep_f(t, x);', backend=OMP, n=n, ins=[x], outs=[t])
        p.call(b, '\t\tstep_g(y, t);', backend=OMP, n=n // 2, ins=[t], outs=[y], params={'strides': [2, 2]})
        end(y, t, x)

    n = size()
    with block('Different backends', '\t// The first call runs on OpenMP, the second on OpenCL: T has to go to the device in between.', n):
        x, t, y = vec('x', 'X', n), vec('t', 'T', n), vec('y', 'Y', n)
        a = two('omp_add', 'Map', '[](float a) { return a + 1.0f; }')
        b = two('ocl_scale', 'Map', '[](float a) { return 2.0f * a; }')
        p.src('\t\tf.setBackend(skepu::BackendSpec(skepu::Backend::Type::OpenMP));\n\t\tg.setBackend(skepu::BackendSpec(skepu::Backend::Type::OpenCL));')
        p.call(a, '\t\tomp_add(t, x);', backend=OMP, n=n, ins=[x], outs=[t])
        p.call(b, '\t\tocl_scale(y, t);', backend=CL, n=n, ins=[t], outs=[y])
        end(y, t, x)

    n = size()
    with block('One random stream', '\t// Both calls draw from the same random stream. In one call, the numbers would come in another order.', n):
        x, t, y = vec('x', 'X', n), vec('t', 'T', n), vec('y', 'Y', n)
        st = p.vec('PRNG stream', 1, '\t\tskepu::Random<> stream;   // shared by the calls below')
        a = two('jitter', 'Map', '[](float a, skepu::Random<1> &r) { return a + r.f(); }')
        b = two('noise', 'Map', '[](float a, skepu::Random<1> &r) { return a * (1.0f + 0.1f * r.f()); }')
        p.call(a, '\t\tjitter(t, x, stream);', backend=OMP, n=n, ins=[x], outs=[t], prng=st)
        p.call(b, '\t\tnoise(y, t, stream);', backend=OMP, n=n, ins=[t], outs=[y], prng=st)
        end(y, t, x, st)

    n = size()
    with block('A call in between', '\t// Between f and combine, prefix overwrites x, which f reads, and produces what combine reads.\n'
               '\t// f cannot move down past prefix, combine cannot move up past it.', n):
        x, q, t, y = vec('x', 'X', n), vec('q', 'Q', n), vec('t', 'T', n), vec('y', 'Y', n)
        a = two('increment', 'Map', '[](float a) { return a + 1.0f; }')
        b = two('prefix', 'Scan', ADD)
        c = two('combine', 'Map', ADD)
        p.call(a, '\t\tincrement(t, x);', backend=OMP, n=n, ins=[x], outs=[t])
        p.call(b, '\t\tprefix(x, q);        // overwrites x', backend=OMP, n=n, ins=[q], outs=[x])
        p.call(c, '\t\tcombine(y, t, x);', backend=OMP, n=n, ins=[t, x], outs=[y])
        end(y, t, q, x)

    n = size()
    side = 200 + count[0] * 4
    with block('MapOverlap into Map', '\t// Only a Map can be fused into the call that reads its output; the stencil\'s result is not a stream\n'
               '\t// of independent elements (and would have to be computed with its halo).', n):
        m = side * side
        img, t, y = (p.vec(nm, m, f'\t\tskepu::Matrix<float> {v}({side}, {side}, "{nm}");') for v, nm in [('image', 'Image'), ('blurred', 'Blurred'), ('mask', 'Mask')])
        a = p.skel('blur', 'MapOverlap 2D', '\t\tauto blur = skepu::MapOverlap(blur_uf);\n\t\tblur.setOverlap(1, 1);   // 3x3 stencil')
        b = two('threshold', 'Map', '[](float a) { return a > 0.5f ? 1.0f : 0.0f; }')
        p.call(a, '\t\tblur(blurred, image);', backend=OMP, elements=[side, side], ins=[img], outs=[t], params={'overlap': [1, 1]})
        p.call(b, '\t\tthreshold(mask, blurred);', backend=OMP, elements=[side, side], ins=[t], outs=[y])
        end(y, t, img)

    n = size()
    with block('A Reduce before a Map', '\t// The Map needs the whole sum before its first element: a Reduce is a barrier.', n):
        x, y = vec('x', 'X', n), vec('y', 'Normalized', n)
        s = p.vec('Sum', 1, '\t\tskepu::Scalar<float> sum("Sum");')
        a = two('total_of', 'Reduce', ADD)
        b = p.skel('normalize_by', 'Map', '\t\tauto normalize_by = skepu::Map<1>([](float a, float total) { return a / total; });')
        p.call(a, '\t\ttotal_of(sum, x);', backend=OMP, n=n, ins=[x], outs=[s])
        p.call(b, '\t\tnormalize_by(y, x, sum);', backend=OMP, n=n, ins=[x], scalars=[s], outs=[y])
        end(y, s, x)

    n = size()
    with block('Maps of different sizes', '\t// Three independent Maps; only the two over the same number of elements can share a launch.', n):
        x, w = vec('x', 'X', n), p.vec('W', n // 2 + 7, f'\t\tskepu::Vector<float> w(n / 2 + 7, "W");')
        a_, b_, c_ = vec('a', 'A', n), vec('b', 'B', n), p.vec('C', n // 2 + 7, f'\t\tskepu::Vector<float> c(n / 2 + 7, "C");')
        f1 = two('inc', 'Map', '[](float a) { return a + 1.0f; }')
        f2 = two('dec', 'Map', '[](float a) { return a - 1.0f; }')
        f3 = two('twice', 'Map', '[](float a) { return 2.0f * a; }')
        p.call(f1, '\t\tinc(a, x);', backend=OMP, n=n, ins=[x], outs=[a_])
        p.call(f2, '\t\tdec(b, x);', backend=OMP, n=n, ins=[x], outs=[b_])
        p.call(f3, '\t\ttwice(c, w);', backend=OMP, n=n // 2 + 7, ins=[w], outs=[c_])
        end(c_, b_, a_, w, x)

    close_group()
    p.src('\treturn 0;\n}')
    p.write('fusion_patterns/trace_fusion_patterns_synthetic.json')


if __name__ == '__main__':
    backend_switching()
    fusion_patterns()
