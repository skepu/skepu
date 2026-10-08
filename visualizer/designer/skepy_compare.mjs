#!/usr/bin/env node
// skepy_compare.mjs — do the programs of a design, in C++ (SkePU) and in Python (SkePY), build, run and print the same?
//   node designer/skepy_compare.mjs [design.json ...] [--count N] [--size small|medium|large] [--arity small|large] [--shape deep|balanced|wide]
//                                   [--seed PREFIX] [--keep DIR] [--python python3.14]
// For each design file given, or N random designs (settings as Generate's, seeds PREFIX1 .. PREFIXN), it writes both programs with function bodies that
// compute something (the same expression in both languages) and an I/O block that prints the last update of all the data, builds both with the
// sequential header-only SkePU (no precompiler: SkePY's own `shedskin` and `c++`), runs them, and compares what they print (skepy_compare.py).
// Needs the SkePY repository (shedskin/, ignored by git) and Python 3.14. Exit code 1 if a pair differs or does not build.
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { lower, defaultSkeleton } from './model.js';
import { emitCpp, emitPython } from './codegen.js';
import { randomDesign } from './random.js';

const here = dirname(fileURLToPath(import.meta.url));
const option = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback; };
const flags = new Set(['--count', '--size', '--arity', '--shape', '--seed', '--keep', '--python']);
const files = process.argv.slice(2).filter((a, i, all) => !a.startsWith('--') && !flags.has(all[i - 1]));
const python = option('python', 'python3.14');

/** The design with an I/O block that prints the last update of every container, and bodies for the user functions that compute something. */
function prepare(design) {
    const d = JSON.parse(JSON.stringify(design));
    const first = lower(d);
    const printed = first.program.containers.filter(c => c.kind !== 'PRNG').map(c => c.nodes[c.nodes.length - 1]);
    d.skeletons.push({ ...defaultSkeleton('IO', 'sfinal', 'final'), label: 'final' });
    d.positions.sfinal = { x: 0, y: 1e6 };
    for (const id of printed) d.edges.push({ id: `efinal_${id}`, from: id, to: 'sfinal', role: 'read' });
    const W = [0.5, 0.25, 0.125, 0.0625];
    for (const f of lower(d).program.functions) {
        if (f.host || f.suffix === '_reduce') continue;
        const s = d.skeletons.find(x => x.id === f.call.skeleton.id);
        const cpp = [], py = [];
        f.params.forEach((p, k) => {
            const w = W[k % W.length];
            if (p.role === 'elwise' || p.role === 'uniform') { cpp.push(`${w}f * ${p.name}`); py.push(`${w} * ${p.name}`); }
            else if (p.role === 'region') {
                const dims = +p.type.match(/Region(\d)D/)[1], z = Array(dims).fill(0), at = i => z.map((v, j) => j === 0 ? i : v).join(', ');
                cpp.push(`0.5f * r(${at(0)}) + 0.25f * r(${at(-1)}) + 0.25f * r(${at(1)})`);
                py.push(`0.5 * r[${at(0)}] + 0.25 * r[${at(-1)}] + 0.25 * r[${at(1)}]`);
            }
            else if (p.role === 'index') { const field = p.type === 'Index2D' ? 'row' : 'i'; cpp.push(`0.001f * (float)index.${field}`); py.push(`0.001 * float(index.${field})`); }
            else if (p.role === 'proxy') {
                const rank = { Vector: 1, Matrix: 2, Tensor3: 3, Tensor4: 4 }[p.container.kind], zeros = Array(rank).fill(0).join(', ');
                cpp.push(`${w}f * ${p.name}(${zeros})`); py.push(`${w} * ${p.name}[${zeros}]`);
            }
            else if (p.role === 'random') { cpp.push('0.5f * random.getNormalized()'); py.push('0.5 * random.get_normalized()'); }
        });
        if (!cpp.length) { cpp.push('1.0f'); py.push('1.0'); }
        s.bodies = { ...(s.bodies ?? {}), [f.suffix]: { cpp: `return ${cpp.join(' + ')};`, py: `return ${py.join(' + ')}` } };
    }
    return d;
}

const designs = files.length ? files.map(f => [f.replace(/^.*\//, '').replace(/\.json$/, '').replace(/\W/g, '_'), JSON.parse(readFileSync(f, 'utf8'))])
    : Array.from({ length: +option('count', 5) }, (_, i) => {
        const seed = `${option('seed', 'skepy')}${i + 1}`;
        return [`r_${seed}`, randomDesign({ size: option('size', 'small'), arity: option('arity', 'small'), shape: option('shape', 'balanced'), seed }).design];
    });
const dir = option('keep', null) ?? mkdtempSync(join(tmpdir(), 'skepy-compare-'));
let failed = 0;
for (const [name, design] of designs) {
    const result = lower(prepare(design));
    if (!result.ok) { console.log(`${name}: the design has problems: ${result.errors.map(e => e.msg).join('; ')}`); failed++; continue; }
    const work = join(dir, name);
    mkdirSync(work, { recursive: true });
    writeFileSync(join(work, `${name}_ref.cpp`), emitCpp(result.program).text);
    writeFileSync(join(work, `${name}.py`), emitPython(result.program).text);
    const r = spawnSync(python, [join(here, 'skepy_compare.py'), work, name, ...(JSON.stringify(design).includes('"Tensor') ? ['--numbers'] : [])], { encoding: 'utf8', timeout: 900000 });
    process.stdout.write(r.stdout + (r.status ? r.stderr.slice(-500) : ''));
    if (r.status !== 0) failed++;
}
console.log(`${designs.length - failed} of ${designs.length} the same${failed ? `; the programs are in ${dir}` : ''}`);
process.exit(failed ? 1 : 0);
