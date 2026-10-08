#!/usr/bin/env node
// roundtrip.mjs — the design, the program, the trace of its run, and the design again: builds the program of a design with the
// SkePU precompiler, runs it on each backend, and compares the trace with the design (compare.js).
//   node designer/roundtrip.mjs [design.json | --all | --sample] [backend ...] [--lambdas] [--first-use] [--skeletons-first-use] [--whitespace=readable|balanced|compact]     (default: cpu openmp opencl; --lambdas: the user functions are lambdas; --first-use: containers before their first use)
// Needs the precompiler (build/llvm/bin/skepu-tool) and the build setup of examples/Makefile; takes a minute.
import { readFileSync, writeFileSync, rmSync, readdirSync, copyFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { lower, sampleDesign, allSkeletonsDesign } from './model.js';
import { emitCpp } from './codegen.js';
import { compareTrace } from './compare.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '../..');
const examples = join(root, 'examples');
const lambdas = process.argv.includes('--lambdas'), firstUse = process.argv.includes('--first-use'), skeletonsFirstUse = process.argv.includes('--skeletons-first-use');
const [what = '--all', ...given] = process.argv.slice(2).filter(a => !['--lambdas', '--first-use', '--skeletons-first-use'].includes(a) && !a.startsWith('--whitespace='));
const backends = given.length ? given : ['cpu', 'openmp', 'opencl'];
const design = what === '--sample' ? sampleDesign() : what === '--all' ? allSkeletonsDesign() : JSON.parse(readFileSync(what, 'utf8'));

const result = lower(design);
if (!result.ok) { for (const e of result.errors) console.error(`error: ${e.msg}`); process.exit(1); }

const name = 'zz_designer_roundtrip';
writeFileSync(join(examples, `${name}.cpp`), emitCpp(result.program, { userFunctions: lambdas ? 'lambdas' : 'functions', containerPlacement: firstUse ? 'first-use' : 'grouped', skeletonPlacement: skeletonsFirstUse ? 'first-use' : 'grouped', whitespace: (process.argv.find(a => a.startsWith('--whitespace=')) ?? '').slice(13) || 'balanced' }).text);
let failed = false;
try {
    try {
        execFileSync('make', ['-C', examples, `parallel/${name}`, 'BACKENDS=-openmp -opencl'], { stdio: 'pipe', maxBuffer: 1 << 28 });
    } catch (e) {
        console.error(String(e.stderr).split('\n').filter(l => /error|FATAL/.test(l)).slice(0, 10).join('\n') || e.message);
        process.exit(2);
    }
    for (const backend of backends) {
        // (the build has the name of the trace file in it: the same for each run)
        const trace = join(examples, 'trace', `trace_${name}.json`);
        execFileSync(join(examples, 'parallel', name), [backend], { cwd: examples, stdio: 'ignore' });
        if (process.env.SAVE_TRACE && backend === 'openmp') copyFileSync(trace, process.env.SAVE_TRACE);      // (to make the fixture of the tests)
        const problems = compareTrace(result.program, JSON.parse(readFileSync(trace, 'utf8')));
        console.log(`${backend}: ${problems.length ? 'DIFFERENT' : 'the trace has the calls of the design'}`);
        for (const p of problems) console.log(`  ${p}`);
        failed ||= problems.length > 0;
    }
} finally {
    for (const dir of [examples, join(examples, 'parallel')])
        for (const f of readdirSync(dir)) if (f.includes(name)) rmSync(join(dir, f), { force: true, recursive: true });
    rmSync(join(examples, 'trace', `trace_${name}.json`), { force: true });
}
process.exit(failed ? 1 : 0);
