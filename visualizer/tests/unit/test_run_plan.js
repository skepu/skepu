// test_run_plan.js — planning a build and run through the Makefile next to the source (src/runPlan.ts), and the options form (src/runOptionsHtml.ts). The rules
// are the same in the examples and in a package (make/skepu.mk): bin/<program> builds, run/<program> runs, the trace is trace/trace_<program>.json.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { isSkepuMakefile, planRun, valuesOf, printTargets, usesBackend } from '../../src/runPlan.ts';
import { runOptionsHtml } from '../../src/runOptionsHtml.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const examples = readFileSync(join(ROOT, 'examples/Makefile'), 'utf8');
const dnn = readFileSync(join(ROOT, 'examples/dnn/Makefile'), 'utf8');
const hello = readFileSync(join(ROOT, 'packaging/files/template/Makefile'), 'utf8');
const options = { overrides: {}, values: {}, make: 'make' };
const run = (extra, makefile = hello, source = '/p/programs/hello.cpp') => planRun({ ...options, makefile, source, ...extra }, '/p/programs');

describe('the Makefiles', () => {
    it('are recognized by including skepu.mk or ../Makefile', () => {
        expect(isSkepuMakefile(examples)).toBe(true);
        expect(isSkepuMakefile(dnn)).toBe(true);
        expect(isSkepuMakefile(hello)).toBe(true);
        expect(isSkepuMakefile('all:\n\techo\n')).toBe(false);
    });
});

describe('what make says', () => {
    it('is asked with print rules and read in order', () => {
        expect(printTargets('median')).toEqual(['print-RUN_BACKEND', 'print-lookup.median', 'print-RUN_CWD', 'print-TRACE_DIR']);
        expect(valuesOf(['OpenCL', '2048 OpenCL', '.', 'trace'])).toEqual({ backend: 'OpenCL', arguments: '2048 OpenCL', runCwd: '.', traceDir: 'trace' });
        expect(valuesOf(['', undefined, '..', 'trace'])).toEqual({ backend: undefined, arguments: undefined, runCwd: '..', traceDir: 'trace' });
    });

    it('tells whether a program uses the backend (gameoflife does not)', () => {
        expect(usesBackend({ backend: 'OpenCL', arguments: '2048 OpenCL' })).toBe(true);
        expect(usesBackend({ backend: 'OpenCL', arguments: '2 64 100 OpenCL redblack' })).toBe(true);
        expect(usesBackend({ backend: 'OpenMP', arguments: '100 100 10' })).toBe(false);
        expect(usesBackend({ backend: 'OpenMP' })).toBe(false);
        expect(usesBackend({ arguments: '100 OpenMP' })).toBe(false);
    });
});

describe('the plan of a run', () => {
    it('is the run rule, with user-set options replacing the Makefile values', () => {
        const dot = overrides => run({ overrides }, examples, '/x/examples/dotproduct.cpp');
        expect(dot({}).commands).toEqual([['make', 'run/dotproduct', 'TRACE=1']]);
        expect(dot({}).trace).toBe('trace/trace_dotproduct.json');
        expect(dot({ backend: 'CPU', arguments: '5 CPU', snapshots: true, rebuild: true }).commands)
            .toEqual([['make', '-B', 'run/dotproduct', 'TRACE=1', 'TRACE_SNAPSHOTS=1', 'RUN_BACKEND=CPU', 'lookup.dotproduct=5 CPU']]);
        expect(dot({ snapshots: false }).commands).toEqual([['make', 'run/dotproduct', 'TRACE=1', 'TRACE_SNAPSHOTS=0']]);
        const untraced = dot({ trace: false, snapshots: true });
        expect(untraced.commands).toEqual([['make', 'run/dotproduct', 'TRACE=0']]);
        expect(untraced.trace).toBeUndefined();
    });

    it('has the trace where programs run (a package: the directory of skepu.mk)', () => {
        expect(run({ values: { runCwd: '..', traceDir: 'trace' } }).trace).toBe('../trace/trace_hello.json');
        expect(run({ values: { runCwd: '.', traceDir: 'trace-mpi' } }).trace).toBe('trace-mpi/trace_hello.json');
        expect(run({}).trace).toBe('../trace/trace_hello.json');                                    // make cannot say: skepu.mk is one directory up
        expect(run({}, hello.replace('../skepu.mk', 'skepu.mk')).trace).toBe('trace/trace_hello.json');
        expect(run({ overrides: { arguments: 'CPU 7' } }).commands).toEqual([['make', 'run/hello', 'TRACE=1', 'lookup.hello=CPU 7']]);
    });

    it('builds only: no arguments, no trace', () => {
        const build = (makefile, source, overrides = {}) => run({ buildOnly: true, overrides, makefile }, makefile, source);
        const plan = build(examples, '/x/examples/dotproduct.cpp', { backend: 'CPU', arguments: '5', snapshots: true });
        expect(plan.commands).toEqual([['make', 'bin/dotproduct', 'TRACE=1', 'TRACE_SNAPSHOTS=1']]);
        expect(plan.trace).toBeUndefined();
        expect(build(hello, '/p/hello.cpp').commands).toEqual([['make', 'bin/hello', 'TRACE=1']]);
    });

    it('is the same for every source: the lookup, the given arguments, or none', () => {
        const other = extra => run(extra, examples, '/x/dotproduct_copy.cpp');
        expect(other({}).commands).toEqual([['make', 'run/dotproduct_copy', 'TRACE=1']]);      // empty form: the lookup if any, else no arguments
        expect(other({ overrides: { arguments: '2048 OpenMP' } }).commands).toEqual([['make', 'run/dotproduct_copy', 'TRACE=1', 'lookup.dotproduct_copy=2048 OpenMP']]);
        expect(other({ buildOnly: true }).commands).toEqual([['make', 'bin/dotproduct_copy', 'TRACE=1']]);
    });

    it('rejects names with spaces and Makefiles without skepu.mk', () => {
        expect(run({}, hello, '/p/hello copy.cpp').error).toMatch(/space/);
        expect(run({}, 'all:\n', '/p/a.cpp').error).toMatch(/skepu\.mk/);
    });
});

describe('the options form', () => {
    const form = { program: 'dotproduct', file: 'dotproduct.cpp', overrides: {}, makefile: { backend: 'OpenCL', arguments: '2048 OpenCL' }, cspSource: 'x', nonce: 'N' };

    it('shows the Makefile values until set', () => {
        const html = runOptionsHtml(form);
        expect(html).toContain('Default backend');
        expect(html).toContain('<h1>Options for dotproduct.cpp</h1>');
        expect(html).toContain('Makefile (OpenCL)');
        expect(html).toContain('placeholder="2048 OpenCL"');
        expect(html).not.toMatch(/\$\{backend\}|\$\(\w+_/);
    });

    it('hides the backend when the program does not use it', () => {
        const html = runOptionsHtml({ ...form, makefile: { backend: 'OpenMP', arguments: '100 100 10' } });
        expect(html).toContain('<select id="backend" hidden>');
        expect(html).toContain('placeholder="100 100 10"');
        expect(html).toContain('Replaces the Makefile\'s arguments.');
        expect(runOptionsHtml({ ...form, makefile: {} })).not.toContain('Replaces');
    });

    it('shows and escapes set values; arguments disable the backend', () => {
        const html = runOptionsHtml({ ...form, overrides: { backend: 'CPU', arguments: '"<b>"', snapshots: true } });
        expect(html).toContain('<option value="CPU" selected>');
        expect(html).toContain('value="&#34;&#60;b&#62;&#34;"');
        expect(html).toMatch(/<select id="backend" disabled>/);
        expect(html).toMatch(/id="trace" checked/);
        expect(html).toMatch(/id="snapshots" checked/);
        expect(html).not.toMatch(/id="rebuild" checked/);
        expect(html).toContain('Enable tracing');
        expect(html).toContain('Enable snapshots');
        expect(html).toContain('Force rebuild');
        expect(html.indexOf('>Build<')).toBeLessThan(html.indexOf('>Build and Run<'));
    });

    it('disables snapshots without tracing', () => {
        const html = runOptionsHtml({ ...form, overrides: { trace: false, snapshots: true } });
        expect(html).not.toMatch(/id="trace" checked/);
        expect(html).toMatch(/id="snapshots" disabled>/);
    });
});
