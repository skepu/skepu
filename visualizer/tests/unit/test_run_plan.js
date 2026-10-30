// test_run_plan.js — how a program is built and run for a trace (src/runPlan.ts): by the rules of the Makefile next to its source.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { makefileKind, programsOf, planRun, splitArguments } from '../../src/runPlan.ts';
import { runOptionsHtml } from '../../src/runOptionsHtml.ts';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const examples = readFileSync(join(ROOT, 'examples/Makefile'), 'utf8');
const smoke = readFileSync(join(ROOT, 'packaging/smoke/Makefile'), 'utf8');
const options = { overrides: {}, make: 'make' };

describe('the Makefiles', () => {
    it('are the one of the examples and those of a package', () => {
        expect(makefileKind(examples)).toBe('examples');
        expect(makefileKind(smoke)).toBe('package');
        expect(makefileKind('all:\n\techo\n')).toBeUndefined();
    });

    it('list their programs', () => {
        expect(programsOf(examples)).toEqual(expect.arrayContaining(['dotproduct', 'coulombic']));
        expect(programsOf(smoke)).toEqual(['smoke']);
        expect(programsOf('PROGS = a b \\\n  c # d\nPROGS += e')).toEqual(['a', 'b', 'c', 'e']);
    });
});

const run2 = (makefile, source) => planRun({ ...options, makefile, source }, '/p').error;

describe('the plan of a run', () => {
    it('is, in the examples, the run rule of the Makefile, with what the user has set instead of what the Makefile has', () => {
        const run = overrides => planRun({ ...options, overrides, makefile: examples, source: '/x/examples/dotproduct.cpp' }, '/x/examples');
        expect(run({}).commands).toEqual([['make', 'run/dotproduct']]);
        expect(run({}).trace).toBe('trace/trace_dotproduct.json');
        expect(run({ backend: 'CPU', arguments: '5 CPU', snapshots: true, rebuild: true }).commands)
            .toEqual([['make', '-B', 'run/dotproduct', 'RUN_BACKEND=CPU', 'lookup.dotproduct=5 CPU', 'TRACE_SNAPSHOTS=1']]);
        expect(run({ snapshots: false }).commands).toEqual([['make', 'run/dotproduct', 'TRACE_SNAPSHOTS=0']]);
        const untraced = run({ trace: false, snapshots: true });      // no tracing: no snapshots either, and no trace
        expect(untraced.commands).toEqual([['make', 'run/dotproduct', 'TRACE=0']]);
        expect(untraced.trace).toBeUndefined();
    });

    it('is, to build only, the build of the program (no arguments, no trace to show)', () => {
        const plan = planRun({ ...options, buildOnly: true, overrides: { backend: 'CPU', arguments: '5', snapshots: true }, makefile: examples, source: '/x/examples/dotproduct.cpp' }, '/x/examples');
        expect(plan.commands).toEqual([['make', 'parallel/dotproduct', 'TRACE_SNAPSHOTS=1']]);
        expect(plan.trace).toBeUndefined();
        const pack = planRun({ ...options, buildOnly: true, makefile: smoke, source: '/p/smoke.cpp' }, '/p');
        expect(pack.commands).toEqual([['make', 'bin/smoke', 'TRACE=1']]);
        expect(pack.trace).toBeUndefined();
    });

    it('is, in a package, the run rule of skepu.mk, with the trace where the programs are run from (the directory of skepu.mk)', () => {
        const run = (overrides, makefile = smoke, buildOnly) => planRun({ ...options, overrides, buildOnly, makefile, source: '/p/programs/smoke.cpp' }, '/p/programs');
        expect(run({}).commands).toEqual([['make', 'run/smoke', 'TRACE=1']]);
        expect(run({}).trace).toBe('../trace_smoke.json');
        expect(run({}, smoke.replace('../skepu.mk', 'skepu.mk')).trace).toBe('trace_smoke.json');
        expect(run({ arguments: 'CPU 7', snapshots: true, rebuild: true }).commands).toEqual([['make', '-B', 'run/smoke', 'TRACE=1', 'TRACE_SNAPSHOTS=1', 'lookup.smoke=CPU 7']]);
        const untraced = run({ trace: false, snapshots: true });
        expect(untraced.commands).toEqual([['make', 'run/smoke', 'TRACE=0']]);
        expect(untraced.trace).toBeUndefined();
        expect(run({}, smoke, true).commands).toEqual([['make', 'bin/smoke', 'TRACE=1']]);
    });

    it('has a reason when make cannot build the name, or the Makefile is not one that builds programs', () => {
        expect(run2(smoke, '/p/smoke copy.cpp')).toMatch(/space/);
        expect(planRun({ ...options, makefile: 'all:\n', source: '/p/a.cpp' }, '/p').error).toMatch(/skepu\.mk/);
    });

    it('splits arguments', () => {
        expect(splitArguments(`a 'b c' "d e"  f`)).toEqual(['a', 'b c', 'd e', 'f']);
    });
});

describe('the form of the options', () => {
    const form = { program: 'dotproduct', file: 'dotproduct.cpp', listed: true, kind: 'examples', overrides: {}, makefile: { backend: 'OpenCL', arguments: '2048 OpenCL' }, cspSource: 'x', nonce: 'N' };

    it('has what the Makefile runs the program with until it is set', () => {
        const html = runOptionsHtml(form);
        expect(html).toContain('Default backend');
        expect(html).toContain('<h1>Options for dotproduct.cpp</h1>');
        expect(html).toContain('Makefile (OpenCL)');
        expect(html).toContain('placeholder="2048 OpenCL"');
        expect(html).not.toMatch(/\$\{backend\}|\$\(\w+_/);
    });

    it('says that the arguments are required for a program that is not in PROGS', () => {
        expect(runOptionsHtml({ ...form, listed: false })).toContain('Required: not in PROGS');
        expect(runOptionsHtml(form)).not.toContain('Required');
    });

    it('has no backend for a package, and the arguments of its Makefile', () => {
        const html = runOptionsHtml({ ...form, kind: 'package', makefile: { arguments: 'OpenMP' } });
        expect(html).toContain('<select id="backend" hidden>');
        expect(html).toContain('placeholder="OpenMP"');
        expect(html).toContain('Replace the arguments of the Makefile.');
        expect(runOptionsHtml({ ...form, kind: 'package', makefile: {} })).toContain('Arguments of the program.');
    });

    it('shows what is set, escapes it, and has the backend off when there are arguments', () => {
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

    it('has snapshots off, and not to be set, without tracing', () => {
        const html = runOptionsHtml({ ...form, overrides: { trace: false, snapshots: true } });
        expect(html).not.toMatch(/id="trace" checked/);
        expect(html).toMatch(/id="snapshots" disabled>/);
    });
});
