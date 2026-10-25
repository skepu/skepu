// test_example_traces.js — the built-in examples (examples/examples.json): every trace is a well-formed trace of the current tracer, and goes with the
// source files that the example ships. The regions nest as their depths say, each event is in a region that was opened before it, an event that
// has a file has its hash in the header, and the hash of a source file of the example is that of the file (the trace is of this version of the program).
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../../examples');
const examples = JSON.parse(readFileSync(join(ROOT, 'examples.json'), 'utf-8'));

function files(value) {
    if (value == null) return [];
    if (typeof value === 'string') return [value];
    if (Array.isArray(value)) return value.flatMap(files);
    return Object.values(value).flatMap(files);
}

const sha = path => createHash('sha256').update(readFileSync(path)).digest('hex');

for (const example of examples) {
    const sources = files(example.cpp_files).filter(f => /\.(cpp|hpp|h)$/.test(f));
    for (const [backend, entry] of Object.entries(example.trace)) {
        // (the trace of the first process of an MPI program; the others have their own tests)
        const path = Array.isArray(entry) ? entry[0] : entry;
        if (typeof path !== 'string' || !existsSync(join(ROOT, path))) continue;
        describe(`${example.title}, ${backend}`, () => {
            const trace = JSON.parse(readFileSync(join(ROOT, path), 'utf-8'));
            const data = Array.isArray(trace) ? trace : trace.data;
            const header = Array.isArray(trace) ? {} : (trace.header || {});

            it('has events with a type', () => {
                expect(data.length).toBeGreaterThan(0);
                for (const e of data) expect(typeof e.type).toBe('string');
            });

            it('has regions that nest as their depths say, with the events in regions that are open', () => {
                const stack = [];   // the regions that are open as far as the trace says (it has no end events: a region is open until one of lesser depth)
                for (const e of data) {
                    if (e.type === 'region') {
                        expect(e.region_depth).toBeGreaterThanOrEqual(1);
                        stack.length = Math.min(stack.length, e.region_depth - 1);
                        // (the region around it is the one that is open at that depth)
                        if (e.region_depth > 1) expect(e.region).toBe(stack[e.region_depth - 2]);
                        else expect(e.region).toBeUndefined();
                        stack[e.region_depth - 1] = e.label;
                    } else if (e.region !== undefined) {
                        expect(stack).toContain(e.region);
                    }
                }
            });

            it('has the hash of each file that an event names, and the hash of a shipped source is that of the file', () => {
                const hashes = header.files || {};
                for (const name of new Set(data.map(e => e.file).filter(Boolean)))
                    if (Object.keys(hashes).length) expect(Object.keys(hashes).some(f => f === name || f.endsWith('/' + basename(name)) || basename(f) === basename(name))).toBe(true);
                for (const source of sources) {
                    const recorded = Object.entries(hashes).find(([f]) => basename(f) === basename(source));
                    if (recorded && existsSync(join(ROOT, source))) expect(recorded[1], source).toBe(sha(join(ROOT, source)));
                }
            });
        });
    }
}
