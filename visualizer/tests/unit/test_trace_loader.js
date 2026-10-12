// test_trace_loader.js — a trace is an array of events or { header, data }: the loader (trace.js and the worker) reads both,
// and the line ranges of the events are those of the events wherever the "data" key is.
import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseTrace, eventsArrayStart, computeTraceLineRanges, headerSourceFiles, headerInstances } from '../../script/backend/trace.js';
import { summarize } from '../../script/backend/analysis.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const events = [
    { type: 'allocation', object_id: 1, label: 'a', time: 5 },
    { type: 'skeleton_call', object_id: 9, label: 'map', pattern: 'Map', start: 6, end: 8, elements: [4], outputs: [1], elwise_inputs: [], proxy_inputs: [], scalar_inputs: [] },
];

describe('trace formats', () => {
    it('an array of events', () => {
        const text = JSON.stringify(events, null, 4);
        const t = parseTrace(text);
        expect(t.header).toBeNull();
        expect(t.events.map(e => e.idx)).toEqual([0, 1]);
        expect(computeTraceLineRanges(text)).toHaveLength(2);
    });

    it('{ header, data }, with arrays in the header and the data first or last', () => {
        const header = { '/src/a.cpp': 'abc', other: [1, [2, 3], '[']  };
        for (const text of [JSON.stringify({ header, data: events }, null, 4), JSON.stringify({ data: events, header }, null, 4)]) {
            const t = parseTrace(text);
            expect(t.header).toEqual(header);
            expect(t.events).toHaveLength(2);
            const ranges = computeTraceLineRanges(text);
            expect(ranges).toHaveLength(2);
            // the lines of the range are the lines of that event
            const lines = text.split('\n');
            expect(lines.slice(ranges[1][0] - 1, ranges[1][1]).join('\n')).toContain('"skeleton_call"');
            expect(lines.slice(ranges[0][0] - 1, ranges[0][1]).join('\n')).toContain('"allocation"');
        }
    });

    it('finds the start of the events', () => {
        expect(eventsArrayStart('  [1]')).toBe(2);
        const text = '{"header": {"x": "[data]"}, "data": [1, 2]}';
        expect(text[eventsArrayStart(text)]).toBe('[');
        expect(text.slice(eventsArrayStart(text))).toBe('[1, 2]}');
        expect(() => eventsArrayStart('{"header": {}}')).toThrow(/no "data"/);
        expect(() => eventsArrayStart('42')).toThrow(/Not a SkePU trace/);
    });

    it('the source files of a header are not its sections', () => {
        const header = { '/a.cpp': 'h1', '/b.cpp': 'h2', mpi: { rank: 0 }, clock: { start: {} } };
        expect(headerSourceFiles(header)).toEqual({ '/a.cpp': 'h1', '/b.cpp': 'h2' });
        const s = summarize(parseTrace(JSON.stringify({ header, data: events })).events, header);
        expect(s.header.files).toBe(2);
        expect(s.header.mpi).toEqual({ rank: 0 });
        expect(summarize([], null).header).toBeNull();
    });
});

describe('the worker loads { header, data }', () => {
    const replies = {};
    let seq = 0;
    const call = async (type, payload = {}) => {
        const id = String(++seq);
        await globalThis.self.onmessage({ data: { id, type, ...payload } });
        const reply = replies[id];
        if (reply.type === 'error') throw new Error(reply.error);
        return reply;
    };
    beforeAll(async () => {
        globalThis.self = {
            postMessage: m => { replies[m.id] = m; },
            location: { href: 'file://' + join(ROOT, 'script/backend/visualizer.js') },
            addEventListener() {},
        };
        await import(join(ROOT, 'script/backend/visualizer.js'));
    });

    const fixture = name => readFileSync(join(ROOT, 'tests/fixtures/mpi/trace_map_3', name), 'utf-8');

    it('the trace of rank 0 of an MPI program', async () => {
        const text = fixture('skepu_trace.json');
        const loaded = await call('load', { traceJson: text, traceFilename: 'skepu_trace.json' });
        expect(loaded.eventCount).toBe(JSON.parse(text).data.length);
        expect(loaded.header.mpi).toMatchObject({ rank: 0, ranks: 3, role: 'full' });
        expect(loaded.traceLineRanges).toHaveLength(loaded.eventCount);
        const graph = await call('graph', {});
        expect(graph.event_count).toBe(loaded.eventCount);
        const calls = JSON.parse(text).data.filter(e => e.type === 'skeleton_call').length;
        expect(calls).toBe(10);
        expect(graph.nodes.length).toBeGreaterThan(0);
    });

    it('the files of the other ranks are there for the trace pane, with the lines of what the main trace has', async () => {
        const names = ['skepu_trace.json', 'skepu_trace.rank1.json', 'skepu_trace.rank2.json'];
        const loaded = await call('load', { traceFiles: names.map(name => ({ name, text: fixture(name) })) });
        expect(loaded.rankTraces.map(t => t.rank)).toEqual([1, 2]);
        const main = JSON.parse(fixture(names[0])).data;
        for (const t of loaded.rankTraces) {
            const lines = t.text.split('\n');
            expect(t.ranges).toHaveLength(main.length);
            main.forEach((e, i) => {
                const hasRange = t.ranges[i] != null;
                expect(hasRange, `${e.type} #${i}`).toBe(e.type === 'communication' || ((e.type === 'skeleton_call' || e.type === 'external') && e.seq != null));
                if (hasRange) {
                    const body = lines.slice(t.ranges[i][0] - 1, t.ranges[i][1]).join('\n');
                    expect(body).toContain(e.type === 'communication' ? '"rank_communication"' : '"rank_call"');
                    expect(body).toContain(e.type === 'communication' ? `"comm_seq": ${e.comm_seq}` : `"seq": ${e.seq}`);
                }
            });
        }
    });

    it('an array trace as before', async () => {
        const text = JSON.stringify(events);
        const loaded = await call('load', { traceJson: text });
        expect(loaded.eventCount).toBe(2);
        expect(loaded.header).toBeNull();
    });
});

describe('the examples', () => {
    const exampleDir = join(ROOT, 'examples');
    const examples = JSON.parse(readFileSync(join(exampleDir, 'examples.json'), 'utf-8'));

    it('have all their files', () => {
        for (const ex of examples) {
            for (const rel of Object.values(ex.trace).flat())
                expect(() => readFileSync(join(exampleDir, rel)), `${ex.title}: ${rel}`).not.toThrow();
            for (const rel of Array.isArray(ex.cpp_files) ? ex.cpp_files : Object.values(ex.cpp_files ?? {}).flat())
                expect(() => readFileSync(join(exampleDir, rel)), `${ex.title}: ${rel}`).not.toThrow();
        }
    });

    it('with the trace files of an MPI program are merged without a warning, and the worker loads them', async () => {
        const mpi = examples.filter(ex => Object.keys(ex.trace).includes('mpi'));
        expect(mpi.length).toBeGreaterThan(0);
        const replies = {};
        globalThis.self = { postMessage: m => { replies[m.id] = m; }, location: { href: 'file://' + join(ROOT, 'script/backend/visualizer.js') }, addEventListener() {} };
        await import(join(ROOT, 'script/backend/visualizer.js?mpi-example'));
        for (const ex of mpi) {
            const traceFiles = ex.trace.mpi.map(rel => ({ name: rel.split('/').pop(), text: readFileSync(join(exampleDir, rel), 'utf-8') }));
            const id = 'x' + ex.title;
            await globalThis.self.onmessage({ data: { id, type: 'load', traceFiles } });
            const loaded = replies[id];
            expect(loaded.type, loaded.error).not.toBe('error');
            expect(loaded.mpi.ranks).toBe(ex.trace.mpi.length);
            expect(loaded.warnings, ex.title).toEqual([]);
            expect(loaded.divergence).toBeNull();
            expect(loaded.traceFilename).toBe(ex.trace.mpi[0].split('/').pop()); // (the trace of rank 0 is what the trace pane shows)
            const gid = 'g' + ex.title;
            await globalThis.self.onmessage({ data: { id: gid, type: 'graph' } });
            expect(replies[gid].nodes.length).toBeGreaterThan(0);
        }
    });
});

describe('the table of the instances in the header', () => {
    const table = {
        '7': { label: 'dotp', instance: { file: '/a.cpp', line: 12, end_line: 13 }, user_functions: [
            { name: 'mult', kind: 'function', file: '/a.cpp', line: 4, end_line: 6 }, { name: '', kind: 'lambda', file: '/a.cpp', line: 12 }] },
        '8': { label: 'Map@a.cpp:30', instance: { file: '/a.cpp', line: 30, anonymous: true }, user_functions: [] },
        '9': { label: 'broken', instance: { file: '/a.cpp' }, user_functions: [] },
    };

    it('is a Map by the id of the instance, with the lines of the instance and of the functions', () => {
        const instances = headerInstances({ '/a.cpp': 'hash', instances: table });
        expect([...instances.keys()]).toEqual([7, 8]);   // (the entry without lines is left out)
        expect(instances.get(7).instance).toEqual({ file: '/a.cpp', line: 12, end_line: 13 });
        expect(instances.get(7).user_functions.map(f => [f.name, f.kind, f.line, f.end_line])).toEqual([['mult', 'function', 4, 6], ['', 'lambda', 12, 12]]);
        expect(instances.get(8).instance).toEqual({ file: '/a.cpp', line: 30, end_line: 30, anonymous: true });
    });

    it('is empty without a table, and is not a source file', () => {
        expect(headerInstances(null).size).toBe(0);
        expect(headerInstances({ mpi: {} }).size).toBe(0);
        expect(headerSourceFiles({ '/a.cpp': 'hash', instances: table })).toEqual({ '/a.cpp': 'hash' });
    });

    it('goes through parseTrace as it is', () => {
        const t = parseTrace(JSON.stringify({ header: { instances: table }, data: events }, null, 4));
        expect(headerInstances(t.header).get(7).label).toBe('dotp');
    });
});
