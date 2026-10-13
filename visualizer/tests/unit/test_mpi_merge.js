// test_mpi_merge.js — the trace files of an MPI program merged (script/backend/mpi.js). The fixture is the trace of
// tests/mpi/trace_map.cpp on 3 processes, with the clocks of the processes started 100 ms apart
// (SKEPU_TRACE_TEST_SKEW_US), so that the times in the files are far from aligned.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { mergeMpiTraces, alignedTime, stats, isMpiHeader } from '../../script/backend/mpi.js';
import { parseTrace } from '../../script/backend/trace.js';

const DIR = join(dirname(fileURLToPath(import.meta.url)), '../fixtures/mpi/trace_map_3');
const NAMES = ['skepu_trace.json', 'skepu_trace.rank1.json', 'skepu_trace.rank2.json'];
const texts = NAMES.map(name => ({ name, text: readFileSync(join(DIR, name), 'utf-8') }));
const withEdit = (i, edit) => texts.map((f, k) => {
    if (k !== i) return f;
    const t = JSON.parse(f.text);
    edit(t);
    return { name: f.name, text: JSON.stringify(t) };
});

describe('clock alignment', () => {
    const clock = { start: { local_us: 0, offset_us: 100 }, end: { local_us: 1000, offset_us: 200 } };
    it('is the offset at the start, then follows the drift', () => {
        expect(alignedTime(clock, 0)).toBe(100);
        expect(alignedTime(clock, 500)).toBe(650);
        expect(alignedTime(clock, 1000)).toBe(1200);
        expect(alignedTime(clock, 2000)).toBe(2300); // (the drift goes on)
    });
    it('is constant with one sample, and the identity without', () => {
        expect(alignedTime({ start: { local_us: 5, offset_us: 7 } }, 50)).toBe(57);
        expect(alignedTime(null, 50)).toBe(50);
    });
});

describe('stats', () => {
    it('min, mean, max and where the max is, over the values that there are', () => {
        expect(stats([4, null, 10, 1])).toEqual({ min: 1, mean: 5, max: 10, argmax: 2 });
        expect(stats([null, null])).toBeNull();
    });
});

describe('merging the fixture', () => {
    const merged = mergeMpiTraces(texts);
    const calls = merged.events.filter(e => e.type === 'skeleton_call');
    const comms = merged.events.filter(e => e.type === 'communication');

    it('is the trace of rank 0', () => {
        const rank0 = parseTrace(texts[0].text);
        expect(merged.events).toHaveLength(rank0.events.length);
        expect(merged.events.map(e => e.idx)).toEqual(rank0.events.map(e => e.idx));
        expect(merged.header).toEqual(rank0.header);
        expect(merged.mpi.ranks).toBe(3);
        expect(merged.mpi.files.map(f => f.rank)).toEqual([0, 1, 2]);
        expect(merged.warnings).toEqual([]);
        expect(merged.divergence).toBeNull();
        expect(calls).toHaveLength(10);
        expect(comms.length).toBeGreaterThan(0);
        // what is not a call or a collective is as it was
        const alloc = merged.events.find(e => e.type === 'allocation');
        expect(alloc.rank_data).toBeUndefined();
    });

    it('has what every rank did in each call, on one clock', () => {
        for (const c of calls) {
            expect(c.rank_data).toHaveLength(3);
            expect(c.rank_data.map(r => r.rank)).toEqual([0, 1, 2]);
            expect(c.rank_data[0].start).toBe(c.start);          // (rank 0 is the clock)
            const elements = c.rank_data.reduce((s, r) => s + r.local_elements, 0);
            expect(elements).toBe(c.elements[0]);
            const s = c.rank_summary;
            for (const r of c.rank_data) {
                expect(r.start).toBeGreaterThanOrEqual(s.interval.start);
                expect(r.end).toBeLessThanOrEqual(s.interval.end);
            }
            expect(s.arrival_spread_us).toBeLessThan(50000);     // (not the 100 ms of the clocks: they are aligned)
            expect(s.work.max).toBeGreaterThanOrEqual(s.work.mean);
            expect(s.slowest_rank).toBe(s.work.argmax);
            expect(s.imbalance).toBeGreaterThanOrEqual(1);
            // (the elements are in blocks of the same size, but for a call with a MatRow matrix: see below)
            if (!c.proxy_inputs.length) expect(s.local_elements.max - s.local_elements.min).toBeLessThanOrEqual(1);
        }
    });

    it('shows the imbalance of a call whose elements follow the rows of a matrix', () => {
        // 5 elements, a matrix of 12 rows: the elements go to the processes that have the rows (4 + 4 + 4), so 4, 1 and 0 of them
        const c = calls[calls.length - 1];
        expect(c.proxy_inputs).toHaveLength(1);
        expect(c.rank_data.map(r => r.local_elements)).toEqual([4, 1, 0]);
        expect(c.rank_data[2].work_start).toBeNull();            // (the third process did nothing)
        expect(c.rank_summary.local_elements.argmax).toBe(0);
        expect(c.rank_summary.work.min).toBe(0);                  // (a rank that did no work counts as zero)
        // 4 elements on rank 0 and 1 on rank 1: how much longer than the mean over the three ranks the busiest worked
        const w = c.rank_data.map(r => r.work_start != null ? r.work_end - r.work_start : 0);
        expect(c.rank_summary.imbalance).toBeCloseTo(Math.max(...w) / (w.reduce((a, b) => a + b, 0) / 3), 6);
        expect(c.rank_summary.slowest_rank).toBe(w.indexOf(Math.max(...w)));
        expect(c.rank_summary.comm_bytes.max).toBeGreaterThan(0); // (and the rows and results moved)
    });

    it('has the collectives of every rank, ending together', () => {
        for (const c of comms) {
            expect(c.rank_data).toHaveLength(3);
            const s = c.rank_summary;
            expect(s.end_spread_us).toBeLessThan(1000);
            expect(s.bytes_sent).toBe(s.bytes_received);
            expect(Math.min(...s.wait_us)).toBe(0);
            expect(s.wait_us[s.last_arrival_rank]).toBe(0);
            expect(s.max_wait_us).toBe(Math.max(...s.wait_us));
        }
        expect(merged.mpi.collective_end_spread_us).toBeLessThan(1000);
    });

    it('would not be aligned without the clocks of the headers', () => {
        // the raw times of rank 1 are about 100 ms off, which is what the offsets of the header correct
        const raw = parseTrace(texts[1].text).events.filter(e => e.type === 'rank_communication');
        const c = comms[0];
        const lean = raw.find(e => e.comm_seq === c.comm_seq);
        expect(Math.abs(lean.end - c.end)).toBeGreaterThan(50000);
        expect(Math.abs(c.rank_data[1].end - c.end)).toBeLessThan(1000);
    });

    it('does not depend on the order of the files', () => {
        const again = mergeMpiTraces([texts[2], texts[0], texts[1]]);
        expect(again.events).toEqual(merged.events);
    });
});

describe('what is wrong with a set of files', () => {
    it('a missing rank is a warning, and its entries are null', () => {
        const m = mergeMpiTraces([texts[0], texts[1]]);
        expect(m.mpi.missing_ranks).toEqual([2]);
        expect(m.warnings.join(' ')).toMatch(/No trace file for rank 2/);
        const c = m.events.find(e => e.type === 'skeleton_call');
        expect(c.rank_data[2]).toBeNull();
        expect(c.rank_summary.interval.end).toBeGreaterThan(c.rank_summary.interval.start);
    });

    it('a rank that makes another call is a divergence', () => {
        const m = mergeMpiTraces(withEdit(1, t => { t.data.filter(e => e.type === 'rank_call')[3].pattern = 'Reduce 1D-Full'; }));
        expect(m.divergence).toMatchObject({ kind: 'call', rank: 1 });
        expect(m.divergence.seq).toBe(4);
        expect(m.warnings.join(' ')).toMatch(/different call/);
    });

    it('a rank with a different label is a divergence (the signature)', () => {
        const m = mergeMpiTraces(withEdit(2, t => { t.data.filter(e => e.type === 'rank_call')[0].signature = 1; }));
        expect(m.divergence).toMatchObject({ kind: 'call', rank: 2, seq: 1 });
    });

    it('a rank that lacks a collective, or has another one', () => {
        const lacks = mergeMpiTraces(withEdit(1, t => { t.data = t.data.filter(e => !(e.type === 'rank_communication' && e.comm_seq === 2)); }));
        expect(lacks.divergence).toMatchObject({ kind: 'communication', rank: 1, comm_seq: 2 });
        const other = mergeMpiTraces(withEdit(1, t => { t.data.find(e => e.type === 'rank_communication').operation = 'bcast'; }));
        expect(other.divergence).toMatchObject({ kind: 'communication', rank: 1, found: 'bcast' });
    });

    it('a rank that has more calls than rank 0', () => {
        const m = mergeMpiTraces(withEdit(1, t => { t.data.push({ type: 'rank_call', seq: 99, pattern: 'Map', signature: 1, start: 1, end: 2, local_elements: 0 }); }));
        expect(m.divergence).toMatchObject({ kind: 'extra', rank: 1 });
    });

    it('clocks that are not aligned are a warning', () => {
        const m = mergeMpiTraces(withEdit(2, t => { t.header.clock.start.offset_us += 500000; t.header.clock.end.offset_us += 500000; }));
        expect(m.warnings.join(' ')).toMatch(/not well aligned/);
    });

    it('files that are not the files of one MPI program', () => {
        expect(() => mergeMpiTraces([])).toThrow(/No trace files/);
        expect(() => mergeMpiTraces([texts[1], texts[2]])).toThrow(/rank 0, which has the trace of the program, is missing/);
        expect(() => mergeMpiTraces([texts[0], texts[0]])).toThrow(/is in two files/);
        expect(() => mergeMpiTraces([{ name: 'x.json', text: '[]' }])).toThrow(/not a trace of an MPI program/);
        expect(() => mergeMpiTraces(withEdit(1, t => { t.header.mpi.ranks = 4; }))).toThrow(/different numbers of processes/);
        expect(() => mergeMpiTraces(withEdit(1, t => { t.header.mpi.role = 'full'; }))).toThrow(/has the role full/);
    });

    it('knows an MPI header', () => {
        expect(isMpiHeader(parseTrace(texts[0].text).header)).toBe(true);
        expect(isMpiHeader({ '/a.cpp': 'h' })).toBe(false);
        expect(isMpiHeader(null)).toBe(false);
    });
});

describe('an skepu::external call in the merged trace', () => {
    const merged = mergeMpiTraces(texts);
    const externals = merged.events.filter(e => e.type === 'external');

    it('is a call: rank 0 runs the code and the other ranks wait for it', () => {
        expect(externals.length).toBeGreaterThan(0);
        for (const e of externals) {
            expect(e.rank_data).toHaveLength(3);
            expect(e.rank_data[0].work_start).not.toBeNull();         // (the code runs on rank 0)
            expect(e.rank_data[1].work_start).toBeNull();
            expect(e.rank_data[2].work_start).toBeNull();
            expect(e.rank_summary.slowest_rank).toBe(0);
            expect(e.rank_summary.local_elements.max).toBe(0);
            // (the code runs on one of the three ranks: as unbalanced as it can be; the work is measured in microseconds, and an external of less than one has none)
            if (e.rank_data[0].work_end - e.rank_data[0].work_start > 0) expect(e.rank_summary.imbalance).toBeCloseTo(3, 6);
            else expect(e.rank_summary.imbalance).toBeNull();
        }
        expect(merged.warnings).toEqual([]);
    });

    it('has the gather before it and the scatter after it', () => {
        const comms = merged.events.filter(e => e.type === 'communication' && e.seq === externals[0].seq);
        expect(comms.length).toBeGreaterThan(0);
        // (the containers that it writes are gathered too: the code may change some of the elements only, and the processes get them all back)
        expect(new Set(comms.map(c => c.phase))).toEqual(new Set(['before', 'after']));
        expect(comms.filter(c => c.phase === 'before').every(c => c.operation === 'allgather')).toBe(true);
        expect(comms.filter(c => c.phase === 'after').every(c => c.operation === 'scatter')).toBe(true);
    });

    it('a rank that does something else than the external is a divergence', () => {
        const m = mergeMpiTraces(withEdit(2, t => { t.data.find(e => e.type === 'rank_call' && e.pattern === 'external').pattern = 'Map'; }));
        expect(m.divergence).toMatchObject({ kind: 'call', rank: 2, expected: 'external', found: 'Map' });
    });
});

describe('the source files of the header', () => {
    const merge = (edits) => mergeMpiTraces(texts.map((f, k) => {
        const t = JSON.parse(f.text);
        t.header.files = edits[k];
        if (edits[k] === undefined) delete t.header.files;
        return { name: f.name, text: JSON.stringify(t) };
    }));

    it('are those of the process with the trace, with the files that the other processes name', () => {
        const m = merge([{ '/a.cpp': 'h1' }, { '/b.cpp': 'h2' }, undefined]);
        expect(m.header.files).toEqual({ '/a.cpp': 'h1', '/b.cpp': 'h2' });
        expect(m.warnings.filter(w => /version/.test(w))).toEqual([]);
    });

    it('keep the hash of the process with the trace, and warn once of another version on another process', () => {
        const m = merge([{ '/a.cpp': 'h1' }, { '/a.cpp': 'h2' }, { '/a.cpp': 'h3' }]);
        expect(m.header.files).toEqual({ '/a.cpp': 'h1' });
        expect(m.warnings.filter(w => /another version of \/a\.cpp/.test(w))).toHaveLength(1);
    });

    it('are not added to a header that had none', () => {
        expect(merge([undefined, undefined, undefined]).header.files).toBeUndefined();
    });
});
