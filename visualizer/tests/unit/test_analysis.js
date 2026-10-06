// test_analysis.js — the events-only analyses (backend/analysis.js), the trace reader and the
// critical path (backend/graph_analysis.js), on small fixtures and on the shipped DNN trace.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseTrace } from '../../script/backend/trace.js';
import * as A from '../../script/backend/analysis.js';
import { criticalPath } from '../../script/backend/graph_analysis.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const load = rel => parseTrace(readFileSync(join(ROOT, rel), 'utf8'));
const fixture = name => load(`tests/fixtures/${name}.json`).events;

describe('parseTrace', () => {
    it('numbers the events of an array', () => {
        const { header, events } = parseTrace('[{"type":"allocation","label":"A"},{"type":"deallocation","label":"A"}]');
        expect(header).toBeNull();
        expect(events.map(e => e.idx)).toEqual([0, 1]);
    });
    it('reads { header, data }', () => {
        const { header, events } = parseTrace('{"header":{"a.cpp":"abc"},"data":[{"type":"allocation","label":"A"}]}');
        expect(header).toEqual({ 'a.cpp': 'abc' });
        expect(events).toHaveLength(1);
    });
    it('rejects other JSON', () => {
        expect(() => parseTrace('{"events":[]}')).toThrow(/Not a SkePU trace/);
        expect(() => parseTrace('3')).toThrow(/Not a SkePU trace/);
    });
});

describe('transferKind', () => {
    const kind = label => A.transferKind({ label });
    it('tells the transfers of the library from those of containers', () => {
        expect(kind('Internal (reduce-temp)')).toBe('reduce-temp');
        expect(kind('Internal (overlap-wrap)')).toBe('overlap-wrap');
        expect(kind('[Internal PRNG]')).toBe('prng');
        expect(kind('Internal something')).toBe('internal');
        expect(kind('Split train data X')).toBe('container');
    });
});

describe('queries on a small trace', () => {
    const events = fixture('map_chain');
    it('groups the time of the skeleton calls', () => {
        const r = A.timeBy(events, 'label');
        expect(r.total_us).toBe(300);
        expect(r.rows.map(x => x.key)).toEqual(['map1', 'map2', 'map3']);
        expect(r.rows[0]).toMatchObject({ count: 1, total_us: 100, mean_us: 100, share: 0.3333 });
    });
    it('rejects an unknown grouping', () => {
        expect(() => A.timeBy(events, 'colour')).toThrow(/group_by/);
    });
    it('finds events and pages them', () => {
        const all = A.findEvents(events, { type: 'skeleton_call', limit: 2 });
        expect(all).toMatchObject({ total: 3, returned: 2, offset: 0 });
        expect(all.events[0]).toMatchObject({ id: 'e4', label: 'map1', duration_us: 100 });
        expect(A.findEvents(events, { label: 'MAP', offset: 2 }).events.map(e => e.label)).toEqual(['map3']);
        expect(A.findEvents(events, { min_duration_us: 101 }).total).toBe(0);
    });
    it('counts how the containers are read and written', () => {
        const c = A.containers(events).containers.find(x => x.object_id === 1002);
        expect(c).toMatchObject({ label: 'B', allocations: 1, reads: 1, writes: 1 });
    });
});

describe('critical path', () => {
    it('is the whole chain of a chain', () => {
        const r = criticalPath(fixture('map_chain'));
        expect(r).toMatchObject({ total_us: 300, work_us: 300, parallelism: 1 });
        expect(r.path.map(n => n.label)).toEqual(['map1', 'map2', 'map3']);
    });
    it('runs the two branches of a fan-out in parallel', () => {
        const r = criticalPath(fixture('fanout'));
        expect(r).toMatchObject({ total_us: 200, work_us: 300, parallelism: 1.5 });
        expect(r.path.map(n => n.label)[0]).toBe('map1');
        expect(r.path).toHaveLength(2);
    });
});

// The small CNN of examples/dnn_cifar10 on OpenCL (examples/dnn_cifar10/trace_dnn_cifar10_opencl.json).
describe('the DNN CIFAR-10 trace (OpenCL)', () => {
    const { events, header } = load('examples/dnn_cifar10/trace_dnn_cifar10_opencl.json');

    it('has the events of the trace', () => {
        const s = A.summarize(events, header);
        expect(s.events).toBe(812);
        expect(s.by_type).toMatchObject({ skeleton_call: 424, transfer: 116, region: 42 });
        expect(s.backends).toEqual(['OpenCL']);
        expect(s.skeleton_calls).toEqual({ count: 424, total_us: 245525 });
    });

    it('splits the transfers by kind, the library\'s own being most of the time', () => {
        const t = A.transfers(events, { limit: 5 });
        expect(t.count).toBe(116);
        expect(t.by_kind.container).toMatchObject({ count: 32, total_us: 6406 });
        expect(t.by_kind['reduce-temp']).toMatchObject({ count: 24, total_us: 5299, device_to_host: 24 });
        expect(t.by_kind['overlap-wrap']).toMatchObject({ count: 57, total_us: 9740, host_to_device: 57 });
        expect(t.by_kind.prng).toMatchObject({ count: 3, total_us: 2735 });
        expect(t.total_us).toBe(24180);
    });

    it('attributes the library\'s transfers to the skeleton calls that cause them', () => {
        const t = A.transfers(events);
        expect(t.internal_by_owner_pattern['Reduce']).toMatchObject({ count: 24 });
        expect(t.internal_by_owner_pattern['MapOverlap 4D']).toMatchObject({ count: 57 });
        expect(t.internal_by_owner_pattern['Map']).toMatchObject({ count: 3 });
        expect(t.internal_by_owner_pattern.unattributed).toBeUndefined();
    });

    it('attributes the time to the patterns', () => {
        const r = A.timeBy(events, 'pattern');
        expect(r.rows.map(x => [x.key, x.count])).toEqual([['Map', 343], ['MapOverlap 4D', 57], ['Reduce', 24]]);
        expect(r.rows[0].share).toBeCloseTo(0.8705, 4);
    });

    it('finds a path of a time that the work could not be done faster than', () => {
        const r = criticalPath(events);
        expect(r.work_us).toBe(245525);
        expect(r.total_us).toBeGreaterThan(100000);
        expect(r.total_us).toBeLessThan(r.work_us);
        expect(r.parallelism).toBeGreaterThan(1);
        const shares = r.path.reduce((s, n) => s + n.share, 0);
        expect(shares).toBeLessThanOrEqual(1.0001);
    });

    it('does not change the events', () => {
        const before = JSON.stringify(events);
        criticalPath(events); A.transfers(events); A.containers(events);
        expect(JSON.stringify(events)).toBe(before);
    });
});
