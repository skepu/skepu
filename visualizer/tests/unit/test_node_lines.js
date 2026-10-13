// test_node_lines.js — the lines that a node of the graph marks in a file (src/nodeLines.ts), which the extension puts badges at. The cases are
// those of real traces: an `external` made without C++20 has no file and line 0 (it once made every file "match" and the editor throw).
import { describe, it, expect } from 'vitest';
import { linesOfNode, pathEndsWith, traceSuffix } from '../../src/nodeLines.ts';

const FILE = '/Users/me/skepu/examples/dotproduct.cpp';
const call = { id: 'n1', type: 'skeleton_call', file: '/examples/dotproduct.cpp', line: 39 };

describe('the lines of a node in a source file', () => {
    it('are the line of the call (from 0) when the call is in the file', () => {
        expect(linesOfNode(call, FILE, false, [])).toEqual([38, 38]);
        expect(linesOfNode({ ...call, type: 'external' }, FILE, false, [])).toEqual([38, 38]);
        expect(linesOfNode({ ...call, file: FILE }, FILE, false, [])).toEqual([38, 38]);
    });

    it('are none for an event without a file or a line (external without C++20: file "" and line 0)', () => {
        expect(linesOfNode({ id: 'e', type: 'external', file: '', line: 0 }, FILE, false, [])).toBeNull();
        expect(linesOfNode({ id: 'e', type: 'external', line: 0 }, FILE, false, [])).toBeNull();
        expect(linesOfNode({ ...call, line: 0 }, FILE, false, [])).toBeNull();
        expect(linesOfNode({ ...call, line: -1 }, FILE, false, [])).toBeNull();
        expect(linesOfNode({ ...call, line: undefined }, FILE, false, [])).toBeNull();
    });

    it('are none for the calls of another file and for nodes that are not calls', () => {
        expect(linesOfNode({ ...call, file: '/examples/other.cpp' }, FILE, false, [])).toBeNull();
        expect(linesOfNode({ ...call, type: 'allocation' }, FILE, false, [])).toBeNull();
        expect(linesOfNode({ ...call, type: 'container_update' }, FILE, false, [])).toBeNull();
    });

    it('never has a negative line, for any node', () => {
        const nodes = [{ type: 'skeleton_call', file: '', line: 0 }, { type: 'external', file: FILE, line: -3 },
                       { type: 'skeleton_call', file: FILE, line: 1 }, { type: 'region' }];
        for (const n of nodes) {
            const lines = linesOfNode({ id: 'x', ...n }, FILE, false, []);
            if (lines) expect(Math.min(...lines)).toBeGreaterThanOrEqual(0);
        }
    });
});

describe('the lines of a node in the file of the trace', () => {
    it('are the lines of its event', () => {
        expect(linesOfNode({ id: 'n', type: 'skeleton_call', trace_index: 1 }, '/t.json', true, [[1, 8], [9, 20]])).toEqual([8, 19]);
    });
    it('are none for a node that has no event in the file', () => {
        expect(linesOfNode({ id: 'n', type: 'skeleton_call' }, '/t.json', true, [[1, 8]])).toBeNull();
        expect(linesOfNode({ id: 'n', type: 'skeleton_call', trace_index: 7 }, '/t.json', true, [[1, 8]])).toBeNull();
    });
});

describe('the path of a file against the end of the path in the trace', () => {
    it('ends at the start of a name', () => {
        expect(pathEndsWith('/a/b/foo.cpp', '/b/foo.cpp')).toBe(true);
        expect(pathEndsWith('/a/b/foo.cpp', 'foo.cpp')).toBe(true);
        expect(pathEndsWith('/a/b/foo.cpp', '/a/b/foo.cpp')).toBe(true);
        expect(pathEndsWith('/a/b/foo.cpp', 'o.cpp')).toBe(false);
        expect(pathEndsWith('/a/b/foo.cpp', '')).toBe(false);
        expect(pathEndsWith('C:\\a\\b\\foo.cpp', '/b/foo.cpp')).toBe(true);
    });
});

describe('the path of a file as the compiler had it', () => {
    it('is made comparable: doubled and . parts go, and so do the .. before a relative path', () => {
        expect(traceSuffix('.//..//skepu-headers/src/skepu-lib/io.hpp')).toBe('skepu-headers/src/skepu-lib/io.hpp');
        expect(traceSuffix('../../include/x.hpp')).toBe('include/x.hpp');
        expect(traceSuffix('a/b/../c.hpp')).toBe('a/c.hpp');
        expect(traceSuffix('/Users/me/skepu//examples/./dotproduct.cpp')).toBe('/Users/me/skepu/examples/dotproduct.cpp');
        expect(traceSuffix('C:\\src\\skepu\\io.hpp')).toBe('C:/src/skepu/io.hpp');
    });

    it('is the file of a node in a library header found by a relative include (it was never matched)', () => {
        const data = { id: 'n', type: 'external', file: './/..//skepu-headers/src/skepu-lib/io.hpp', line: 54 };
        expect(linesOfNode(data, '/Users/me/skepu-alpha/skepu-headers/src/skepu-lib/io.hpp', false, [])).toEqual([53, 53]);
        expect(linesOfNode(data, '/Users/me/skepu-alpha/visualizer/static/skepu-lib/io.hpp', false, [])).toBeNull();
        expect(linesOfNode(data, 'C:\\src\\skepu-headers\\src\\skepu-lib\\io.hpp', false, [])).toEqual([53, 53]);
    });
});
