// test_node_lines.js — the lines that a node of the graph marks in a file (src/nodeLines.ts), which the extension puts badges at. The cases are
// those of real traces: an `external` made without C++20 has no file and line 0 (it once made every file "match" and the editor throw).
import { describe, it, expect } from 'vitest';
import { linesOfNode, pathEndsWith, traceSuffix, lensesOfBadges, lensTitle, nextNodeIndex, SOURCE_NODE_TYPES, nodesWithBadges } from '../../src/nodeLines.ts';
import { SOURCE_TYPES, badgeNodesToShow } from '../../script/badges.js';

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

describe('the lenses of a file', () => {
    const badges = [
        { nodeID: 'a', line: 4, count: 3 },
        { nodeID: 'b', line: 9, count: 1 },
        { nodeID: 'c', line: 4, count: 7 },
    ];

    it('are one for a line, with its nodes in the order of the graph', () => {
        expect(lensesOfBadges(badges)).toEqual([
            { line: 4, nodeIDs: ['a', 'c'], count: 3 },
            { line: 9, nodeIDs: ['b'], count: 1 },
        ]);
    });

    it('say the count for one node and the number of nodes for several', () => {
        const [two, one] = lensesOfBadges(badges);
        expect(lensTitle(one)).toBe('1 | Highlight Node');
        expect(lensTitle(two)).toBe('2 nodes | Highlight Next');
    });

    it('select the nodes of a line one after the other, and start again', () => {
        expect(nextNodeIndex(undefined, 3)).toBe(0);
        expect(nextNodeIndex(0, 3)).toBe(1);
        expect(nextNodeIndex(2, 3)).toBe(0);
        expect(nextNodeIndex(undefined, 1)).toBe(0);
        expect(nextNodeIndex(0, 1)).toBe(0);
    });
});


describe('the lines of the nodes that the viewer puts badges at', () => {
    it('are those of the snapshots and the regions, as of the calls', () => {
        expect(linesOfNode({ ...call, type: 'snapshot', line: 12 }, FILE, false, [])).toEqual([11, 11]);
        expect(linesOfNode({ ...call, type: 'region', line: 20 }, FILE, false, [])).toEqual([19, 19]);
        expect(linesOfNode({ ...call, type: 'snapshot', file: '/examples/other.cpp' }, FILE, false, [])).toBeNull();
        expect(linesOfNode({ ...call, type: 'snapshot', file: '', line: 0 }, FILE, false, [])).toBeNull(); // (a snapshot made without C++20)
    });

    it('are of the same types as the badges of the viewer (script/badges.js), and the regions are left out by the same rule', () => {
        expect([...SOURCE_NODE_TYPES].sort()).toEqual([...SOURCE_TYPES].sort());
        const nodes = [{ data: { type: 'region' } }, { data: { type: 'skeleton_call' } }, { data: { type: 'snapshot' } }, { data: { type: 'allocation' } }];
        for (const shown of [true, false])
            expect(nodesWithBadges(nodes, shown)).toEqual(badgeNodesToShow(nodes, shown));
    });
});
