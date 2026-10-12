// test_definitions.js — where the instance of a call and its user functions are defined: the call nodes have it from the table of the
// instances in the trace header (infoData), and the info pane has a link for each place (definitionRows).
import { describe, it, expect } from 'vitest';

import { buildDisplayGraph } from '../../script/backend/pipeline.js';
import { headerInstances } from '../../script/backend/trace.js';
import { definitionRows } from '../../script/InfoView.js';
import { collectBadges, definitionMarks, definitionPlace, sitesAt } from '../../script/badges.js';
import { builder, trace } from '../event_builders.js';

const table = {
    '7': { label: 'dotp', instance: { file: '/src/a.cpp', line: 12, end_line: 13 }, user_functions: [
        { name: 'mult', kind: 'function', file: '/src/a.cpp', line: 4, end_line: 6 }, { name: '', kind: 'lambda', file: '/src/a.cpp', line: 12 }] },
    '8': { label: 'Map@a.cpp:30', instance: { file: '/src/a.cpp', line: 30, anonymous: true }, user_functions: [
        { name: 'inc', kind: 'function', file: '/src/b.hpp', line: 2, end_line: 2 }] },
};

function callNodes(instances) {
    const b = builder();
    const events = trace([
        b.alloc(1, 'A'),
        { ...b.call('dotp', { outs: [1] }), object_id: 7 },
        { ...b.call('Map@a.cpp:30', { ins: [1] }), object_id: 8 },
        { ...b.call('other', { ins: [1] }), object_id: 99 },
    ]);
    const { graph } = buildDisplayGraph(events, { instances });
    return Object.fromEntries(graph.getAllNodes().filter(n => n.type === 'skeleton_call').map(n => [n.label, n]));
}

describe('the definitions in the data of a call node', () => {
    const nodes = callNodes(headerInstances({ instances: table }));

    it('a call has the entry of its instance', () => {
        const d = nodes.dotp.infoData().definitions;
        expect(d.label).toBe('dotp');
        expect(d.instance.line).toBe(12);
        expect(d.user_functions.map(f => f.kind)).toEqual(['function', 'lambda']);
    });

    it('a call of an instance that the table does not have, or a trace without a table, has none', () => {
        expect(nodes.other.infoData().definitions).toBeUndefined();
        expect(callNodes(undefined).dotp.infoData().definitions).toBeUndefined();
    });
});

describe('the links in the info pane', () => {
    const entry = headerInstances({ instances: table });

    it('are a row for the instance and one for each user function, with the lines', () => {
        const html = definitionRows(entry.get(7), () => true);
        expect(html.match(/class="info-row"/g)).toHaveLength(3);
        expect(html).toContain('Instance');
        expect(html).toContain('User function 1');
        expect(html).toContain('User function 2');
        expect(html).toContain('dotp @ a.cpp:12');
        expect(html).toContain('mult @ a.cpp:4');
        expect(html).toContain('lambda @ a.cpp:12');
        expect(html).toContain('data-file="/src/a.cpp" data-line="4" data-end-line="6"');
    });

    it('say that an instance is anonymous, and have one function without a number', () => {
        const html = definitionRows(entry.get(8), () => true);
        expect(html).toContain('(anonymous)');
        expect(html).toContain('User function</span>');
        expect(html).toContain('inc @ b.hpp:2');
    });

    it('are not links to a file that is not open', () => {
        const html = definitionRows(entry.get(8), file => file.endsWith('a.cpp'));
        expect(html.match(/<a /g)).toHaveLength(1);                 // the instance
        expect(html.match(/info-link-disabled/g)).toHaveLength(1);   // the function in b.hpp
    });

    it('are nothing without definitions', () => {
        expect(definitionRows(undefined, () => true)).toBe('');
    });
});

describe('the badge nodes and the sites of the definitions', () => {
    const b = builder();
    const events = trace([
        b.alloc(1, 'A'),
        { ...b.call('dotp', { outs: [1] }), object_id: 7 },
        { ...b.call('dotp', { ins: [1] }), object_id: 7 },      // (a second call of the instance)
        { ...b.call('Map@a.cpp:30', { ins: [1] }), object_id: 8 },
        { ...b.call('other', { ins: [1] }), object_id: 99 },
    ]);
    const { graph } = buildDisplayGraph(events, { instances: headerInstances({ instances: table }) });
    const [, badgeNodes] = graph.toTimelineAndBadges();
    const calls = badgeNodes.filter(n => n.data.type === 'skeleton_call');

    it('a badge node of a call has the places of its instance and user functions', () => {
        const dotp = calls.find(n => n.data.definitions?.instance.line === 12);
        expect(dotp.data.definitions.functions.map(f => f.line)).toEqual([4, 12]);
        expect(calls.find(n => n.data.definitions === undefined)).toBeDefined();   // the call of the instance without an entry
    });

    it('the sites are the places, each with the nodes that use it', () => {
        const { sites } = collectBadges(badgeNodes, []);
        const at = (role, line) => sites.find(s => s.role === role && s.line === line);
        expect(sites.map(s => `${s.role}:${s.line}`).sort()).toEqual(['function:12', 'function:2', 'function:4', 'instance:12', 'instance:30']);
        expect(at('instance', 12).ids).toHaveLength(2);                // (the two calls of the instance dotp: two nodes)
        expect(at('function', 4)).toMatchObject({ file: '/src/a.cpp', end_line: 6 });
        expect(at('instance', 30).ids).toHaveLength(1);
    });

    it('sitesAt finds the sites that cover a line of a file, by the basename', () => {
        const { sites } = collectBadges(badgeNodes, []);
        expect(sitesAt(sites, '/other/dir/a.cpp', 5).map(s => `${s.role}:${s.line}`)).toEqual(['function:4']);
        expect(sitesAt(sites, 'a.cpp', 12).map(s => s.role).sort()).toEqual(['function', 'instance']);   // (a lambda on the line of the instance)
        expect(sitesAt(sites, 'a.cpp', 20)).toEqual([]);
        expect(sitesAt(sites, 'b.hpp', 2).map(s => s.role)).toEqual(['function']);
    });
});

describe('the marks and the shortcuts of a call', () => {
    const d = headerInstances({ instances: table }).get(7);

    it('the marks are the lines of the instance and of each user function', () => {
        expect(definitionMarks(d)).toEqual([
            { role: 'instance', file: '/src/a.cpp', from: 12, to: 13 },
            { role: 'function', file: '/src/a.cpp', from: 4, to: 6 },
            { role: 'function', file: '/src/a.cpp', from: 12, to: 12 }]);
        expect(definitionMarks(null)).toEqual([]);
    });

    it('the instance, and the user functions one after the other', () => {
        expect(definitionPlace(d, 'instance').line).toBe(12);
        expect([0, 1, 2, 3].map(i => definitionPlace(d, 'function', i).line)).toEqual([4, 12, 4, 12]);   // (it wraps around)
        expect(definitionPlace({ ...d, user_functions: [] }, 'function')).toBeNull();
        expect(definitionPlace(undefined, 'instance')).toBeNull();
    });
});

// A data node that a call made (a container update, a scalar) has the source line and the definitions of the call in its info, so that selecting it marks the
// same lines in the source as selecting the call does.
describe('the info of the data nodes that a call made', () => {
    const ROOT = new URL('../..', import.meta.url).pathname;
    const join = (a, b) => a.replace(/\/$/, '') + '/' + b;
    const replies = {};
    let seq = 0;
    const call = async (type, payload = {}) => {
        const id = String(++seq);
        await globalThis.self.onmessage({ data: { id, type, ...payload } });
        if (replies[id].type === 'error') throw new Error(replies[id].error);
        return replies[id];
    };
    const header = { instances: { 7: { label: 'sum', instance: { file: '/p.cpp', line: 12, end_line: 12 }, user_functions: [{ name: 'add', kind: 'function', file: '/p.cpp', line: 4, end_line: 6 }] } } };

    it('are the file, the line and the definitions of the call, for an update and for a scalar', async () => {
        globalThis.self = { postMessage: m => { replies[m.id] = m; }, location: { href: 'file://' + join(ROOT, 'script/backend/visualizer.js') }, addEventListener() {} };
        await import(join(ROOT, 'script/backend/visualizer.js?datanodes'));
        const b = builder();
        const events = [
            b.alloc(1, 'v'), b.alloc(2, 'w'),
            { ...b.call('map', { ins: [1], outs: [2] }), object_id: 7, file: '/p.cpp', line: 30 },
            { type: 'scalar_value', object_id: 3, value: 4, time: 0 },
            { ...b.call('sum', { pattern: 'Reduce 1D-Full', ins: [2], outs: [3] }), object_id: 7, file: '/p.cpp', line: 31 },
        ];
        await call('load', { traceJson: JSON.stringify({ header, data: events }), traceFilename: 't.json' });
        const g = await call('graph', { show_regions: false });
        const infos = {};
        for (const node of g.nodes) {
            const data = node.data ?? node;
            if (data.type === 'container_update' || data.type === 'scalar' || data.type === 'skeleton_call')
                (infos[data.type] ??= []).push(await call('getData', { nodeId: data.id }));
        }
        const scalar = infos.scalar[0];
        expect(scalar).toMatchObject({ File: '/p.cpp', Line: 31 });
        expect(scalar.definitions.instance).toEqual({ file: '/p.cpp', line: 12, end_line: 12 });
        const updates = infos.container_update.filter(i => i.Line != null);
        expect(updates.map(i => i.Line).sort()).toEqual([30]);            // (the update that the map made; the container v has no call)
        expect(updates[0].definitions.user_functions[0].name).toBe('add');
    });
});
