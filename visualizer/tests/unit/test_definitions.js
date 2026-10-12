// test_definitions.js — where the instance of a call and its user functions are defined: the call nodes have it from the table of the
// instances in the trace header (infoData), and the info pane has a link for each place (definitionRows).
import { describe, it, expect } from 'vitest';

import { buildDisplayGraph } from '../../script/backend/pipeline.js';
import { headerInstances } from '../../script/backend/trace.js';
import { definitionRows } from '../../script/InfoView.js';
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
