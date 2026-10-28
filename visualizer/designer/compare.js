// compare.js — round-trip check: compares the design's calls, containers and loops with the skeleton calls and regions of the trace of its program.
'use strict';
import { skeletonTypeName, isAnonymous } from './codegen.js';

const expandSteps = (steps, out = []) => {
    for (const step of steps) {
        if (step.type === 'call') out.push(step.call, ...(step.call.pattern === 'MapOverlap' && step.call.skeleton.params.update === 'RedBlack' ? [step.call] : []));          // RedBlack: a pass for each colour
        else for (let i = 0; i < step.count; i++) expandSteps(step.steps, out);
    }
    return out;
};

const base = pattern => String(pattern ?? '').split(' ')[0];

/** The differences between a lowered program and the events of its trace (empty: they agree). `options`: the code options of the program. */
export function compareTrace(program, events, options = {}) {
    const problems = [];
    const label = new Map();
    for (const e of events) if (e.type === 'allocation') label.set(e.object_id, e.label);
    const names = ids => (ids ?? []).map(id => label.get(id)).filter(Boolean);

    const everything = expandSteps(program.steps);
    const expected = everything.filter(c => !c.host);
    const calls = events.filter(e => e.type === 'skeleton_call');
    if (calls.length !== expected.length) problems.push(`the trace has ${calls.length} calls, the design ${expected.length}`);
    expected.forEach((call, i) => {
        const e = calls[i];
        if (!e) return;
        const at = `call ${i + 1} (${call.name})`;
        if (isAnonymous(call.skeleton, options) ? !e.label.startsWith(`${skeletonTypeName(call.skeleton, call)}@`) : e.label !== call.name) problems.push(`${at}: the trace has ${e.label}`);
        if (base(e.pattern) !== call.pattern) problems.push(`${at}: the pattern is ${e.pattern}, not ${call.pattern}`);
        const wantIn = call.pattern === 'MapPairs' || call.pattern === 'MapPairsReduce' ? [call.vertical.label, call.horizontal.label] : call.elwise.map(c => c.label);
        if (names(e.elwise_inputs).join() !== wantIn.join()) problems.push(`${at}: it reads ${names(e.elwise_inputs).join(', ')}, the design ${wantIn.join(', ')}`);
        const wantProxy = call.proxies.map(c => c.label);
        if (names(e.proxy_inputs).join() !== wantProxy.join()) problems.push(`${at}: its proxies are ${names(e.proxy_inputs).join(', ')}, the design ${wantProxy.join(', ')}`);
        if (call.out && names(e.outputs).join() !== call.out.label) problems.push(`${at}: it writes ${names(e.outputs).join(', ')}, the design ${call.out.label}`);
        if (call.random && e.prng == null) problems.push(`${at}: it draws random numbers, and the trace has no random stream`);
    });

    // external blocks: the external events with the design's labels
    const hosts = everything.filter(c => c.host && !c.io), hostLabels = new Set(hosts.map(c => c.skeleton.label));
    const externals = events.filter(e => e.type === 'external' && hostLabels.has(e.label));
    if (externals.length !== hosts.length) problems.push(`${externals.length} external blocks in the trace, ${hosts.length} in the design`);
    hosts.forEach((call, i) => {
        const e = externals[i];
        if (!e) return;
        const at = `external ${i + 1} (${call.skeleton.label})`;
        if (e.label !== call.skeleton.label) problems.push(`${at}: the trace has ${e.label}`);
        if (names(e.proxy_inputs).join() !== call.reads.map(c => c.label).join()) problems.push(`${at}: it reads ${names(e.proxy_inputs).join(", ")}, the design ${call.reads.map(c => c.label).join(', ')}`);
        if (names(e.outputs).join() !== call.writes.map(c => c.label).join()) problems.push(`${at}: it writes ${names(e.outputs).join(', ')}, the design ${call.writes.map(c => c.label).join(', ')}`);
    });

    // I/O blocks (skepu::io::cout): one event per printed container
    const printed = everything.filter(c => c.io).flatMap(c => c.reads.map(r => r.label));
    const ios = events.filter(e => e.type === 'external' && e.label === 'I/O' && (e.proxy_inputs ?? []).length === 1 && label.has(e.proxy_inputs[0])).map(e => label.get(e.proxy_inputs[0]));
    if (printed.join() !== ios.join()) problems.push(`the I/O blocks print ${ios.join(', ') || 'nothing'}, the design ${printed.join(', ') || 'nothing'}`);

    // loop iterations: one region per iteration (an inner loop: per iteration of the enclosing loops)
    for (const step of program.steps) {
        if (step.type !== 'loop') continue;
        const n = events.filter(e => e.type === 'region' && e.label === step.label).length;
        if (n !== step.count) problems.push(`loop ${step.label}: ${n} regions in the trace, ${step.count} iterations in the design`);
    }
    return problems;
}
