// event_builders.js — small traces written by hand: allocations, skeleton calls, and the events that
// observe or free a container. Each builder returns an event; trace() numbers them as parseTrace() does.
import { parseTrace } from '../script/backend/trace.js';

export function builder() {
    let t = 0;
    const span = d => { const start = t; t += d; return { start, end: t }; };
    return {
        alloc:   (id, label) => ({ type: 'allocation', label, object_id: id, time: t }),
        dealloc: (id, label) => ({ type: 'deallocation', label, object_id: id, time: t }),
        call: (label, { pattern = 'Map', n = 100, elements, ins = [], proxy = [], scalars = [], outs = [], backend = 'OpenMP', prng, params, dur = 100 } = {}) => ({
            type: 'skeleton_call', label, pattern, elements: elements ?? [n], elwise_inputs: ins, proxy_inputs: proxy, scalar_inputs: scalars,
            outputs: outs, backend, ...(prng != null ? { prng } : {}), ...(params ? { parameters: params } : {}), ...span(dur), file: 'x.cpp', line: 1,
        }),
        external: (label, { scalars = [], ins = [] } = {}) => ({
            type: 'external', label, elwise_inputs: ins, proxy_inputs: [], scalar_inputs: scalars, outputs: [], ...span(1), file: '', line: 2,
        }),
        copy:     (source, id, label = 'copy') => ({ type: 'copy', label, source, object_id: id, elements: 100, ...span(3) }),
        copy:     (source, id, label = 'copy') => ({ type: 'copy', label, source, object_id: id, elements: 100, ...span(3) }),
        access:   (id, source) => ({ type: 'element_access', label: 'x', object_id: id, source, index: 0, time: t }),
        snapshot: id => ({ type: 'snapshot', object_id: id, data_file: 's.txt', time: t }),
        // a collective operation of an MPI program that moves the containers ids (seq: the call it is in; phase: before or after its work)
        comm:     (ids, { operation = 'allgather', reason = 'host access: the container is gathered', seq = 0, phase, comm_seq = 1 } = {}) => ({
            type: 'communication', operation, reason, object_ids: ids, seq, comm_seq, ...(phase ? { phase } : {}),
            bytes_sent: 64, bytes_received: 32, ranks: 4, ...span(4) }),
        transfer: (id, label = 'T', direction = 'device-to-host', backend = 'OpenCL') => ({ type: 'transfer', label, object_id: id, direction, backend, elements: 100, ...span(5) }),
    };
}

/** Events in the order given; the builder `b` must be the one that made them (its clock is shared). */
export const trace = events => parseTrace(JSON.stringify(events)).events;
