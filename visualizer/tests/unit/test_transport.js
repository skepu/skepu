// test_transport.js — the replies of the worker: the id that pairs a reply with its request is not in the result (the info pane shows every key of the result).
import { describe, it, expect } from 'vitest';
import { initWorkerBackend, fetchGetData, workerCall } from '../../script/transport.js';

// A worker that answers each message with { id, type: 'result', ...result }, as backend/visualizer.js does
function fakeWorker(answer) {
    const listeners = [];
    return {
        addEventListener(type, listener) { listeners.push(listener); },
        postMessage(message) {
            const { id, type, ...payload } = message;
            Promise.resolve().then(() => listeners.forEach(l => l({ data: { id, type: 'result', ...answer(type, payload) } })));
        },
    };
}

describe('the replies of the worker', () => {
    const worker = fakeWorker((type, payload) => type === 'getData' ? { type: 'skeleton_call', Pattern: 'Map', node: payload.nodeId } : { value: 7 });
    initWorkerBackend(worker);

    it('have no id', async () => {
        const first = await workerCall('anything');
        const second = await workerCall('anything');
        expect(first).not.toHaveProperty('id');
        expect(second).not.toHaveProperty('id');
        expect(first.value).toBe(7);
    });

    it('keep the type that the result has (the type of a node)', async () => {
        const data = await fetchGetData('n1');
        expect(data).toEqual({ type: 'skeleton_call', Pattern: 'Map', node: 'n1' });
    });
});
