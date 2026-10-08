// test_history.js — the steps of undo and redo of the designer (designer/history.js): the designs that a person made.
import { describe, it, expect } from 'vitest';
import { History } from '../../designer/history.js';

const tick = () => Promise.resolve();            // (the changes of one task are one step: a microtask ends the task)

describe('the history of the designer', () => {
    it('has a step for each change of the design, and goes back and forth through them', async () => {
        const h = new History('a');
        h.commit('b'); await tick();
        h.commit('c'); await tick();
        expect([h.canUndo(), h.canRedo()]).toEqual([true, false]);
        expect([h.undo(), h.undo(), h.undo()]).toEqual(['b', 'a', null]);
        expect(h.canUndo()).toBe(false);
        expect([h.redo(), h.redo(), h.redo()]).toEqual(['b', 'c', null]);
    });

    it('has no step for a change that changes nothing, and forgets what was undone when something new is made', async () => {
        const h = new History('a');
        expect(h.commit('a')).toBe(false);
        expect(h.canUndo()).toBe(false);
        h.commit('b'); await tick();
        h.undo();
        expect(h.canRedo()).toBe(true);
        h.commit('x'); await tick();
        expect([h.canRedo(), h.undo()]).toEqual([false, 'a']);
    });

    it('makes one step of what is done in several calls in the same task', async () => {
        const h = new History('a');
        h.commit('b'); h.commit('c'); h.commit('d');          // (an action that calls changed more than once)
        await tick();
        expect(h.undo()).toBe('a');
        expect(h.canUndo()).toBe(false);
    });

    it('joins a quick series of the same kind of change (the keys of a text, the nodes of a drag), not a slow one or another kind', async () => {
        let t = 0;
        const h = new History('a', 100, () => t);
        for (const [text, at] of [['k1', 0], ['k12', 200], ['k123', 500]]) { t = at; h.commit(text, { merge: 'body:s1' }); await tick(); }
        expect(h.past).toEqual(['a']);                                   // one step for the three
        t = 3000; h.commit('k1234', { merge: 'body:s1' }); await tick();      // (a pause: a step of its own)
        expect(h.past).toEqual(['a', 'k123']);
        t = 3100; h.commit('moved', { merge: 'move' }); await tick();         // (another kind)
        expect(h.past).toEqual(['a', 'k123', 'k1234']);
        t = 3200; h.commit('moved again', { merge: 'move', window: 400 }); await tick();
        expect(h.past).toEqual(['a', 'k123', 'k1234']);
        expect(h.undo()).toBe('k1234');
    });

    it('does not join a change that has no kind, to the last one that had one', async () => {
        let t = 0;
        const h = new History('a', 100, () => t);
        h.commit('b', { merge: 'move' }); await tick();
        t = 10; h.commit('c'); await tick();
        expect(h.past).toEqual(['a', 'b']);
    });

    it('keeps the last steps only', async () => {
        const h = new History('0', 3);
        for (let i = 1; i <= 6; i++) { h.commit(String(i)); await tick(); }
        expect(h.past).toEqual(['3', '4', '5']);
        expect([h.undo(), h.undo(), h.undo(), h.undo()]).toEqual(['5', '4', '3', null]);
    });

    it('can be started again with a design: nothing to undo', async () => {
        const h = new History('a');
        h.commit('b'); await tick();
        h.reset('z');
        expect([h.canUndo(), h.canRedo(), h.present]).toEqual([false, false, 'z']);
    });
});
