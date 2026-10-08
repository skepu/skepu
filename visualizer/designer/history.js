// history.js — undo and redo of the design: the states of it that a person made, as the text (JSON) of the design after each change. No DOM.
//
// The design is all of the state (everything else is made from it), so a step is just the design before it. A change is a step when it changes
// the design; not every event is one: a text field makes its change when it is left (not for each key), a drag when it ends (not for each
// pixel), and what one action does in several calls of `changed` (they come in the same task) is one step. What comes in a quick series of
// the same kind of change (the keys in the body of a function, the nodes that one drag moves) is one step too: `merge` names the kind.

export class History {
    /**
     * @param text   the design to start from (its JSON)
     * @param limit  the most steps to keep
     * @param now    the clock, in milliseconds (a parameter for the tests)
     */
    constructor(text, limit = 100, now = () => performance.now()) {
        this.limit = limit;
        this.now = now;
        this.reset(text);
    }

    reset(text) {
        this.present = text;
        this.past = [];
        this.future = [];
        this.merging = null;           // { merge, at }: the kind and time of the last change, for the next one to be joined to it
        this.task = false;             // the last change was in this task
    }

    /**
     * The design is `text` now. If it is not what it was, the one before it is a step, unless the change is joined to the last one: it is in
     * the same task, or has the same `merge` as the last and comes within `window` milliseconds of it. Returns true if a step was made.
     */
    commit(text, { merge = null, window = 1000 } = {}) {
        if (text === this.present) return false;
        const t = this.now();
        const joined = this.task || (merge !== null && this.merging?.merge === merge && t - this.merging.at < window);
        if (!joined) {
            this.past.push(this.present);
            while (this.past.length > this.limit) this.past.shift();
        }
        this.present = text;
        this.future = [];
        this.merging = { merge, at: t };
        if (!this.task) { this.task = true; queueMicrotask(() => { this.task = false; }); }
        return !joined;
    }

    canUndo() { return this.past.length > 0; }
    canRedo() { return this.future.length > 0; }

    /** The design before the last step (its JSON), or null. */
    undo() {
        if (!this.past.length) return null;
        this.future.push(this.present);
        this.present = this.past.pop();
        this.merging = null;
        return this.present;
    }

    /** The design after the step that was undone, or null. */
    redo() {
        if (!this.future.length) return null;
        this.past.push(this.present);
        this.present = this.future.pop();
        this.merging = null;
        return this.present;
    }
}
