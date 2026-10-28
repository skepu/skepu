// history.js — undo and redo: the design (as JSON) after each change. No DOM.
//
// A change is a step if it alters the design. Changes made by one action in the same task form one step; so do rapid changes of one kind (`merge`):
// text fields commit on blur, a drag on release, keys in a function body together.

export class History {
    /**
     * @param text   the design to start from (its JSON)
     * @param limit  the most steps to keep
     * @param now    the clock in milliseconds (injectable for tests)
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
        this.merging = null;           // { merge, at }: kind and time of the last change
        this.task = false;             // the last change was in the current task
    }

    /**
     * The design is `text` now. If it changed, the previous design becomes a step, unless the change is joined to the last: same task, or same `merge`
     * within `window` milliseconds. Returns true if a step was made.
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

    /** The design before the last step (JSON), or null. */
    undo() {
        if (!this.past.length) return null;
        this.future.push(this.present);
        this.present = this.past.pop();
        this.merging = null;
        return this.present;
    }

    /** The design after the undone step, or null. */
    redo() {
        if (!this.future.length) return null;
        this.past.push(this.present);
        this.present = this.future.pop();
        this.merging = null;
        return this.present;
    }
}
