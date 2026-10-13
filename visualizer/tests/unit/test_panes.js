// test_panes.js — the panes that open and close (panes.js): a closed pane has the class pane-closed; one that the user opens or closes slides (its
// size limit goes from 0 to the size it has, or the other way, with the padding and the border) and is a plain class change when the page is made.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { setPaneOpen, isPaneOpen, PANE_SLIDE_MS } from '../../script/panes.js';

function pane({ scrollHeight = 120, offsetHeight = 120, offsetWidth = 380, closed = false } = {}) {
    const classes = new Set(closed ? ['pane-closed'] : []);
    return {
        scrollHeight, offsetHeight, offsetWidth, style: {},
        classList: {
            add: c => classes.add(c), remove: c => classes.delete(c), contains: c => classes.has(c),
            toggle: (c, on) => { if (on === undefined ? !classes.has(c) : on) classes.add(c); else classes.delete(c); },
        },
        classes,
    };
}

// the inline sizes that are set (a property that is set to '' is none)
const inline = el => Object.fromEntries(Object.entries(el.style).filter(([, v]) => v !== ''));

beforeEach(() => { vi.useFakeTimers(); });
afterEach(() => { vi.useRealTimers(); delete globalThis.matchMedia; });

describe('a pane that is put in place (the page is made, the settings are restored)', () => {
    it('is closed or opened by the class alone, with no sliding', () => {
        const el = pane();
        setPaneOpen(el, false, 'height', false);
        expect([...el.classes]).toEqual(['pane-closed']);
        expect(inline(el)).toEqual({});
        setPaneOpen(el, true, 'height', false);
        expect([...el.classes]).toEqual([]);
        expect(inline(el)).toEqual({});
    });

    it('does nothing to a pane that is already as it is to be', () => {
        const el = pane();
        setPaneOpen(el, true, 'height', true);
        expect([...el.classes]).toEqual([]);
        expect(inline(el)).toEqual({});
    });
});

describe('a pane that the user opens', () => {
    it('slides from nothing to the size it has: the class goes at once, the limit is 0 and then the size, and everything is clean when it is done', () => {
        const el = pane({ closed: true, scrollHeight: 120 });
        const seen = [];
        // (what the limit is when the browser first sees the pane open: this is where the transition starts from)
        Object.defineProperty(el, 'offsetWidth', { get() { seen.push(el.style.maxHeight); return 380; } });
        setPaneOpen(el, true, 'height', true);
        expect(el.classes.has('pane-closed')).toBe(false);
        expect(el.classes.has('pane-sliding')).toBe(true);
        expect(seen).toEqual(['0px']);                                        // (the reflow came while the limit was 0)
        expect(el.style.maxHeight).toBe('120px');                             // (and then it is the size that it has)
        expect(el.style.paddingTop).toBe('');                                 // (the padding is its own again: it comes back with the transition)
        vi.advanceTimersByTime(PANE_SLIDE_MS + 60);
        expect(el.classes.has('pane-sliding')).toBe(false);
        expect(inline(el)).toEqual({});
    });

    it('is a pane that slides in its width for the source pane: the size is its width', () => {
        const el = pane({ closed: true, offsetWidth: 420 });
        setPaneOpen(el, true, 'width', true);
        expect(el.style.maxWidth).toBe('420px');
        expect(el.classes.has('pane-sliding')).toBe(true);
    });
});

describe('a pane that the user closes', () => {
    it('slides to nothing and is closed (display none) only when that is done', () => {
        const el = pane({ offsetHeight: 90 });
        setPaneOpen(el, false, 'height', true);
        expect(el.classes.has('pane-closed')).toBe(false);                    // (still there: it is sliding)
        expect(isPaneOpen(el)).toBe(false);                                   // (but it is not open: it is going)
        expect(el.style.maxHeight).toBe('0px');                               // (from 90px, the size it had)
        expect([el.style.paddingTop, el.style.paddingBottom, el.style.borderTopWidth]).toEqual(['0px', '0px', '0px']);
        vi.advanceTimersByTime(PANE_SLIDE_MS + 60);
        expect(el.classes.has('pane-closed')).toBe(true);
        expect(el.classes.has('pane-sliding')).toBe(false);
        expect(el.style.maxHeight).toBe('');                                  // (and its sizes are its own again, for the next time)
        expect(isPaneOpen(el)).toBe(false);
    });

    it('starts the slide from the size that the pane has now (the first style is the limit that is where it is)', () => {
        const el = pane({ offsetHeight: 90 });
        const limits = [];
        Object.defineProperty(el, 'offsetWidth', { get() { limits.push(el.style.maxHeight); return 380; } });
        setPaneOpen(el, false, 'height', true);
        expect(limits).toEqual(['90px']);
    });
});

describe('a slide that is not finished', () => {
    it('is ended by the next change of the pane, which then goes on from the pane as it was to be', () => {
        const el = pane();
        setPaneOpen(el, false, 'height', true);          // (closing)
        setPaneOpen(el, true, 'height', true);           // (the user changes their mind: open again)
        expect(el.classes.has('pane-closed')).toBe(false);
        expect(isPaneOpen(el)).toBe(true);
        vi.advanceTimersByTime(PANE_SLIDE_MS * 3);
        expect(el.classes.has('pane-closed')).toBe(false);                    // (the closing that was waiting did not close it)
        expect(el.style.maxHeight).toBe('');
    });

    it('does not leave a closing that is waiting to close a pane that was opened since', () => {
        const el = pane({ closed: true });
        setPaneOpen(el, true, 'height', true);
        setPaneOpen(el, false, 'height', true);
        setPaneOpen(el, true, 'height', false);
        vi.advanceTimersByTime(PANE_SLIDE_MS * 3);
        expect(isPaneOpen(el)).toBe(true);
    });
});

describe('a user who has asked for less motion', () => {
    it('gets the panes put in place at once also when they open and close', () => {
        globalThis.matchMedia = () => ({ matches: true });
        const el = pane({ closed: true });
        setPaneOpen(el, true, 'height', true);
        expect([...el.classes]).toEqual([]);
        expect(inline(el)).toEqual({});
    });
});
