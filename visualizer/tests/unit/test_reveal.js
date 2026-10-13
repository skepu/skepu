// test_reveal.js — from the splash to the viewer (reveal.js): the panes are in their places and transparent while the first trace is drawn, the
// welcome panel stays; when the first render has settled the splash fades out and the panes fade in, and the splash is hidden when that is done.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { holdForReveal, reveal, REVEAL_FADE_MS, REVEAL_MAX_WAIT_MS } from '../../script/reveal.js';

function element(classes = []) {
    const set = new Set(classes);
    return { classes: set, classList: { add: c => set.add(c), remove: c => set.delete(c), contains: c => set.has(c) } };
}
// the welcome panel: it has a place and a size in the layout (which the test changes after the panel is held, as the panes do when they take their places)
function welcomePanel(classes = [], box = { left: 0, top: 58, width: 1280, height: 600 }) {
    const el = element(classes);
    el.style = {};
    el.box = box;
    el.getBoundingClientRect = () => el.box;
    el.listeners = new Set();
    el.addEventListener = (type, f) => { if (type === 'transitionend') el.listeners.add(f); };
    el.removeEventListener = (type, f) => { el.listeners.delete(f); };
    el.endTransition = (propertyName = 'opacity', target = el) => [...el.listeners].forEach(f => f({ target, propertyName }));
    return el;
}
const fakeBody = () => ({ children: [], appendChild(el) { this.children.push(el); } });
const has = (el, ...names) => names.map(n => el.classes.has(n));

let html, panel, body;
beforeEach(() => { vi.useFakeTimers(); html = element(); panel = welcomePanel(); body = fakeBody(); });
afterEach(() => { reveal(html, panel); vi.useRealTimers(); });

describe('the first trace, with the splash on the page', () => {
    it('keeps the splash while the first trace is drawn', () => {
        expect(holdForReveal(html, panel, body)).toBe(true);
        expect(has(html, 'reveal-pending')).toEqual([true]);
        expect(has(panel, 'hidden', 'fading')).toEqual([false, false]);
    });

    it('fades the splash out when the first render has settled, and hides it when the fade has ended, not before', () => {
        holdForReveal(html, panel, body);
        reveal(html, panel);
        expect(has(html, 'reveal-pending')).toEqual([false]);
        expect(has(panel, 'fading', 'hidden')).toEqual([true, false]);
        vi.advanceTimersByTime(REVEAL_FADE_MS + 60);                 // (a fade that is late, because the page is busy: it is not cut short)
        expect(has(panel, 'hidden')).toEqual([false]);
        panel.endTransition();
        expect(has(panel, 'hidden', 'fading')).toEqual([true, false]);
        expect(panel.listeners.size).toBe(0);                        // (and the listener is gone)
    });

    it('does not take the end of another transition, or of the transition of something in the panel, for the end of the fade', () => {
        holdForReveal(html, panel, body);
        reveal(html, panel);
        panel.endTransition('background-color');
        panel.endTransition('opacity', element());
        expect(has(panel, 'hidden')).toEqual([false]);
        panel.endTransition('opacity');
        expect(has(panel, 'hidden')).toEqual([true]);
    });

    it('hides the splash after a while when no transition runs (a page that is not drawn, reduced motion)', () => {
        holdForReveal(html, panel, body);
        reveal(html, panel);
        vi.advanceTimersByTime(REVEAL_FADE_MS * 3 - 1);
        expect(has(panel, 'hidden')).toEqual([false]);
        vi.advanceTimersByTime(1);
        expect(has(panel, 'hidden', 'fading')).toEqual([true, false]);
        panel.endTransition();                                       // (a late end of the fade does nothing more)
        expect(has(panel, 'hidden')).toEqual([true]);
    });

    it('does not keep the splash for ever when the first render does not settle', () => {
        holdForReveal(html, panel, body);
        vi.advanceTimersByTime(REVEAL_MAX_WAIT_MS - 1);
        expect(has(html, 'reveal-pending')).toEqual([true]);
        vi.advanceTimersByTime(1);
        expect(has(html, 'reveal-pending')).toEqual([false]);
        expect(has(panel, 'fading')).toEqual([true]);
        vi.advanceTimersByTime(REVEAL_FADE_MS * 3);
        expect(has(panel, 'hidden')).toEqual([true]);
    });

    it('is revealed once: a settled render after the timeout (or twice) does not start another fade', () => {
        holdForReveal(html, panel, body);
        reveal(html, panel);
        panel.endTransition();
        panel.classList.remove('hidden');                       // (if it were done again, it would hide this panel)
        reveal(html, panel);
        vi.advanceTimersByTime(REVEAL_MAX_WAIT_MS * 2);
        expect(has(panel, 'hidden', 'fading')).toEqual([false, false]);
    });
});

describe('the splash that is held is an overlay, out of the layout', () => {
    it('is moved to the body, with the place and the size that it had when it was held', () => {
        holdForReveal(html, panel, body);
        expect(body.children).toEqual([panel]);
        expect(has(panel, 'welcome-overlay')).toEqual([true]);
        expect(panel.style).toEqual({ top: '58px', left: '0px', width: '1280px', height: '600px' });
    });

    it('does not follow the layout that changes afterwards (the panes take their places: the graph area gets smaller, and the splash must not)', () => {
        holdForReveal(html, panel, body);
        panel.box = { left: 0, top: 58, width: 880, height: 420 };         // (the area that it was in, after the right pane and the timeline came)
        expect(panel.style).toEqual({ top: '58px', left: '0px', width: '1280px', height: '600px' });
    });

    it('is measured before it is moved, not after (the box of a panel that has left the layout is not what it had)', () => {
        const order = [];
        panel.getBoundingClientRect = () => { order.push('measured'); return panel.box; };
        body.appendChild = () => order.push('moved');
        holdForReveal(html, panel, body);
        expect(order).toEqual(['measured', 'moved']);
    });

    it('is not moved when there is no splash to keep (a later trace)', () => {
        panel = welcomePanel(['hidden']);
        holdForReveal(html, panel, body);
        expect(body.children).toEqual([]);
        expect(panel.style).toEqual({});
    });
});

describe('a later trace, without the splash', () => {
    it('holds nothing, and reveals nothing', () => {
        panel = welcomePanel(['hidden']);
        expect(holdForReveal(html, panel, body)).toBe(false);
        expect(has(html, 'reveal-pending')).toEqual([false]);
        reveal(html, panel);
        expect(has(panel, 'fading')).toEqual([false]);
    });

    it('does nothing for a page that has no welcome panel', () => {
        expect(holdForReveal(html, null, body)).toBe(false);
        reveal(html, null);
        expect(has(html, 'reveal-pending')).toEqual([false]);
    });
});
