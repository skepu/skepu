// test_panzoom.js — how the wheel and the pinch move the graph (script/panzoom.js). No device is guessed: a plain wheel or a swipe pans; Ctrl or Cmd
// with the wheel zooms (that is how Chromium and Firefox report the pinch of a trackpad); the gestures of Safari zoom, and the wheel waits for them.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { bindPanZoom, wheelZoomFactor, wheelPan, wheelPixels, isClearlyInView, animateFitUnlessInView, FitKeeper, FIT_SETTLE_MS, FIT_QUIET_MS, FIT_WAIT_MAX_MS } from '../../script/panzoom.js';

function graph(zoom = 1) {
    const calls = [];
    return { calls, zoom(arg) { if (arg === undefined) return zoom; calls.push(['zoom', arg]); zoom = arg.level; }, panBy(d) { calls.push(['panBy', d]); } };
}
function element() {
    const listeners = {};
    return {
        listeners, getBoundingClientRect: () => ({ left: 10, top: 20, height: 500 }),
        addEventListener(type, fn) { listeners[type] = fn; }, removeEventListener(type) { delete listeners[type]; },
    };
}
const wheel = (props) => ({ deltaX: 0, deltaY: 0, deltaMode: 0, ctrlKey: false, metaKey: false, shiftKey: false, clientX: 110, clientY: 120, preventDefault() { this.prevented = true; }, ...props });

describe('the wheel', () => {
    it('pans for a plain wheel or a swipe, in the direction of the fingers, and keeps the page still', () => {
        const cy = graph(), el = element();
        bindPanZoom(cy, el);
        const e = wheel({ deltaX: 4, deltaY: -10 });
        el.listeners.wheel(e);
        expect(e.prevented).toBe(true);
        expect(cy.calls).toEqual([['panBy', { x: -6, y: 15 }]]);
    });

    it('pans sideways with Shift for a wheel that only goes up and down', () => {
        expect(wheelPan(wheel({ deltaY: 40, shiftKey: true }), 500)).toEqual({ x: -60, y: -0 });
        expect(wheelPan(wheel({ deltaX: 8, deltaY: 40, shiftKey: true }), 500)).toEqual({ x: -12, y: -60 }); // (a swipe already has its sideways)
    });

    it('zooms around the pointer with Ctrl or Cmd (the pinch of Chromium and Firefox)', () => {
        for (const key of ['ctrlKey', 'metaKey']) {
            const cy = graph(2), el = element();
            bindPanZoom(cy, el);
            el.listeners.wheel(wheel({ deltaY: -5, [key]: true }));
            expect(cy.calls).toHaveLength(1);
            const [name, arg] = cy.calls[0];
            expect(name).toBe('zoom');
            expect(arg.renderedPosition).toEqual({ x: 100, y: 100 });   // (clientX - left, clientY - top)
            expect(arg.level).toBeCloseTo(2 * Math.exp(0.1), 10);        // (toward the screen: in)
        }
    });

    it('is the same events and then the same result, whichever device made them', () => {
        // a trackpad pinch (small steps) and a mouse wheel with Ctrl (big steps, cut) both zoom in, and the notch is not a jump
        expect(wheelZoomFactor(-3)).toBeGreaterThan(1);
        expect(wheelZoomFactor(-100)).toBeGreaterThan(1);
        expect(wheelZoomFactor(-100)).toBeLessThan(1.4);
        expect(wheelZoomFactor(100)).toBeGreaterThan(0.7);
        expect(wheelZoomFactor(0)).toBe(1);
        expect(wheelZoomFactor(-1000)).toBe(wheelZoomFactor(-100));   // (cut)
    });

    it('counts lines and pages of a wheel (Firefox) in pixels', () => {
        expect(wheelPixels(3, 0, 500)).toBe(3);
        expect(wheelPixels(3, 1, 500)).toBe(48);
        expect(wheelPixels(1, 2, 500)).toBe(500);
        expect(wheelPan(wheel({ deltaY: 3, deltaMode: 1 }), 500)).toEqual({ x: -0, y: -72 });
    });
});

describe('the gestures of Safari', () => {
    it('zoom around the pointer from the zoom at the start, and the wheel does not pan meanwhile', () => {
        const cy = graph(1.5), el = element();
        bindPanZoom(cy, el);
        el.listeners.gesturestart();
        el.listeners.gesturechange({ scale: 2, clientX: 60, clientY: 70, target: el });
        expect(cy.calls[0][0]).toBe('zoom');
        expect(cy.calls[0][1].level).toBeCloseTo(Math.pow(2, 1.2) * 1.5, 10);
        expect(cy.calls[0][1].renderedPosition).toEqual({ x: 50, y: 50 });
        el.listeners.wheel(wheel({ deltaX: 5, deltaY: 5 }));
        el.listeners.wheel(wheel({ deltaY: 5, ctrlKey: true }));
        expect(cy.calls).toHaveLength(1);
        el.listeners.gestureend();
        el.listeners.wheel(wheel({ deltaY: 5 }));
        expect(cy.calls[1][0]).toBe('panBy');
    });
});

it('takes its listeners away again', () => {
    const el = element();
    const unbind = bindPanZoom(graph(), el);
    expect(Object.keys(el.listeners).sort()).toEqual(['gesturechange', 'gestureend', 'gesturestart', 'wheel']);
    unbind();
    expect(Object.keys(el.listeners)).toEqual([]);
});

describe('a highlight in another view moves the graph only if the node is not clearly in view', () => {
    const view = { width: 1000, height: 600 };
    const box = (x, y, w, h) => ({ x1: x, y1: y, x2: x + w, y2: y + h });

    it('is clearly in view when all of it is inside, with room from the edges, and it is not tiny', () => {
        expect(isClearlyInView(box(400, 250, 80, 60), view)).toBe(true);
        expect(isClearlyInView(box(40, 40, 60, 60), view)).toBe(true);
    });

    it('is not when any part is outside the view, or close to an edge', () => {
        expect(isClearlyInView(box(-10, 250, 80, 60), view)).toBe(false);   // (left of the view)
        expect(isClearlyInView(box(960, 250, 80, 60), view)).toBe(false);   // (half out on the right)
        expect(isClearlyInView(box(400, 570, 80, 60), view)).toBe(false);   // (below)
        expect(isClearlyInView(box(5, 250, 80, 60), view)).toBe(false);     // (inside, but at the edge)
        expect(isClearlyInView(box(400, 10, 80, 60), view)).toBe(false);
        expect(isClearlyInView(box(400, 300, 2000, 60), view)).toBe(false); // (bigger than the view)
    });

    it('is not when it is too small to see what it is: in pixels and against the view', () => {
        expect(isClearlyInView(box(400, 250, 10, 8), view)).toBe(false);
        expect(isClearlyInView(box(400, 250, 27, 8), view)).toBe(false);    // (the least is 28 pixels, or 3% of the shorter side, 600: 18)
        expect(isClearlyInView(box(400, 250, 28, 8), view)).toBe(true);     // (the longer side counts)
        expect(isClearlyInView(box(400, 250, 44, 44), { width: 1400, height: 900 })).toBe(true);     // (a node of the normal size at the normal zoom)
        expect(isClearlyInView(box(400, 250, 40, 40), { width: 4000, height: 2400 })).toBe(false);   // (small against a big view: 3% of 2400 is 72)
    });

    it('is clearly in view at the edge of a graph that was just fitted (Reset View leaves 30 px: its first and last nodes must not move it)', () => {
        // (a fit makes a small graph as big as the view allows: the nodes of a big view are big)
        for (const [v, size] of [[{ width: 1000, height: 600 }, 44], [{ width: 1400, height: 900 }, 44], [{ width: 4000, height: 2400 }, 100]]) {
            const pad = 30;
            expect(isClearlyInView(box(pad, pad, size, size), v), `top left, ${v.width}x${v.height}`).toBe(true);
            expect(isClearlyInView(box(v.width - pad - size, v.height - pad - size, size, size), v), `bottom right, ${v.width}x${v.height}`).toBe(true);
        }
    });

    function graphOf(b, viewSize = view) {
        const calls = [];
        return {
            calls, width: () => viewSize.width, height: () => viewSize.height,
            animate: (...args) => calls.push(args),
            eles: { renderedBoundingBox: (opts) => { calls.push(['box', opts]); return b; } },
        };
    }

    it('leaves the view alone for a node that is clearly in it', () => {
        const cy = graphOf(box(400, 250, 80, 60));
        expect(animateFitUnlessInView(cy, cy.eles, 100)).toBe(false);
        expect(cy.calls.filter(c => c[0] !== 'box')).toEqual([]);
    });

    it('pans and zooms to a node that is not, as before (fit with the padding)', () => {
        for (const b of [box(-200, 250, 80, 60), box(400, 250, 6, 6)]) {
            const cy = graphOf(b);
            expect(animateFitUnlessInView(cy, cy.eles, 100)).toBe(true);
            const animate = cy.calls.filter(c => c[0] !== 'box');
            expect(animate).toHaveLength(1);
            expect(animate[0][0]).toEqual({ fit: { eles: cy.eles, padding: 100 } });
        }
    });

    it('looks at the node itself, not at its labels or the overlay of a hover', () => {
        const cy = graphOf(box(400, 250, 80, 60));
        animateFitUnlessInView(cy, cy.eles, 100);
        expect(cy.calls[0]).toEqual(['box', { includeLabels: false, includeOverlays: false }]);
    });
});

describe('the graph stays fitted when the element it is in changes size', () => {
    beforeEach(() => { vi.useFakeTimers(); });
    afterEach(() => { vi.useRealTimers(); });

    // a graph that says what is done to it, and sends 'viewport' as Cytoscape does (for each pan and zoom, also those of fit and resize)
    function fakeGraph() {
        const handlers = {};
        const cy = {
            log: [], handlers,
            on(name, f) { (handlers[name] ??= []).push(f); },
            emit(name) { (handlers[name] ?? []).forEach(f => f()); },
            fit(eles, padding) { cy.log.push(['fit', padding]); cy.emit('viewport'); },
            resize() { cy.log.push(['resize']); },
            panBy() { cy.log.push(['panBy']); cy.emit('viewport'); },
        };
        return cy;
    }
    function keeper(padding = 30) {
        const cy = fakeGraph();
        let resized = null, stopped = false;
        const fit = new FitKeeper({}, padding, (el, callback) => { resized = callback; return () => { stopped = true; }; });
        fit.attach(cy);
        // (`resize`: the size changes and then stays as it is, so that what waits for it is done; `change`: it changes, and what waits goes on waiting)
        return { cy, fit, resize: () => { resized(); vi.advanceTimersByTime(FIT_SETTLE_MS); }, change: () => resized(), stopped: () => stopped };
    }
    const names = cy => cy.log.map(c => c[0]);

    it('fits again with the same padding when the element changes size after the graph was fitted (the legend and the timeline come in)', () => {
        const { cy, fit, resize } = keeper(30);
        fit.fit();
        resize();                      // (the timeline has opened: the element is smaller)
        resize();                      // (and the legend)
        expect(cy.log).toEqual([['fit', 30], ['resize'], ['fit', 30], ['resize'], ['fit', 30]]);
    });

    it('tells the graph of the new size, but leaves the view, when the user has panned or zoomed since', () => {
        const { cy, fit, resize } = keeper();
        fit.fit();
        cy.panBy();                    // (the user, a highlight, a pinch: anything that is not the fit)
        cy.log.length = 0;
        resize();
        expect(names(cy)).toEqual(['resize']);
    });

    it('is fitted again by a new fit (Reset View), and then follows the size again', () => {
        const { cy, fit, resize } = keeper();
        fit.fit();
        cy.panBy();
        fit.fit();                     // (Reset View)
        cy.log.length = 0;
        resize();
        expect(names(cy)).toEqual(['resize', 'fit']);
    });

    it('does not take its own fit and resize for a move by the user', () => {
        const { cy, fit, resize } = keeper();
        fit.fit();
        resize();
        resize();
        expect(names(cy).filter(n => n === 'fit')).toHaveLength(3);   // (it stayed the fitted view through its own events)
    });

    it('does not fit a graph that was never fitted (the first size of the element, before the layout)', () => {
        const { cy, resize } = keeper();
        resize();
        expect(names(cy)).toEqual(['resize']);
    });

    it('starts again for the next graph (a render makes a new one): not fitted until it is', () => {
        const { cy, fit, resize } = keeper();
        fit.fit();
        const next = fakeGraph();
        fit.attach(next);
        resize();
        expect(names(next)).toEqual(['resize']);
        fit.fit();
        expect(names(next)).toEqual(['resize', 'fit']);
        expect(names(cy)).toEqual(['fit']);                         // (nothing more for the old one)
    });

    it('stops watching the element, and does nothing for a graph that is not there', () => {
        const { fit, resize, stopped, cy } = keeper();
        fit.fit();
        fit.destroy();
        expect(stopped()).toBe(true);
        cy.log.length = 0;
        resize();
        expect(cy.log).toEqual([]);
    });

    it('waits until the size is still: a drag of a pane changes it for every pointer move and the graph is not touched until it stops', () => {
        const { cy, fit, change } = keeper();
        fit.fit();
        cy.log.length = 0;
        for (let i = 0; i < 20; i++) { change(); vi.advanceTimersByTime(16); }   // (a drag: 20 changes, a frame apart)
        expect(cy.log).toEqual([]);                                            // (no resize and no fit while it moves)
        vi.advanceTimersByTime(FIT_SETTLE_MS);
        expect(cy.log).toEqual([['resize'], ['fit', 30]]);                     // (one, when it has stopped)
    });

    it('waits for the last change: the wait starts again with each', () => {
        const { cy, fit, change } = keeper();
        fit.fit();
        cy.log.length = 0;
        change();
        vi.advanceTimersByTime(FIT_SETTLE_MS - 1);
        change();
        vi.advanceTimersByTime(FIT_SETTLE_MS - 1);
        expect(cy.log).toEqual([]);
        vi.advanceTimersByTime(1);
        expect(names(cy)).toEqual(['resize', 'fit']);
    });

    it('does nothing after it is destroyed, also for a change that was waiting', () => {
        const { cy, fit, change } = keeper();
        fit.fit();
        cy.log.length = 0;
        change();
        fit.destroy();
        vi.advanceTimersByTime(FIT_SETTLE_MS * 2);
        expect(cy.log).toEqual([]);
    });

    describe('whenSettled: what a render brings in (the legend, the timeline) is in place before the graph is seen', () => {
        it('fits at once, to the size that the element has, and then calls back, when the size does not change', () => {
            const { cy, fit } = keeper();
            fit.fit();
            cy.log.length = 0;
            const done = [];
            fit.whenSettled(() => done.push(cy.log.map(c => c[0]).join(',')));
            vi.advanceTimersByTime(FIT_QUIET_MS - 1);
            expect(done).toEqual([]);
            vi.advanceTimersByTime(40);
            expect(done).toEqual(['resize,fit']);                  // (the graph was told and fitted before the callback: what is seen is fitted)
        });

        it('waits while the size goes on changing (the timeline opens), then fits once and calls back', () => {
            const { cy, fit, change } = keeper();
            fit.fit();
            cy.log.length = 0;
            let calledAt = null; const t0 = Date.now();
            fit.whenSettled(() => { calledAt = Date.now() - t0; });
            for (let i = 0; i < 10; i++) { vi.advanceTimersByTime(16); change(); }     // (an animation of 160 ms: a change each frame)
            expect(calledAt).toBe(null);
            expect(cy.log).toEqual([]);                                            // (nothing is done to the graph meanwhile)
            vi.advanceTimersByTime(FIT_QUIET_MS + 20);
            expect(calledAt).toBeGreaterThanOrEqual(160 + FIT_QUIET_MS);
            expect(names(cy)).toEqual(['resize', 'fit']);                          // (once)
            vi.advanceTimersByTime(FIT_SETTLE_MS * 3);
            expect(names(cy)).toEqual(['resize', 'fit']);                          // (the ordinary refit that was waiting does not come on top)
        });

        it('does not wait for ever for a size that never settles', () => {
            const { cy, fit, change } = keeper();
            fit.fit();
            cy.log.length = 0;
            let called = false;
            fit.whenSettled(() => { called = true; });
            for (let t = 0; t < FIT_WAIT_MAX_MS + 100; t += 16) { change(); vi.advanceTimersByTime(16); }
            expect(called).toBe(true);
            expect(names(cy)).toEqual(['resize', 'fit']);
        });

        it('does not fit a view that was moved since the layout (but tells the graph of the size, and calls back)', () => {
            const { cy, fit } = keeper();
            fit.fit();
            cy.panBy();
            cy.log.length = 0;
            let called = false;
            fit.whenSettled(() => { called = true; });
            vi.advanceTimersByTime(FIT_QUIET_MS + 20);
            expect(called).toBe(true);
            expect(names(cy)).toEqual(['resize']);
        });

        it('is not called after destroy', () => {
            const { fit } = keeper();
            fit.fit();
            let called = false;
            fit.whenSettled(() => { called = true; });
            fit.destroy();
            vi.advanceTimersByTime(FIT_WAIT_MAX_MS * 2);
            expect(called).toBe(false);
        });
    });
});
