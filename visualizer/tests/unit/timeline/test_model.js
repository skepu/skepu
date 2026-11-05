// test_model.js — the rows and items of the timeline (script/timeline/model.js), what is drawn of them (draw.js), the axis (axis.js), the zoom strip and the momentum.
import { describe, it, expect } from 'vitest';
import { buildTimeline, forEachItemIn, itemOfNode, tooltip, stackRows, fitRowHeight } from '../../../script/timeline/model.js';
import { makeView, drawPlot, drawAxis, drawLabels, scaleToFit, labelColor } from '../../../script/timeline/draw.js';
import { formatTick, niceStep, ticks, formatDuration } from '../../../script/timeline/axis.js';
import { palette, METRICS } from '../../../script/timeline/style.js';
import { releaseVelocity, decayVelocity, isMoving, FLING } from '../../../script/timeline/fling.js';
import { scaleToFraction, fractionToScale, formatScale } from '../../../script/timeline/zoombar.js';

const call = (id, start, end, extra = {}) => ({ data: { id, type: 'skeleton_call', label: id, backend: 'CPU', start, end, ...extra } });
const items = m => m.items.map(i => [i.id, i.row, i.t0, i.t1]);

describe('the model', () => {
    it('is null when nothing is timed', () => {
        expect(buildTimeline([])).toBeNull();
        expect(buildTimeline([{ data: { id: 'x', type: 'skeleton_call', label: 'x' } }])).toBeNull();
    });

    it('has a row for each backend, in order, and the rows that the trace has after them', () => {
        const m = buildTimeline([call('a', 0, 10), call('b', 5, 9, { backend: 'OpenMP' }),
            { data: { id: 't', type: 'transfer', label: 't', start: 1, end: 2 } },
            { data: { id: 'k', type: 'compile', label: 'k', start: 0, end: 1, ok: false } },
            { data: { id: 'm', type: 'allocation', label: 'm', timestamp: 3 } }]);
        const R = METRICS.rowH;
        expect(m.rows.map(r => [r.kind, r.label, r.y])).toEqual([['backend', 'CPU', 0], ['backend', 'OpenMP', R], ['transfers', 'Transfers', 2 * R],
            ['builds', 'Builds', 3 * R], ['events', 'Events', 4 * R]]);
        expect(m.rowsHeight).toBe(5 * R);
        expect(m.height).toBe(5 * R + METRICS.axisH);
        expect(m.items.find(i => i.id === 'k').style).toBe('build-failed');
        expect(m.items.find(i => i.id === 'm')).toMatchObject({ shape: 'marker', t0: 3, t1: 3 });
    });

    it('has the time of everything, also the marker and the interval of a region', () => {
        const m = buildTimeline([call('a', 10, 20), { data: { id: 'm', type: 'deallocation', label: 'm', timestamp: 40 } },
            { data: { id: 'r', type: 'region', label: 'r', region_start: 5, region_end: 12 } }]);
        expect(m.range).toEqual({ minT: 5, maxT: 40, span: 35 });
        expect(m.rows[0]).toMatchObject({ kind: 'regions', y: 0, h: METRICS.rowH });
    });

    it('has an item for every interval of a call, with the iteration and the label of it', () => {
        const m = buildTimeline([call('a', 0, 30, { intervals: [[0, 10], [20, 30]] })]);
        expect(m.items.map(i => [i.iter, i.label, i.t0, i.t1])).toEqual([[0, 'a 1', 0, 10], [1, 'a 2', 20, 30]]);
        expect(m.count).toBe(2);
        expect(itemOfNode(m, 'a', 1).t0).toBe(20);
        expect(itemOfNode(m, 'a', 9).t0).toBe(0);
        expect(itemOfNode(m, 'nothing')).toBeNull();
    });

    it('gives the colour of the data to calls and transfers only', () => {
        const m = buildTimeline([call('a', 0, 1), { data: { id: 't', type: 'transfer', label: 't', start: 0, end: 1 } }, { data: { id: 'k', type: 'compile', label: 'k', start: 0, end: 1 } }],
            { colorOf: () => 'red' });
        expect(m.items.map(i => i.fill)).toEqual(['red', 'red', null]);
    });

    it('is of ranks for an MPI program: a bar on every rank, light for the interval and solid for the work', () => {
        const rd = [{ rank: 0, start: 0, end: 10, work_start: 2, work_end: 8, local_elements: 5 }, { rank: 1, start: 1, end: 12, local_elements: 3 }];
        const m = buildTimeline([call('a', 0, 10, { rank_datas: [rd] })]);
        expect(m.ranks).toBe(2);
        expect(m.rows.map(r => r.kind + (r.rank ?? ''))).toEqual(['rank0', 'backend0', 'rank1', 'backend1']);
        expect(m.items.map(i => [i.rank, i.row, i.spans])).toEqual([
            [0, 1, [{ t0: 0, t1: 10, light: true }, { t0: 2, t1: 8, light: false }]],
            [1, 3, [{ t0: 1, t1: 12, light: true }]]]);
        expect(m.range.maxT).toBe(12);
        expect(tooltip(m.items[0], m)).toContain('Rank 0: 10.0µs, worked 6.0µs');
    });

    it('has a row for the ranks in the collectives that is opened up, and a mark that says which way it is', () => {
        const rd = [{ rank: 0, start: 0, end: 10 }, { rank: 1, start: 0, end: 10 }];
        const comm = { data: { id: 'c', type: 'communication', label: 'c', start: 2, end: 8, rank_summaries: [{ interval: { start: 2, end: 9 }, max_wait_us: 3 }],
            rank_datas: [[{ rank: 0, start: 2, end: 9, bytes_sent: 4 }, { rank: 1, start: 3, end: 9 }]] } };
        const closed = buildTimeline([call('a', 0, 10, { rank_datas: [rd] }), comm]);
        const open = buildTimeline([call('a', 0, 10, { rank_datas: [rd] }), comm], { commExpanded: true });
        expect(closed.rows.find(r => r.kind === 'comm').toggle).toBe('closed');
        expect(closed.rows.some(r => r.kind === 'rank-comm')).toBe(false);
        expect(open.rows.find(r => r.kind === 'comm').toggle).toBe('open');
        expect(open.rows.filter(r => r.kind === 'rank-comm')).toHaveLength(2);
        expect(open.items.filter(i => i.rank != null && i.style === 'comm')).toHaveLength(2);
        expect(closed.items.find(i => i.style === 'comm').spans).toEqual([{ t0: 2, t1: 5, light: true }, { t0: 5, t1: 9, light: false }]);
    });
});

describe('the height of the rows', () => {
    const rd = [{ rank: 0, start: 0, end: 10 }, { rank: 1, start: 0, end: 10 }];
    const nodes = [call('a', 0, 10, { rank_datas: [rd] }), { data: { id: 't', type: 'transfer', label: 't', start: 1, end: 2 } },
        { data: { id: 'r', type: 'region', label: 'r', region_start: 0, region_end: 5 } }];

    it('are stacked as the layout has them at the normal height', () => {
        const m = buildTimeline(nodes);
        const before = m.rows.map(r => [r.y, r.h]), height = m.height;
        stackRows(m, METRICS.rowH);
        expect(m.rows.map(r => [r.y, r.h])).toEqual(before);
        expect(m.height).toBe(height);
    });

    it('are taller for the flexible rows only: the headers of the ranks keep theirs', () => {
        const m = buildTimeline(nodes);
        stackRows(m, 50);
        expect(m.rows.map(r => [r.kind, r.h])).toEqual([['regions', 50], ['rank', METRICS.headerH], ['backend', 50], ['rank', METRICS.headerH], ['backend', 50], ['transfers', 50]]);
        expect(m.rows[2].y).toBe(50 + METRICS.headerH);
        expect(m.rowsHeight).toBe(m.rows.at(-1).y + 50);
        expect(m.height).toBe(m.rowsHeight + METRICS.axisH);
        expect(m.naturalHeight).toBeLessThan(m.height);
    });

    it('fill the room that the rows have, but not less than the normal height or more than the largest', () => {
        const m = buildTimeline(nodes);
        const fixed = 2 * METRICS.headerH;
        expect(fitRowHeight(m, fixed + 4 * 40)).toBe(40);
        expect(fitRowHeight(m, fixed + 4 * 41 + 3)).toBe(41);
        expect(fitRowHeight(m, 10)).toBe(METRICS.rowH);
        expect(fitRowHeight(m, 5000)).toBe(METRICS.maxRowH);
    });
});

describe('the items of a stretch of time', () => {
    const many = Array.from({ length: 100 }, (_, i) => call('n' + i, i * 10, i * 10 + 5));
    const m = buildTimeline(many);
    const found = (t0, t1) => { const out = []; forEachItemIn(m, 0, t0, t1, i => out.push(i.id)); return out; };

    it('are those that overlap it, found in order', () => {
        expect(found(100, 125)).toEqual(['n10', 'n11', 'n12']);
        expect(found(106, 109)).toEqual([]);
        expect(found(-50, -1)).toEqual([]);
        expect(found(0, 1e9)).toHaveLength(100);
    });

    it('include a long one that starts before it', () => {
        const long = buildTimeline([call('long', 0, 1000), ...many.slice(50)]);
        const out = []; forEachItemIn(long, 0, 700, 720, i => out.push(i.id));
        expect(out).toContain('long');
    });

    it('have the nested regions after the one they are in, if they start together', () => {
        const r = (id, depth, end) => ({ data: { id, type: 'region', label: id, region_start: 0, region_end: end, region_depth: depth } });
        const g = buildTimeline([r('inner', 1, 5), r('outer', 0, 10), call('a', 0, 10)]);
        expect(g.byRow[0].map(i => i.id)).toEqual(['outer', 'inner']);
    });
});

describe('what is drawn', () => {
    const m = buildTimeline(Array.from({ length: 50 }, (_, i) => call('n' + i, i * 100, i * 100 + 50)));
    const pal = palette(false);

    it('is only the items of the window, in the coordinates of the window', () => {
        const view = makeView(m, 100000);
        const win = { x0: 20000, x1: 21000 };
        const svg = drawPlot(m, view, win, pal, null);
        const xs = [...svg.matchAll(/class="gantt-bar[^"]*"[^>]* x="([-\d.]+)"/g)].map(x => +x[1]);
        expect(xs.length).toBeGreaterThan(0);
        expect(xs.length).toBeLessThan(10);
        for (const x of xs) expect(x).toBeGreaterThan(-100000 * 60 / 4950 - 1);
        expect(svg).not.toContain('data-i="0"');
    });

    it('has everything when the window is all of the content', () => {
        const view = makeView(m, 3000);
        const svg = drawPlot(m, view, { x0: 0, x1: 3000 }, pal, null);
        expect(svg.match(/<rect class="gantt-bar/g)).toHaveLength(50);
    });

    it('marks the selected node, and leaves out what is thinner than a pixel and starts inside the last bar', () => {
        const view = makeView(m, 3000);
        expect(drawPlot(m, view, { x0: 0, x1: 3000 }, pal, 'n3').match(/gantt-selected/g)).toHaveLength(1);
        const thin = buildTimeline([call('a', 0, 1000), ...Array.from({ length: 30 }, (_, i) => call('t' + i, 1 + i * 0.01, 1.001 + i * 0.01)), call('z', 990, 1000)]);
        const out = drawPlot(thin, makeView(thin, 400), { x0: 0, x1: 400 }, pal, null);
        expect(out.match(/<rect class="gantt-bar/g).length).toBeLessThan(10);
    });

    it('escapes what the trace says', () => {
        const evil = buildTimeline([call('a', 0, 100, { label: '<b>"x"</b>' })]);
        const out = drawPlot(evil, makeView(evil, 2000), { x0: 0, x1: 2000 }, pal, null);
        expect(out).not.toContain('<b>');
        expect(out).toContain('&lt;b&gt;');
    });

    it('has the labels of the rows, and the way of the collectives', () => {
        const rd = [{ rank: 0, start: 0, end: 10 }, { rank: 1, start: 0, end: 10 }];
        const g = buildTimeline([call('a', 0, 10, { rank_datas: [rd] }), { data: { id: 'c', type: 'communication', label: 'c', start: 2, end: 8 } }]);
        const labels = drawLabels(g, pal);
        expect(labels).toContain('Rank 1');
        expect(labels).toMatch(/data-comm-toggle="1"[^>]*>.*▸ Comm\./);
    });

    it('has the ticks at their place also when the trace does not start at 0', () => {
        const late = buildTimeline([call('a', 3509, 6334)]);
        const view = makeView(late, 1070);
        const axis = drawAxis(late, view, { x0: 0, x1: 1070 }, pal);
        const first = /<text x="([-\d.]+)"[^>]*>\+0µs/.exec(axis);
        expect(+first[1]).toBeCloseTo(METRICS.labelW + 0.5);
        for (const m of axis.matchAll(/<text x="([-\d.]+)"/g)) expect(+m[1]).toBeGreaterThanOrEqual(METRICS.labelW);
    });

    it('has the ticks of the window on the axis', () => {
        const view = makeView(m, 3000);
        const axis = drawAxis(m, view, { x0: 0, x1: 3000 }, pal);
        expect(axis).toContain('>+0');
        expect(axis).toContain('text-anchor="start"');
    });
});

describe('the label of a bar', () => {
    it('has the colour of its style: white or black on a call by its fill, dark on a build, that of the theme on a region', () => {
        const pal = palette(false);
        expect(labelColor({ style: 'call' }, pal)).toBe(pal.barText);
        expect(labelColor({ style: 'call', fill: 'rgb(40,90,160)' }, pal)).toBe(pal.barText);
        expect(labelColor({ style: 'transfer', fill: 'rgb(255,255,0)' }, pal)).toBe(pal.barTextDark);
        expect(labelColor({ style: 'call', fill: 'rgb(0,255,0)' }, pal)).toBe(pal.barTextDark);
        expect(labelColor({ style: 'call', lightText: true }, pal)).toBe(pal.lightBarText);
        expect(labelColor({ style: 'build-failed' }, pal)).toBe(pal.buildText);
        expect(labelColor({ style: 'region' }, pal)).toBe(pal.regionText);
    });

    it('is drawn with the number of its item, so that the view can pin it', () => {
        const m = buildTimeline([call('a', 0, 1000)]);
        const out = drawPlot(m, makeView(m, 2000), { x0: 0, x1: 2000 }, palette(false), null);
        expect(out).toMatch(/<svg data-li="0" x="[\d.]+"/);
    });
});

describe('the zoom to a range', () => {
    it('is the one at which the range fills the plot of the pane', () => {
        const model = buildTimeline([call('a', 0, 10000)]);
        for (const dt of [50, 2000, 10000]) {
            const view = makeView(model, 900 * scaleToFit(model, 900, dt));
            expect(view.pxPerUs * dt).toBeCloseTo(900 - METRICS.labelW);
        }
    });
});

describe('the axis', () => {
    it('has round steps', () => {
        expect(niceStep(0.7)).toBe(1);
        expect(niceStep(1.1)).toBe(2);
        expect(niceStep(2.1)).toBe(5);
        expect(niceStep(5.1)).toBe(10);
        expect(niceStep(180)).toBe(200);
        expect(niceStep(0.03)).toBeCloseTo(0.05);
    });

    it('has ticks at multiples of the step, at least minPx apart, from 0', () => {
        const t = ticks(-10, 1000, 1, 90);
        expect(t.map(x => x.t)).toEqual([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]);
        expect(ticks(450, 700, 1, 90).map(x => x.t)).toEqual([500, 600, 700]);
        expect(ticks(0, 10, 0, 90)).toEqual([]);
    });

    it('says a time with the digits that tell it from the next tick', () => {
        expect(formatTick(0, 1000)).toBe('+0ms');
        expect(formatTick(1500, 500)).toBe('+1.5ms');
        expect(formatTick(2e6, 5e5)).toBe('+2.0s');
        expect(formatTick(12, 2)).toBe('+12µs');
        expect(formatTick(0.3, 0.1)).toBe('+0.3µs');
        expect(formatDuration(2500)).toBe('2.5ms');
    });
});

describe('the zoom strip', () => {
    it('is logarithmic, from 1× at the bottom to 1000× at the top', () => {
        expect(scaleToFraction(1)).toBe(0);
        expect(scaleToFraction(1000)).toBeCloseTo(1);
        expect(scaleToFraction(Math.sqrt(1000))).toBeCloseTo(0.5);
        expect(fractionToScale(0.5)).toBeCloseTo(Math.sqrt(1000));
        expect(fractionToScale(-1)).toBe(1);
        expect(fractionToScale(2)).toBeCloseTo(1000);
        expect(formatScale(1)).toBe('1.0×');
        expect(formatScale(12.4)).toBe('12×');
    });
});

describe('the momentum of a drag', () => {
    const drag = (...pts) => pts.map(([t, x, y = 0]) => ({ t, x, y }));

    it('is the speed of the last stretch of the drag', () => {
        expect(releaseVelocity(drag([0, 0], [50, 100], [90, 300], [100, 340]), 105)).toEqual({ x: (340 - 100) / 50, y: 0 });
        expect(releaseVelocity(drag([20, 0, 0], [100, 40, -16]), 110)).toEqual({ x: 0.5, y: -0.2 });
    });

    it('is none if the drag was still before the release, or has too little to tell', () => {
        expect(releaseVelocity(drag([0, 0], [50, 100], [90, 300]), 90 + FLING.staleMs + 1)).toEqual({ x: 0, y: 0 });
        expect(releaseVelocity(drag([90, 300]), 95)).toEqual({ x: 0, y: 0 });
        expect(releaseVelocity(drag([0, 0], [10, 50]), 400)).toEqual({ x: 0, y: 0 });
    });

    it('falls with the time, the same for any way of dividing it, and stops when slow', () => {
        const once = decayVelocity({ x: 2, y: -1 }, 200);
        const twice = decayVelocity(decayVelocity({ x: 2, y: -1 }, 100), 100);
        expect(once.x).toBeCloseTo(twice.x);
        expect(once.y).toBeCloseTo(twice.y);
        expect(decayVelocity({ x: 1, y: 0 }, FLING.tauMs).x).toBeCloseTo(Math.exp(-1));
        expect(isMoving({ x: 0.5, y: 0 })).toBe(true);
        expect(isMoving({ x: 0.01, y: 0.01 })).toBe(false);
    });
});
