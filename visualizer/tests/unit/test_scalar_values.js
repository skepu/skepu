// test_scalar_values.js — the values of scalars (scalar_value events) on the scalar nodes: one value per iteration for the scalar of a call that is
// repeated; scalars of different calls (and of one call with several results) stay apart; a scalar that is made from another one has the later
// value; and the plot of the values (scalarplot.js).
import { describe, it, expect, beforeAll } from 'vitest';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { builder } from '../event_builders.js';
import { formatScalar, scalarSeries, scalarPlotSvg, scalarPlotHtml, renderScalarPlot, startsLogarithmic } from '../../script/scalarplot.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const replies = {};
let seq = 0;
async function call(type, payload = {}) {
    const id = String(++seq);
    await globalThis.self.onmessage({ data: { id, type, ...payload } });
    const reply = replies[id];
    if (reply.type === 'error') throw new Error(reply.error);
    return reply;
}
beforeAll(async () => {
    globalThis.self = {
        postMessage: (m) => { replies[m.id] = m; },
        location: { href: 'file://' + join(ROOT, 'script/backend/visualizer.js') },
        addEventListener() {},
    };
    await import(join(ROOT, 'script/backend/visualizer.js'));
});

const value = (id, v) => ({ type: 'scalar_value', object_id: id, value: v, time: 0 });

// n iterations of a region: a reduction that has a scalar of its own (ids 100+i) and a second one with two results (200+2i, 201+2i); the value events
// come before their calls, as the tracer writes them
function iterations(n, { error, total, pair, derived = false }) {
    const b = builder();
    const events = [b.alloc(1, 'x')];
    for (let i = 0; i < n; i++) {
        events.push({ type: 'region', label: 'Iteration', region_depth: 1, time: 0, file: 'p.cpp', line: 3 });
        if (error) events.push(value(100 + i, error(i)));
        if (derived) events.push(value(100 + i, derived(i)));
        events.push({ ...b.call('error', { pattern: 'MapReduce', ins: [1], outs: [100 + i] }), region: 'Iteration' });
        if (total) events.push(value(300 + i, total(i)));
        events.push({ ...b.call('total', { pattern: 'Reduce', ins: [1], outs: [300 + i] }), region: 'Iteration' });
        if (pair) { events.push(value(200 + 2 * i, pair(i)[0])); events.push(value(201 + 2 * i, pair(i)[1])); }
        if (pair) events.push({ ...b.call('both', { pattern: 'MapReduce', ins: [1], outs: [200 + 2 * i, 201 + 2 * i] }), region: 'Iteration' });
    }
    return events;
}

async function scalarInfos(events, opts = {}) {
    await call('load', { traceJson: JSON.stringify(events), traceFilename: 't.json' });
    const g = await call('graph', { show_regions: true, expand_all_regions: true, collapse_iteration: true, show_scalars: true, ...opts });
    const infos = [];
    for (const node of g.nodes) {
        const data = node.data ?? node;
        if (data.type !== 'scalar') continue;
        infos.push({ data, info: await call('getData', { nodeId: data.id }) });
    }
    return infos;
}

// the series of the scalar nodes that have values, in the order of their first value
const seriesOf = infos => infos.map(i => i.info.scalar_values).filter(Boolean).sort((a, b) => a[0] - b[0]);

describe('the values of the scalars on the scalar nodes', () => {
    it('are one value for each iteration, and scalars of different calls stay apart', async () => {
        const infos = await scalarInfos(iterations(4, { error: i => 1 / 10 ** i, total: i => 10 * i }));
        expect(infos).toHaveLength(2);
        expect(seriesOf(infos)).toEqual([[0, 10, 20, 30], [1, 0.1, 0.01, 0.001]]);
        for (const { data } of infos) expect(data.iteration_count).toBe(4);
    });

    it('stay apart for the results of one call', async () => {
        const infos = await scalarInfos(iterations(3, { pair: i => [i, 100 + i] }));
        expect(seriesOf(infos)).toEqual([[0, 1, 2], [100, 101, 102]]);
    });

    it('are the later value of a scalar that has the id of another one', async () => {
        // (as blas::nrm2: the sum, and its square root with the same id)
        const infos = await scalarInfos(iterations(3, { error: i => (i + 1) ** 2, derived: i => i + 1 }));
        expect(seriesOf(infos)).toEqual([[1, 2, 3]]);
    });

    it('keep values that are not finite, as strings, in their iteration', async () => {
        const infos = await scalarInfos(iterations(3, { error: i => (i === 1 ? 'nan' : i) }));
        expect(seriesOf(infos)).toEqual([[0, 'nan', 2]]);
    });

    it('are in the info of a scalar that is not repeated, as its value', async () => {
        const infos = await scalarInfos(iterations(1, { error: () => 0.25 }));
        const info = infos.find(i => i.info.scalar_values).info;
        expect(info.scalar_values).toEqual([0.25]);
        expect(info['Value']).toBe(0.25);
        expect(info['Result of']).toBe('error');
    });

    it('are not there when the trace has none', async () => {
        const infos = await scalarInfos(iterations(3, { error: null }));
        for (const { info } of infos) expect(info).not.toHaveProperty('scalar_values');
    });
});

describe('the plot of the values', () => {
    it('formats numbers for the labels', () => {
        expect(formatScalar(0)).toBe('0');
        expect(formatScalar(12)).toBe('12');
        expect(formatScalar(0.123456789)).toBe('0.12346');
        expect(formatScalar(1e-7)).toBe('1e-7');
        expect(formatScalar(2.5e9)).toBe('2.5e+9');
        expect(formatScalar('nan')).toBe('nan');
    });

    it('takes the numbers and their iterations, and leaves out the others', () => {
        const s = scalarSeries([3, 'nan', 1, 'inf', 2]);
        expect(s.points).toEqual([{ i: 0, v: 3 }, { i: 2, v: 1 }, { i: 4, v: 2 }]);
        expect([s.min, s.max, s.positive]).toEqual([1, 3, true]);
        expect(scalarSeries([-1, 2]).positive).toBe(false);
    });

    it('starts logarithmic for positive numbers over three orders of magnitude', () => {
        expect(startsLogarithmic(scalarSeries([1, 0.1, 0.01, 0.001]))).toBe(true);
        expect(startsLogarithmic(scalarSeries([1, 2, 3]))).toBe(false);
        expect(startsLogarithmic(scalarSeries([0, 1000]))).toBe(false);
    });

    it('draws a point for each number, the line broken at the others, and nothing for fewer than two numbers', () => {
        const svg = scalarPlotSvg([1, 'nan', 3, 4]);
        expect((svg.match(/<circle/g) || []).length).toBe(3);
        expect((svg.match(/M/g) || []).length).toBe(2);              // (two pieces of the line)
        expect(svg).toContain('Iteration 4: 4');
        expect(scalarPlotSvg([1])).toBe('');
        expect(scalarPlotSvg([1, 'nan', 'inf'])).toBe('');
    });

    it('puts the log scale only on positive numbers, and the points of a constant series in the middle', () => {
        const ys = svg => [...svg.matchAll(/cy="([0-9.]+)"/g)].map(m => +m[1]);
        const linear = ys(scalarPlotSvg([1, 10, 100], {}));
        const log = ys(scalarPlotSvg([1, 10, 100], { log: true }));
        expect(log[1]).toBeCloseTo((log[0] + log[2]) / 2, 0);        // (10 is halfway on a log scale)
        expect(linear[1]).toBeGreaterThan(log[1]);                    // (and low on the linear one)
        expect(ys(scalarPlotSvg([-1, 1, 3], { log: true }))).toEqual(ys(scalarPlotSvg([-1, 1, 3], {})));
        const flat = ys(scalarPlotSvg([5, 5, 5]));
        expect(new Set(flat).size).toBe(1);
    });
});

// The log scale checkbox, which toggled the scale once and then stopped: the title row was made once, with the state that the plot started in, so a redraw
// put back the checkbox of that state. A page is simulated with an element that has the HTML and the checkbox in it.
describe('the log scale checkbox', () => {
    it('is in the state of the plot in every redraw', () => {
        const values = [1, 0.1, 0.01, 0.001, 0.0001];
        expect(scalarPlotHtml(values, true)).toContain('type="checkbox" checked');
        expect(scalarPlotHtml(values, false)).not.toContain('checked');
        expect(scalarPlotHtml([1, -1, 2], false)).not.toContain('checkbox');     // (no log scale for numbers that are not above 0)
    });

    it('switches the scale every time it is clicked', () => {
        const values = [1, 0.1, 0.01, 0.001, 0.0001];                           // (starts logarithmic)
        const container = {
            style: {}, _html: '', box: null,
            get innerHTML() { return this._html; },
            set innerHTML(html) { this._html = html; this.box = /<input[^>]*checkbox/.test(html) ? { checked: /checkbox"? checked/.test(html), onchange: null } : null; },
            querySelector() { return this.box; },
        };
        const ys = () => [...container.innerHTML.matchAll(/cy="([0-9.]+)"/g)].map(m => +m[1]);
        renderScalarPlot(container, values, {});
        const logY = ys();
        expect(container.box.checked).toBe(true);
        for (let click = 1; click <= 4; click++) {
            container.box.checked = !container.box.checked;                      // (what the click does to the box, then its change handler)
            container.box.onchange();
            const expectLog = click % 2 === 0;
            expect(container.box.checked, 'click ' + click).toBe(expectLog);     // (the redraw has the checkbox in the state of the scale)
            expect(ys(), 'click ' + click).toEqual(expectLog ? logY : ys());
            if (!expectLog) expect(ys()).not.toEqual(logY);
        }
    });
});
