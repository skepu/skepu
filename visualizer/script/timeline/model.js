// timeline/model.js — what the timeline shows, as rows and items, apart from how it is drawn. Pure: no DOM.
//
// The timeline nodes of a trace (`timeline_nodes` of TraceDataProvider) become
//   rows:  { kind, label, y, h, shade, rank?, toggle? }, top to bottom (the rows of layout.js)
//   items: { idx, row, style, shape, id, iter, rank, t0, t1, spans, fill, depth, label, hidden, src }
// `spans` are the filled parts of a bar ({ t0, t1, light }; light: waiting and communication); t0 and t1 are their extent.
// `fill` is a colour from the data; the others follow from `style` and the palette (style.js).
// Items are sorted by time in every row, for `forEachItemIn`.
'use strict';

import { rankCount, layoutRows, rankSegments, segmentExtent, collectiveParts, commSegments } from './layout.js';
import { METRICS } from './style.js';
import { formatDuration as fmt } from './axis.js';

const timed = n => n.data.start != null && n.data.end != null;
const intervalsOf = d => d.intervals || [[d.start, d.end]];
const regionIntervalsOf = d => (d.region_intervals && d.region_intervals.length) ? d.region_intervals : [[d.region_start, d.region_end]];
const suffix = (iter, total) => total > 1 ? ' ' + (iter + 1) : '';

/**
 * The model of the timeline nodes, or null if there is nothing timed to show.
 * options.colorOf(data): the fill of a call or a transfer. options.commExpanded: a row for every rank to show what it did in the collectives.
 */
export function buildTimeline(nodes, { colorOf = () => null, commExpanded = false } = {}) {
    const calls     = nodes.filter(n => timed(n) && (n.data.type === 'skeleton_call' || n.data.type === 'external'));
    const transfers = nodes.filter(n => n.data.type === 'transfer' && timed(n));
    const builds    = nodes.filter(n => n.data.type === 'compile' && timed(n));
    const comms     = nodes.filter(n => n.data.type === 'communication' && timed(n));
    const markers   = nodes.filter(n => n.data.timestamp != null && ['allocation', 'deallocation', 'snapshot'].includes(n.data.type));
    const regions   = nodes.filter(n => n.data.type === 'region' && n.data.region_start != null && n.data.region_end != null);
    if (!calls.length && !markers.length && !regions.length) return null;

    // ── the time ──────────────────────────────────────────────────────────────
    let minT = Infinity, maxT = -Infinity;
    const see = (a, b = a) => { minT = Math.min(minT, a); maxT = Math.max(maxT, b); };
    for (const group of [calls, transfers, builds, comms]) for (const n of group) for (const iv of intervalsOf(n.data)) see(iv[0], iv[1]);
    for (const n of markers) see(n.data.timestamp);
    for (const n of regions) for (const iv of regionIntervalsOf(n.data)) see(iv[0], iv[1]);
    // what the other ranks did can be outside the interval of rank 0
    const extent = segmentExtent(calls);
    if (extent) see(extent[0], extent[1]);
    for (const n of comms) for (const s of n.data.rank_summaries || []) if (s && s.interval) see(s.interval.start, s.interval.end);
    const range = { minT, maxT, span: Math.max(maxT - minT, 1) };

    // ── the rows ──────────────────────────────────────────────────────────────
    const ranks = rankCount(calls);   // 0: not an MPI program, the rows are those of one process
    const backends = [...new Set(calls.map(n => n.data.backend || 'CPU'))].sort();
    const { rowH, headerH, axisH } = METRICS;
    const rankComm = ranks > 0 && comms.length > 0 && commExpanded;
    const L = layoutRows({ ranks, backends, regionH: regions.length ? rowH : 0, rowH, headerH,
        transferH: transfers.length ? rowH : 0, compileH: builds.length ? rowH : 0, commH: comms.length ? rowH : 0,
        eventH: markers.length ? rowH : 0, axisH, rankComm });

    const rows = [];
    const addRow = (kind, label, y, h, shade, extra) => rows.push({ kind, label, y, h, shade, ...extra }) - 1;
    const rowOf = { regions: -1, transfers: -1, builds: -1, comm: -1, events: -1, backend: new Map(), rankComm: new Map() };
    if (regions.length) rowOf.regions = addRow('regions', 'Regions', 0, rowH, 'rowA');
    for (const g of L.groups) {
        if (g.headerH > 0) addRow('rank', 'Rank ' + g.rank, g.headerY, g.headerH, 'rowRank', { rank: g.rank });
        backends.forEach((b, i) => rowOf.backend.set(`${g.rank}/${b}`, addRow('backend', b, g.rowY[b], rowH, i % 2 ? 'rowOdd' : 'rowEven', { rank: g.rank })));
        if (g.commY != null) rowOf.rankComm.set(g.rank, addRow('rank-comm', 'MPI', g.commY, rowH, 'rowComm', { rank: g.rank }));
    }
    if (transfers.length) rowOf.transfers = addRow('transfers', 'Transfers', L.transferY, rowH, 'rowTransfer');
    if (builds.length)    rowOf.builds    = addRow('builds', 'Builds', L.compileY, rowH, 'rowBuild');
    if (comms.length)     rowOf.comm      = addRow('comm', 'Comm.', L.commY, rowH, 'rowComm', ranks > 0 ? { toggle: rankComm ? 'open' : 'closed' } : {});
    if (markers.length)   rowOf.events    = addRow('events', 'Events', L.eventY, rowH, 'rowEvent');

    // ── the items ─────────────────────────────────────────────────────────────
    const items = [];
    const add = it => { it.idx = items.length; it.t0 = Math.min(...it.spans.map(s => s.t0)); it.t1 = Math.max(...it.spans.map(s => s.t1)); items.push(it); return it; };
    const bar = (row, style, d, iter, total, spans, extra) =>
        add({ row, style, shape: 'bar', id: d.id, iter, rank: null, spans, fill: null, depth: 0, hidden: !!d.hidden,
              label: d.label + suffix(iter, total), ...extra });
    const whole = (t0, t1) => [{ t0, t1, light: false }];

    for (const n of regions) {
        const d = n.data, depth = d.region_depth !== undefined ? d.region_depth : (d.nesting_level || 0);
        regionIntervalsOf(d).forEach((iv, i) => bar(rowOf.regions, 'region', d, i, 1, whole(iv[0], iv[1]), { depth, label: d.label, src: { d, iv } }));
    }

    for (const n of calls) {
        const d = n.data, backend = d.backend || 'CPU', fill = colorOf(d), ivs = intervalsOf(d);
        ivs.forEach((iv, i) => {
            const segs = ranks > 0 ? rankSegments((d.rank_datas || [])[i]) : [];
            const list = segs.length ? segs : [{ rank: 0, start: iv[0], end: iv[1], workStart: null, workEnd: null, plain: true }];
            for (const s of list) {
                const spans = s.plain ? whole(s.start, s.end) : [{ t0: s.start, t1: s.end, light: true }];
                if (!s.plain && s.workStart != null) spans.push({ t0: s.workStart, t1: s.workEnd, light: false });
                bar(rowOf.backend.get(`${ranks > 0 ? s.rank : null}/${backend}`), 'call', d, i, ivs.length, spans,
                    { fill, rank: ranks > 0 ? s.rank : null, lightText: ranks > 0 && !s.plain, src: { d, iv, seg: s, total: ivs.length } });
            }
        });
    }

    for (const [group, row, style] of [[transfers, rowOf.transfers, 'transfer'], [builds, rowOf.builds, d => d.ok === false ? 'build-failed' : 'build']])
        for (const n of group) {
            const d = n.data, ivs = intervalsOf(d);
            ivs.forEach((iv, i) => bar(row, typeof style === 'function' ? style(d) : style, d, i, ivs.length, whole(iv[0], iv[1]),
                { fill: style === 'transfer' ? colorOf(d) : null, src: { d, iv, total: ivs.length } }));
        }

    for (const n of comms) {
        const d = n.data, ivs = intervalsOf(d);
        ivs.forEach((iv, i) => {
            const summary = (d.rank_summaries || [])[i], parts = collectiveParts(summary);
            // a collective of an MPI program: the light part is the wait for the last rank to arrive, then the exchange
            const spans = parts ? [{ t0: parts.waitStart, t1: parts.waitEnd, light: true }, { t0: parts.exchangeStart, t1: parts.end, light: false }] : whole(iv[0], iv[1]);
            bar(rowOf.comm, 'comm', d, i, ivs.length, spans, { src: { d, iv, summary: parts ? summary : null, total: ivs.length } });
            if (rankComm)
                commSegments((d.rank_datas || [])[i], summary).forEach(s => {
                    const row = rowOf.rankComm.get(s.rank);
                    if (row == null) return;
                    const spans = s.exchangeStart > s.start ? [{ t0: s.start, t1: s.exchangeStart, light: true }, { t0: s.exchangeStart, t1: s.end, light: false }] : whole(s.start, s.end);
                    bar(row, 'comm', d, i, ivs.length, spans, { rank: s.rank, label: '', src: { d, iv, rankComm: s, total: ivs.length } });
                });
        });
    }

    for (const n of markers) {
        const d = n.data;
        add({ row: rowOf.events, style: d.type, shape: 'marker', id: d.id, iter: 0, rank: null, spans: [{ t0: d.timestamp, t1: d.timestamp, light: false }],
              fill: null, depth: 0, hidden: !!d.hidden, label: d.label, src: { d } });
    }

    // ── indexes ───────────────────────────────────────────────────────────────
    const byRow = rows.map(() => []);
    const byNode = new Map();
    for (const it of items) {
        byRow[it.row].push(it);
        if (!byNode.has(it.id)) byNode.set(it.id, []);
        byNode.get(it.id).push(it);
    }
    // by start, nested regions after the ones they are in; reach[k]: the latest end of items 0..k
    const reach = byRow.map(list => {
        list.sort((a, b) => a.t0 - b.t0 || a.depth - b.depth);
        let m = -Infinity;
        return list.map(it => (m = Math.max(m, it.t1)));
    });

    let count = markers.length;
    for (const group of [calls, transfers, builds, comms]) for (const n of group) count += intervalsOf(n.data).length;
    for (const n of regions) count += regionIntervalsOf(n.data).length;

    return { range, ranks, rows, rowsHeight: L.axisY, height: L.height, naturalHeight: L.height, items, byRow, reach, byNode, count, rankComm };
}

const isFlexRow = row => row.kind !== 'rank';

/**
 * Stacks the rows, `rowH` high for all but the rank headers, and updates the heights of the model. Items name their row, so they need no change.
 */
export function stackRows(model, rowH) {
    let y = 0;
    for (const row of model.rows) {
        row.y = y;
        row.h = row.kind === 'rank' ? METRICS.headerH : rowH;
        y += row.h;
    }
    model.rowsHeight = y;
    model.height = y + METRICS.axisH;   // naturalHeight is that of the normal row height
}

/** The flexible row height that fills `rowsPx` (without the axis), between METRICS.rowH and METRICS.maxRowH. */
export function fitRowHeight(model, rowsPx) {
    const flex = model.rows.filter(isFlexRow).length;
    if (!flex) return METRICS.rowH;
    const fixed = model.rows.reduce((h, row) => h + (isFlexRow(row) ? 0 : row.h), 0);
    return Math.min(METRICS.maxRowH, Math.max(METRICS.rowH, Math.floor((rowsPx - fixed) / flex)));
}

/** Calls `fn(item)` for the items of a row that overlap [t0, t1], in the order they are drawn. */
export function forEachItemIn(model, row, t0, t1, fn) {
    const list = model.byRow[row], reach = model.reach[row];
    let lo = 0, hi = list.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (reach[mid] < t0) lo = mid + 1; else hi = mid; }
    for (let k = lo; k < list.length && list[k].t0 <= t1; k++) if (list[k].t1 >= t0) fn(list[k]);
}

/** The item of a node that is the `iter`-th interval of it, or else the first one it has. */
export function itemOfNode(model, id, iter = 0) {
    const list = model.byNode.get(id);
    return list ? (list.find(it => it.iter === iter) ?? list[0]) : null;
}

export function tooltip(item, model) {
    const { d, iv, seg, summary, rankComm, total } = item.src;
    const iterLabel = total > 1 ? ` [${item.iter + 1}/${total}]` : '';
    switch (item.style) {
        case 'region': return `${d.label}\nRegion [depth ${d.nesting_level || 0}]\n${fmt(iv[1] - iv[0])}`;
        case 'call': {
            let tip = d.label + (d.pattern ? ` [${d.pattern}]` : '') + iterLabel + '\n' + (d.backend || '');
            if (model.ranks > 0) {
                tip += `\nRank ${seg.rank}: ${fmt(seg.end - seg.start)}`;
                if (seg.workStart != null) tip += `, worked ${fmt(seg.workEnd - seg.workStart)}`;
                else if (!seg.plain) tip += ', no work';
                if (!seg.plain) tip += `\n${seg.localElements} elements`;
                if (seg.comm && seg.comm.collectives)
                    tip += `\ncommunication: ${seg.comm.collectives} collective${seg.comm.collectives > 1 ? 's' : ''}, ${seg.comm.bytes_sent + seg.comm.bytes_received} bytes, ${fmt(seg.comm.time_us)}`;
            } else tip += '\n' + fmt(iv[1] - iv[0]);
            return tip;
        }
        case 'transfer': return `${d.label}${iterLabel}\n${d.direction || 'transfer'}\n${fmt(iv[1] - iv[0])}`;
        case 'build': case 'build-failed':
            return `${d.label}${iterLabel}\n${d.device || 'OpenCL'}${d.ok === false ? '\nthe build failed' : ''}\n${fmt(iv[1] - iv[0])}`;
        case 'comm':
            if (rankComm)
                return `${d.label}\nRank ${rankComm.rank}: ${fmt(rankComm.end - rankComm.start)} in it\nwaited for the last to arrive: ${fmt(rankComm.exchangeStart - rankComm.start)}\nsent ${rankComm.bytesSent} bytes, received ${rankComm.bytesReceived}`;
            if (summary)
                return `${d.label}${iterLabel}\n${d.operation || 'communication'}\nall ranks: ${fmt(summary.interval.end - summary.interval.start)}\nwaiting for the last to arrive: ${fmt(summary.max_wait_us)} (rank ${summary.last_arrival_rank})\n${summary.bytes_sent} bytes moved`;
            return `${d.label}${iterLabel}\n${d.operation || 'communication'}\n${fmt(iv[1] - iv[0])}`;
        default: return `${d.label}\n${d.type}\n@+${fmt(d.timestamp - model.range.minT)}`;
    }
}
