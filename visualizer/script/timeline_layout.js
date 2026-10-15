// timeline_layout.js — the rows of the timeline, and what is drawn in them for an MPI program.  Pure: no DOM.
//
// A trace of a single node program has a row for each backend (the contents of "one rank"). In an MPI program the ranks are
// the top level: each rank has a header and a row for each backend, so that the timeline looks like the hardware does. What
// happens between the ranks, the collective operations, has a row of its own, as it does not belong to a rank.
//
//   regions | rank 0: header, backend rows | rank 1: ... | transfers | kernel builds | communication | events | axis
'use strict';

/** The number of ranks that the timeline nodes (the data of the timed calls) have data of: 0 if they are not from an MPI program. */
export function rankCount(nodes) {
    let ranks = 0;
    for (const n of nodes) {
        const d = n.data ?? n;
        for (const rd of d.rank_datas ?? []) if (rd) ranks = Math.max(ranks, rd.length);
    }
    return ranks;
}

/**
 * The y of the rows. For a trace that is not from an MPI program (ranks = 0) there is one group of backend rows and no header.
 * heights: { regionH, rowH, headerH, transferH, compileH, commH, eventH, axisH } (a row that is not there has the height 0).
 * With rankComm (the collectives expanded) every rank also has a row for what it did in them: rowY.commY of its group.
 * Returns { groups: [{ rank, headerY, headerH, rowY: { backend: y }, commY }], transferY, compileY, commY, eventY, axisY, height }.
 */
export function layoutRows({ ranks, backends, regionH, rowH, headerH, transferH, compileH = 0, commH, eventH, axisH, rankComm = false }) {
    const groups = [];
    let y = regionH;
    const groupCount = ranks > 0 ? ranks : 1;
    for (let r = 0; r < groupCount; r++) {
        const g = { rank: ranks > 0 ? r : null, headerY: y, headerH: ranks > 0 ? headerH : 0, rowY: {} };
        y += g.headerH;
        for (const b of backends) { g.rowY[b] = y; y += rowH; }
        if (rankComm && ranks > 0) { g.commY = y; y += rowH; }
        groups.push(g);
    }
    const transferY = y; y += transferH;
    const compileY  = y; y += compileH;
    const commY     = y; y += commH;
    const eventY    = y; y += eventH;
    return { groups, transferY, compileY, commY, eventY, axisY: y, height: y + axisH };
}

/**
 * What every rank did in one iteration of a call (rank_data of mpi.js): the interval of the rank, and the part of it that the
 * rank worked (the rest is communication and waiting). A rank that has no data (its file is missing) is left out.
 */
export function rankSegments(rankData) {
    const out = [];
    (rankData ?? []).forEach(e => {
        if (!e) return;
        out.push({ rank: e.rank, start: e.start, end: e.end,
                   workStart: e.work_start ?? null, workEnd: e.work_end ?? null, localElements: e.local_elements ?? 0,
                   comm: e.comm ?? null });
    });
    return out;
}

/** The first and the last time of the segments of all ranks of the timed calls (so that the time axis has them). */
export function segmentExtent(nodes) {
    let min = Infinity, max = -Infinity;
    for (const n of nodes) {
        const d = n.data ?? n;
        for (const rd of d.rank_datas ?? [])
            for (const s of rankSegments(rd)) { min = Math.min(min, s.start); max = Math.max(max, s.end); }
    }
    return min <= max ? [min, max] : null;
}

/**
 * The two parts of the bar of a collective operation: the processes arrive (waiting for the last one), then they exchange.
 * summary: rank_summary of the collective. Returns null if there is none (a collective of a single trace file).
 */
export function collectiveParts(summary) {
    if (!summary?.interval) return null;
    const first = summary.interval.start, last = first + (summary.max_wait_us ?? 0);
    return { waitStart: first, waitEnd: Math.min(last, summary.interval.end), exchangeStart: Math.min(last, summary.interval.end), end: summary.interval.end };
}

/**
 * What every rank did in a collective operation (rank_data of the collective, with its summary): the rank arrived at `start`, waited until
 * the last rank arrived (`exchangeStart`), and left at `end`. Ranks without data are left out.
 */
export function commSegments(rankData, summary) {
    const last = (summary?.interval?.start ?? 0) + (summary?.max_wait_us ?? 0);
    const out = [];
    (rankData ?? []).forEach(e => {
        if (!e) return;
        out.push({ rank: e.rank, start: e.start, end: e.end, exchangeStart: Math.min(Math.max(last, e.start), e.end),
                   bytesSent: e.bytes_sent ?? 0, bytesReceived: e.bytes_received ?? 0 });
    });
    return out;
}
