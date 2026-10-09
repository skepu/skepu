// mpi.js — the trace files of an MPI program, merged.  Pure; usable in the worker, in Node and in tests.
'use strict';

// A program built with the MPI backend writes a file for each process (see TRACE_FORMAT.md, "MPI programs"):
//
//  * the file of rank 0 has the trace of the program: the events of a single node program, a `seq` for each call
//    and a `comm_seq` for each collective operation (the same on every process: the processes make the same calls and the
//    same collectives in the same order), and what rank 0 did in them;
//  * the files of the other ranks are lean: a `rank_call` for each call and a `rank_communication` for each collective,
//    with the same `seq` and `comm_seq`, the interval, and a few numbers.
//
// Every file has the clock of its process, and a header that says how to put it on the clock of rank 0.
//
// The merged trace is the trace of rank 0 (its events and their `idx` are those of the file, so that the trace text and
// the events stay together), with, on each call and each collective, what every rank did:
//
//   rank_data     one entry for each rank (null for a rank whose file is missing), with the times on the clock of rank 0
//   rank_summary  the interval of the call on all ranks, and min/mean/max of what the ranks did (with the rank of the max)
//
// A merged trace is read like any other: nothing is removed from the events, and what is added has names that
// the trace does not use.

import { parseTrace } from './trace.js';

/** Whether the header of a trace file says that it belongs to an MPI program */
export function isMpiHeader(header) {
    return !!(header && header.mpi && typeof header.mpi.rank === 'number');
}

/**
 * The time of a process on the clock of rank 0.  The header has the offset (what to add) at the start and at the end
 * of the program; in between it is interpolated, so that a clock that runs fast or slow is followed.
 */
export function alignedTime(clock, t) {
    const a = clock?.start, b = clock?.end;
    if (!a) return t;
    let offset = a.offset_us;
    if (b && b.local_us > a.local_us)
        offset = a.offset_us + (b.offset_us - a.offset_us) * (t - a.local_us) / (b.local_us - a.local_us);
    return t + offset;
}

/** The statistics of values: { min, mean, max, argmax (the index of the max) }, or null for no values */
export function stats(values) {
    let n = 0, sum = 0, min = Infinity, max = -Infinity, argmax = -1;
    values.forEach((v, i) => {
        if (v == null) return;
        n++;
        sum += v;
        if (v < min) min = v;
        if (v > max) { max = v; argmax = i; }
    });
    return n ? { min, mean: sum / n, max, argmax } : null;
}

const ratio = s => (s && s.mean > 0) ? s.max / s.mean : null;

const zeroComm = { bytes_sent: 0, bytes_received: 0, time_us: 0, collectives: 0 };

// What a process did in a call, on the clock of rank 0
function callEntry(rank, clock, e, local) {
    const align = t => alignedTime(clock, t);
    const worked = local.work_end > 0 || local.work_start > 0;
    return {
        rank,
        start: align(e.start), end: align(e.end),
        work_start: worked ? align(local.work_start) : null,
        work_end:   worked ? align(local.work_end) : null,
        local_elements: local.local_elements ?? 0,
        comm: { ...zeroComm, ...(local.comm ?? {}) },
    };
}

function callSummary(entries) {
    const present = entries.filter(Boolean);
    const at = f => entries.map(x => x ? f(x) : null);
    // (a rank that did no work counts as zero: the mean is over all ranks, so that a call that one rank does alone is as unbalanced as it is)
    const work = stats(at(x => x.work_start != null ? x.work_end - x.work_start : 0));
    return {
        interval: { start: Math.min(...present.map(x => x.start)), end: Math.max(...present.map(x => x.end)) },
        arrival_spread_us: Math.max(...present.map(x => x.start)) - Math.min(...present.map(x => x.start)),
        duration: stats(at(x => x.end - x.start)),
        work,
        comm_time: stats(at(x => x.comm.time_us)),
        comm_bytes: stats(at(x => x.comm.bytes_sent + x.comm.bytes_received)),
        local_elements: stats(at(x => x.local_elements)),
        // The slowest rank is the one that worked the longest; the imbalance is how much longer than the mean over all ranks (1: balanced, the number of ranks: one rank does it all)
        slowest_rank: work ? work.argmax : null,
        imbalance: ratio(work),
    };
}

function commEntry(rank, clock, e) {
    return { rank, start: alignedTime(clock, e.start), end: alignedTime(clock, e.end),
             bytes_sent: e.bytes_sent ?? 0, bytes_received: e.bytes_received ?? 0 };
}

function commSummary(entries) {
    const present = entries.filter(Boolean);
    const first = Math.min(...present.map(x => x.start)), last = Math.max(...present.map(x => x.start));
    const ends = present.map(x => x.end);
    return {
        interval: { start: first, end: Math.max(...ends) },
        // A collective ends when the last process has arrived: what a process waited is the distance to the last arrival
        wait_us: entries.map(x => x ? last - x.start : null),
        max_wait_us: last - first,
        last_arrival_rank: entries.findIndex(x => x && x.start === last),
        end_spread_us: Math.max(...ends) - Math.min(...ends),
        bytes_sent: present.reduce((s, x) => s + x.bytes_sent, 0),
        bytes_received: present.reduce((s, x) => s + x.bytes_received, 0),
    };
}

// The collectives that no process leaves before all have arrived
const SYNCHRONIZING = new Set(['allgather', 'allgatherv', 'allreduce', 'barrier']);

function toTrace(file) {
    if (file.trace) return file.trace;
    return parseTrace(file.text);
}

/**
 * Merge the trace files of an MPI program.
 *
 * files: [{ name, text }] (the text of a trace file) or [{ name, trace }] (the result of parseTrace).
 *
 * Returns { header, events, mpi, warnings, divergence }:
 *   header      the header of the file of rank 0
 *   events      the events of rank 0 with rank_data and rank_summary on the calls and the collectives
 *   mpi         { ranks, files: [{ name, rank, role, host, pid, uncertainty_us }], missing_ranks, uncertainty_us }
 *   warnings    what is wrong with the set (a missing rank, the processes not agreeing, a clock that is not aligned well)
 *   divergence  the first call or collective that the processes do not agree on, or null
 *
 * Throws if the files are not the files of one MPI program or have no file of rank 0.
 */
export function mergeMpiTraces(files) {
    if (!files.length) throw new Error('No trace files');
    const parsed = files.map(f => ({ name: f.name ?? '(trace)', ...toTrace(f) }));
    for (const f of parsed)
        if (!isMpiHeader(f.header))
            throw new Error(`${f.name}: not a trace of an MPI program (no mpi in the header)`);

    const ranks = parsed[0].header.mpi.ranks;
    const warnings = [];
    for (const f of parsed) {
        const m = f.header.mpi;
        if (m.ranks !== ranks) throw new Error(`${f.name}: the files are of programs with different numbers of processes (${m.ranks} and ${ranks})`);
        if (!(m.rank >= 0 && m.rank < ranks)) throw new Error(`${f.name}: rank ${m.rank} of ${ranks}`);
    }
    const byRank = new Map();
    for (const f of parsed) {
        const r = f.header.mpi.rank;
        if (byRank.has(r)) throw new Error(`${f.name}: rank ${r} is in two files (${byRank.get(r).name} too)`);
        byRank.set(r, f);
    }
    const detail = parsed[0].header.mpi.detail_rank ?? 0;
    const full = byRank.get(detail);
    if (!full || full.header.mpi.role !== 'full')
        throw new Error(`The file of rank ${detail}, which has the trace of the program, is missing`);
    for (const f of parsed)
        if ((f.header.mpi.role === 'full') !== (f.header.mpi.rank === detail))
            throw new Error(`${f.name}: rank ${f.header.mpi.rank} has the role ${f.header.mpi.role}`);

    const missing = [];
    for (let r = 0; r < ranks; r++) if (!byRank.has(r)) missing.push(r);
    if (missing.length) warnings.push(`No trace file for rank${missing.length > 1 ? 's' : ''} ${missing.join(', ')}: their times are missing`);

    const clocks = [], info = [];
    for (let r = 0; r < ranks; r++) {
        const f = byRank.get(r);
        clocks[r] = f ? f.header.clock : null;
        if (f) info.push({ name: f.name, rank: r, role: f.header.mpi.role, host: f.header.mpi.host, pid: f.header.mpi.pid,
                           uncertainty_us: Math.max(f.header.clock?.start?.uncertainty_us ?? 0, f.header.clock?.end?.uncertainty_us ?? 0) });
    }

    // The lean events of every rank, by call and by collective
    const leanCalls = [], leanComms = [];
    for (let r = 0; r < ranks; r++) {
        leanCalls[r] = new Map();
        leanComms[r] = new Map();
        const f = byRank.get(r);
        if (!f || r === detail) continue;
        for (const e of f.events) {
            if (e.type === 'rank_call') leanCalls[r].set(e.seq, e);
            else if (e.type === 'rank_communication') leanComms[r].set(e.comm_seq, e);
        }
    }

    let divergence = null;
    const diverge = (what) => { if (!divergence) divergence = what; warnings.push(what.message); };

    const events = full.events.map(e => ({ ...e }));
    const seenCalls = new Set(), seenComms = new Set();
    for (const e of events) {
        // (an skepu::external call is one too: rank 0 runs the code, and the others wait for it)
        if ((e.type === 'skeleton_call' || e.type === 'external') && e.seq != null && e.mpi) {
            const pattern = e.pattern ?? 'external';
            seenCalls.add(e.seq);
            const entries = [];
            for (let r = 0; r < ranks; r++) {
                if (r === detail) { entries[r] = callEntry(r, clocks[r], e, e.mpi); continue; }
                if (!byRank.has(r)) { entries[r] = null; continue; }
                const l = leanCalls[r].get(e.seq);
                if (!l) { diverge({ kind: 'call', seq: e.seq, rank: r, message: `Rank ${r} has no call ${e.seq} (${pattern}): the processes do not make the same calls` }); entries[r] = null; continue; }
                if (l.pattern !== pattern || l.signature !== e.mpi.signature)
                    diverge({ kind: 'call', seq: e.seq, rank: r, expected: pattern, found: l.pattern,
                              message: `Rank ${r} made a different call than rank ${detail} as call ${e.seq}: ${l.pattern} and ${pattern}` });
                entries[r] = callEntry(r, clocks[r], l, l);
            }
            e.rank_data = entries;
            e.rank_summary = callSummary(entries);
        } else if (e.type === 'communication' && e.comm_seq != null) {
            seenComms.add(e.comm_seq);
            const entries = [];
            for (let r = 0; r < ranks; r++) {
                if (r === detail) { entries[r] = commEntry(r, clocks[r], e); continue; }
                if (!byRank.has(r)) { entries[r] = null; continue; }
                const l = leanComms[r].get(e.comm_seq);
                if (!l) { diverge({ kind: 'communication', comm_seq: e.comm_seq, rank: r, message: `Rank ${r} has no collective ${e.comm_seq} (${e.operation}): the processes do not make the same collectives` }); entries[r] = null; continue; }
                if (l.operation !== e.operation)
                    diverge({ kind: 'communication', comm_seq: e.comm_seq, rank: r, expected: e.operation, found: l.operation,
                              message: `Rank ${r} made a different collective than rank ${detail} as collective ${e.comm_seq}: ${l.operation} and ${e.operation}` });
                entries[r] = commEntry(r, clocks[r], l);
            }
            e.rank_data = entries;
            e.rank_summary = commSummary(entries);
        }
    }
    for (let r = 0; r < ranks; r++) {
        if (r === detail || !byRank.has(r)) continue;
        const extraCalls = [...leanCalls[r].keys()].filter(s => !seenCalls.has(s)).length;
        const extraComms = [...leanComms[r].keys()].filter(s => !seenComms.has(s)).length;
        if (extraCalls || extraComms)
            diverge({ kind: 'extra', rank: r, message: `Rank ${r} has ${extraCalls} calls and ${extraComms} collectives that rank ${detail} does not have` });
    }

    // How well the clocks are aligned: the uncertainty of the measurement, and what the collectives say (those that need every process end at
    // about the same time. An alltoallv does not: a process that gets nothing from a late one can leave before it arrives)
    const uncertainty = Math.max(0, ...info.map(i => i.uncertainty_us));
    let worstSpread = 0;
    for (const e of events) if (e.type === 'communication' && e.rank_summary && SYNCHRONIZING.has(e.operation)) worstSpread = Math.max(worstSpread, e.rank_summary.end_spread_us);
    if (worstSpread > Math.max(1000, 20 * uncertainty))
        warnings.push(`The collectives end up to ${Math.round(worstSpread)} us apart on the clock of rank ${detail}: the clocks are not well aligned (measured to within ${Math.round(uncertainty)} us)`);

    return {
        header: full.header,
        events,
        mpi: { ranks, detail_rank: detail, files: info, missing_ranks: missing, uncertainty_us: uncertainty, collective_end_spread_us: worstSpread },
        warnings,
        divergence,
    };
}
