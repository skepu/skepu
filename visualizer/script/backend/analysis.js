// analysis.js — questions about a SkePU trace, answered from its events.
//
// Pure functions over the events of parseTrace() (trace.js): no graph, no I/O, no state, so
// they work in Node, in tests and in a server alike. Times are microseconds, as in the trace.
// Graph based analyses (dependences, the critical path) are in graph_analysis.js.
'use strict';

export const duration = e => (e.start != null && e.end != null) ? e.end - e.start : 0;
export const eventId  = e => 'e' + e.idx;

const isCall = e => e.type === 'skeleton_call';

/** The compact form of an event that the analyses return. */
export function describe(e) {
    const d = { id: eventId(e), type: e.type, label: e.label };
    if (e.pattern  != null) d.pattern  = e.pattern;
    if (e.backend  != null) d.backend  = e.backend;
    if (e.start    != null) { d.start = e.start; d.end = e.end; d.duration_us = duration(e); }
    else if (e.time != null) d.time = e.time;
    if (e.elements != null) d.elements = e.elements;
    if (e.direction != null) d.direction = e.direction;
    if (e.file) d.file = e.file;
    if (e.line != null) d.line = e.line;
    if (e.region != null) d.region = e.region;
    if (e.type === 'transfer') d.transfer_kind = transferKind(e);
    return d;
}

// ── Transfers ────────────────────────────────────────────────────────────────

/**
 * The kind of a transfer. Transfers of the library itself carry 'Internal' in their label:
 * 'Internal (reduce-temp)' (partial results of a reduction brought to the host),
 * 'Internal (overlap-wrap)' (edge data of a MapOverlap), '[Internal PRNG]' (random number state).
 * All others are transfers of containers ('container').
 * (A convention of the labels; the tracer does not record it as a field.)
 */
export function transferKind(e) {
    const label = e.label ?? '';
    if (!/internal/i.test(label)) return 'container';
    const m = /\(([^)]+)\)/.exec(label);
    if (m) return m[1].trim().toLowerCase().replace(/\s+/g, '-');
    if (/prng/i.test(label)) return 'prng';
    return 'internal';
}

// The skeleton call a transfer belongs to: the one whose time span contains it (on the same
// backend; the innermost if there are several), else the next call after it in the trace on
// that backend. Calls are recorded when they finish, so the transfers of a call come before it
// in the trace; some (the edge data of a MapOverlap) happen just before the call starts.
// Returns a Map transfer -> { call, match: 'contained' | 'next' }.
function transferOwners(events) {
    const calls = events.filter(e => isCall(e) && e.start != null).sort((a, b) => a.start - b.start);
    const byIdx = events.filter(isCall).sort((a, b) => a.idx - b.idx);
    const owner = new Map();
    for (const t of events) {
        if (t.type !== 'transfer' || t.start == null) continue;
        let lo = 0, hi = calls.length;                       // the last call that starts at or before t
        while (lo < hi) { const mid = (lo + hi) >>> 1; if (calls[mid].start <= t.start) lo = mid + 1; else hi = mid; }
        let best = null;
        for (let i = lo - 1, n = 0; i >= 0 && n < 64; i--, n++) {
            const c = calls[i];
            if (c.end >= t.end && (c.backend == null || c.backend === t.backend)
                && (best === null || duration(c) < duration(best))) best = c;
        }
        if (best) { owner.set(t, { call: best, match: 'contained' }); continue; }
        let a = 0, z = byIdx.length;                         // the first call recorded after t
        while (a < z) { const mid = (a + z) >>> 1; if (byIdx[mid].idx > t.idx) z = mid; else a = mid + 1; }
        for (let i = a; i < byIdx.length && i < a + 8; i++)
            if (byIdx[i].backend == null || byIdx[i].backend === t.backend) { owner.set(t, { call: byIdx[i], match: 'next' }); break; }
    }
    return owner;
}

/**
 * All transfers, by kind and direction, with the owning skeleton call of the internal ones,
 * and per container (how often and in which direction it moves).
 */
export function transfers(events, { kind = null, limit = 20 } = {}) {
    const all   = events.filter(e => e.type === 'transfer');
    const owner = transferOwners(events);
    const sel   = kind ? all.filter(t => transferKind(t) === kind) : all;

    const byKind = {};
    for (const t of all) {
        const k = transferKind(t);
        const g = (byKind[k] ??= { count: 0, total_us: 0, host_to_device: 0, device_to_host: 0 });
        g.count++; g.total_us += duration(t);
        if (t.direction === 'host-to-device') g.host_to_device++; else if (t.direction === 'device-to-host') g.device_to_host++;
    }
    const total = all.reduce((s, t) => s + duration(t), 0);

    // Containers: direction changes of a container's transfers in time order are a ping-pong.
    const perObj = new Map();
    for (const t of all) {
        if (transferKind(t) !== 'container') continue;
        let g = perObj.get(t.object_id);
        if (!g) perObj.set(t.object_id, g = { object_id: t.object_id, label: t.label, count: 0, total_us: 0, direction_changes: 0, last: null });
        g.count++; g.total_us += duration(t);
        if (g.last && g.last !== t.direction) g.direction_changes++;
        g.last = t.direction;
    }
    const containers = [...perObj.values()].map(({ last, ...g }) => g)
        .sort((a, b) => b.total_us - a.total_us).slice(0, limit);

    // Internal transfers per owning skeleton call pattern.
    const byOwner = {};
    for (const t of all) {
        if (transferKind(t) === 'container') continue;
        const o = owner.get(t)?.call;
        const key = o ? `${o.pattern ?? 'call'}` : 'unattributed';
        const g = (byOwner[key] ??= { count: 0, total_us: 0 });
        g.count++; g.total_us += duration(t);
    }

    const rows = sel.sort((a, b) => duration(b) - duration(a)).slice(0, limit).map(t => {
        const d = describe(t);
        const o = owner.get(t);
        if (o) d.owner = { id: eventId(o.call), label: o.call.label, pattern: o.call.pattern, duration_us: duration(o.call), match: o.match };
        return d;
    });

    return {
        count: all.length, total_us: total, by_kind: byKind, internal_by_owner_pattern: byOwner,
        container_transfers: containers,
        longest: rows, selected: sel.length,
    };
}

// ── Time ─────────────────────────────────────────────────────────────────────

const GROUP_KEYS = {
    pattern:   e => e.pattern ?? '(none)',
    label:     e => e.label ?? '(none)',
    backend:   e => e.backend ?? '(none)',
    region:    e => e.region ?? '(none)',
    file:      e => e.file || '(none)',
    file_line: e => e.file ? `${e.file}:${e.line}` : '(none)',
};

/**
 * Time of the events of a type (skeleton calls by default) grouped by pattern, label, backend,
 * region, file or file_line: count, total, mean, maximum and share of the total.
 */
export function timeBy(events, groupBy = 'pattern', { type = 'skeleton_call', top = 20 } = {}) {
    const key = GROUP_KEYS[groupBy];
    if (!key) throw new Error(`group_by must be one of: ${Object.keys(GROUP_KEYS).join(', ')}`);
    const groups = new Map();
    let total = 0;
    for (const e of events) {
        if (e.type !== type) continue;
        const k = key(e), d = duration(e);
        let g = groups.get(k);
        if (!g) groups.set(k, g = { key: k, count: 0, total_us: 0, max_us: 0 });
        g.count++; g.total_us += d; g.max_us = Math.max(g.max_us, d); total += d;
    }
    const rows = [...groups.values()].sort((a, b) => b.total_us - a.total_us);
    return {
        type, group_by: groupBy, groups: rows.length, total_us: total,
        rows: rows.slice(0, top).map(g => ({ ...g, mean_us: Math.round(g.total_us / g.count), share: total ? +(g.total_us / total).toFixed(4) : 0 })),
    };
}

// ── Overview ─────────────────────────────────────────────────────────────────

export function summarize(events, header = null) {
    const byType = {};
    let first = Infinity, last = -Infinity;
    const backends = new Set(), regions = new Set(), files = new Set();
    let callTime = 0, calls = 0;
    for (const e of events) {
        byType[e.type] = (byType[e.type] ?? 0) + 1;
        const a = e.start ?? e.time, b = e.end ?? e.time;
        if (a != null) first = Math.min(first, a);
        if (b != null) last  = Math.max(last, b);
        if (e.backend) backends.add(e.backend);
        if (e.region)  regions.add(e.region);
        if (e.file)    files.add(e.file);
        if (isCall(e)) { calls++; callTime += duration(e); }
    }
    const tr = transfers(events, { limit: 0 });
    const span = last >= first ? last - first : 0;
    return {
        events: events.length, by_type: byType,
        time_us: { first: isFinite(first) ? first : null, last: isFinite(last) ? last : null, span },
        backends: [...backends], regions: [...regions], files: [...files],
        skeleton_calls: { count: calls, total_us: callTime },
        transfers: { count: tr.count, total_us: tr.total_us, by_kind: tr.by_kind },
        header: header ? { files: Object.keys(header).length } : null,
    };
}

// ── Queries ──────────────────────────────────────────────────────────────────

/** Events matching a filter, paged; sorted by trace order or by duration. */
export function findEvents(events, q = {}) {
    const { type, label, file, line, region, backend, pattern, min_duration_us, sort_by = 'trace', limit = 50, offset = 0 } = q;
    const lc = label ? String(label).toLowerCase() : null;
    const hits = events.filter(e =>
        (!type    || e.type === type) &&
        (!lc      || (e.label ?? '').toLowerCase().includes(lc)) &&
        (!file    || (e.file ?? '').endsWith(file)) &&
        (line == null || e.line === line) &&
        (!region  || e.region === region) &&
        (!backend || e.backend === backend) &&
        (!pattern || e.pattern === pattern) &&
        (min_duration_us == null || duration(e) >= min_duration_us));
    if (sort_by === 'duration') hits.sort((a, b) => duration(b) - duration(a));
    const page = hits.slice(offset, offset + limit);
    return { total: hits.length, offset, returned: page.length, events: page.map(describe) };
}

/** The data containers: allocation, transfers, and how many skeleton calls read and write them. */
export function containers(events, { top = 20 } = {}) {
    const m = new Map();
    const get = (id, label) => { let c = m.get(id); if (!c) m.set(id, c = { object_id: id, label: label ?? null, allocations: 0, transfers: 0, transfer_us: 0, reads: 0, writes: 0 }); if (c.label == null && label) c.label = label; return c; };
    for (const e of events) {
        if (e.type === 'allocation')  get(e.object_id, e.label).allocations++;
        else if (e.type === 'transfer' && transferKind(e) === 'container') { const c = get(e.object_id, e.label); c.transfers++; c.transfer_us += duration(e); }
        else if (isCall(e)) {
            for (const id of [...(e.elwise_inputs ?? []), ...(e.proxy_inputs ?? []), ...(e.scalar_inputs ?? [])]) get(id).reads++;
            for (const id of e.outputs ?? []) get(id).writes++;
        }
    }
    const rows = [...m.values()].sort((a, b) => b.transfer_us - a.transfer_us || b.writes - a.writes);
    return { count: rows.length, containers: rows.slice(0, top) };
}

/** The events of a source line (the file is matched by its end, so a base name will do). */
export function sourceLine(events, file, line) {
    const hits = events.filter(e => e.line === line && (e.file ?? '').endsWith(file));
    return {
        file, line, count: hits.length,
        total_us: hits.reduce((s, e) => s + duration(e), 0),
        events: hits.slice(0, 50).map(describe),
    };
}
