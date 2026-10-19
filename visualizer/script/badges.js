// badges.js — which gutter badges the source panes get from the nodes of a graph. Pure, no DOM.
'use strict';

// The nodes that have a badge at their line of a source file (the extension marks the lines of the same types: src/nodeLines.ts, which a
// test compares with this). A snapshot is the line of the SKEPU_TRACE_SNAPSHOT call.
export const SOURCE_TYPES = ['skeleton_call', 'external', 'region', 'snapshot'];

/**
 * @param {Array} badgeNodes        [{ data: { id, type, file, line, total_order, total_orders, locations,
 *                                     trace_index, trace_indices, hidden } }] from the graph response
 * @param {Array} traceLineRanges   [[first line, last line]] of each event in the trace file
 * @returns {{ badges, traceBadges, orders, sites }}
 *   badges       'file:line' -> { count, hiddenCount, entries: [{ id, iterIdx }], file, line, onlySnapshots } (a line of snapshots has no arrows)
 *   traceBadges  trace index -> { count, hiddenCount, entries, range }
 *   orders       node id -> total order of each of its iterations
 *   sites        the places where the instances and user functions of the calls are defined, for the marks and the
 *                links back to the calls: [{ role: 'instance' | 'function', file, line, end_line, ids: [node id] }]
 *
 * A node that stands for several iterations (iterations coalesced) has an entry for each iteration, at the
 * place where that iteration ran: node.data.locations has [file, line] for each iteration if they are not
 * all at the node's own place (the nodes of a key path can be called from different lines).
 */
export function collectBadges(badgeNodes, traceLineRanges) {
    const badges = {}, traceBadges = {}, orders = {}, siteMap = {};

    for (const { data } of badgeNodes) {
        const list = data.total_orders || [data.total_order];
        orders[data.id] = list;
        const iters = data.total_orders ? data.total_orders.length : 1;
        const hidden = !!data.hidden;

        // ── Where the instance and the user functions are defined: the calls that use the place ──────
        if (data.definitions) {
            const places = [['instance', data.definitions.instance], ...(data.definitions.functions || []).map(f => ['function', f])];
            for (const [role, p] of places) {
                if (!p || !p.file || !Number.isInteger(p.line)) continue;
                const end = p.end_line ?? p.line;
                const site = siteMap[`${role}|${p.file}|${p.line}|${end}`] ??= { role, file: p.file, line: p.line, end_line: end, ids: [] };
                if (!site.ids.includes(data.id)) site.ids.push(data.id);
            }
        }

        // ── Source-file badges: one entry per iteration, at the line of that iteration ─────────────
        if (SOURCE_TYPES.includes(data.type)) {
            for (let i = 0; i < iters; i++) {
                const [file, line] = (data.locations && data.locations[i]) || [data.file, data.line];
                if (!file || line === -1 || line == null) continue;
                const key = file + ':' + line;
                const b = badges[key] ??= { count: 0, hiddenCount: 0, entries: [], file, line, onlySnapshots: true };
                if (data.type !== 'snapshot') b.onlySnapshots = false;
                b.count++;
                if (hidden) b.hiddenCount++;
                b.entries.push({ id: data.id, iterIdx: i });
            }
        }

        // ── Trace-file badges ──────────────────────────────────────────────────────────────────────
        (data.trace_indices || [data.trace_index]).forEach((traceKey, i) => {
            const b = traceBadges[traceKey] ??= { count: 0, hiddenCount: 0, entries: [], range: traceLineRanges[traceKey] };
            b.count++;
            if (hidden) b.hiddenCount++;
            b.entries.push({ id: data.id, iterIdx: i });
        });
    }
    return { badges, traceBadges, orders, sites: Object.values(siteMap) };
}

/**
 * The nodes that get badges: all of them, but the regions when the regions are not shown. (The backend always has the regions in
 * the graph, for its layout: the graph draws them as invisible nodes, and their badges would show in the source panes.)
 */
export function badgeNodesToShow(badgeNodes, showRegions) {
    return showRegions ? badgeNodes : badgeNodes.filter(n => n.data.type !== 'region');
}

/**
 * The nodes of the timeline (Gantt chart): all of them, but the regions when the regions are not shown. (The frontend asks the backend for
 * the regions always, for the layout of the graph, so what is not shown has to be left out here: the chart has a row for the regions.)
 */
export function timelineNodesToShow(timelineNodes, showRegions) {
    return showRegions ? timelineNodes : timelineNodes.filter(n => n.data.type !== 'region');
}

/**
 * The marks of the definitions of one call in the source panes (the selected call): the lines of the instance and of the
 * user functions. Returns [{ role, file, from, to }] (the lines from..to of the file); a place that is not
 * in the source is left out by the pane, which knows its files.
 */
export function definitionMarks(definitions) {
    if (!definitions) return [];
    const marks = [];
    const add = (role, p) => {
        if (p && p.file && Number.isInteger(p.line)) marks.push({ role, file: p.file, from: p.line, to: Math.max(p.line, p.end_line ?? p.line) });
    };
    add('instance', definitions.instance);
    for (const f of definitions.user_functions ?? definitions.functions ?? []) add('function', f);
    return marks;
}

/**
 * The places that a node's definitions has, in the order of the shortcuts: the instance, then the user functions.
 * `which` is 'instance' or 'function'; `index` picks the user function (it wraps around, so that repeating the shortcut cycles).
 */
export function definitionPlace(definitions, which, index = 0) {
    if (!definitions) return null;
    if (which === 'instance') return definitions.instance ?? null;
    const fns = definitions.user_functions ?? definitions.functions ?? [];
    return fns.length ? fns[((index % fns.length) + fns.length) % fns.length] : null;
}

/** The sites that cover a line of a file (a basename is matched, as the panes do). */
export function sitesAt(sites, file, line) {
    const base = f => String(f).split('\\').pop().split('/').pop();
    const want = base(file);
    return (sites ?? []).filter(s => base(s.file) === want && line >= s.line && line <= s.end_line);
}
