// badges.js — which gutter badges the source panes get from the nodes of a graph. Pure, no DOM.
'use strict';

const SOURCE_TYPES = ['skeleton_call', 'external', 'region'];

/**
 * @param {Array} badgeNodes        [{ data: { id, type, file, line, total_order, total_orders, locations,
 *                                     trace_index, trace_indices, hidden } }] from the graph response
 * @param {Array} traceLineRanges   [[first line, last line]] of each event in the trace file
 * @returns {{ badges, traceBadges, orders }}
 *   badges       'file:line' -> { count, hiddenCount, entries: [{ id, iterIdx }], file, line }
 *   traceBadges  trace index -> { count, hiddenCount, entries, range }
 *   orders       node id -> total order of each of its iterations
 *
 * A node that stands for several iterations (iterations coalesced) has an entry for each iteration, at the
 * place where that iteration ran: node.data.locations has [file, line] for each iteration if they are not
 * all at the node's own place (the nodes of a key path can be called from different lines).
 */
export function collectBadges(badgeNodes, traceLineRanges) {
    const badges = {}, traceBadges = {}, orders = {};

    for (const { data } of badgeNodes) {
        const list = data.total_orders || [data.total_order];
        orders[data.id] = list;
        const iters = data.total_orders ? data.total_orders.length : 1;
        const hidden = !!data.hidden;

        // ── Source-file badges: one entry per iteration, at the line of that iteration ─────────────
        if (SOURCE_TYPES.includes(data.type)) {
            for (let i = 0; i < iters; i++) {
                const [file, line] = (data.locations && data.locations[i]) || [data.file, data.line];
                if (!file || line === -1 || line == null) continue;
                const key = file + ':' + line;
                const b = badges[key] ??= { count: 0, hiddenCount: 0, entries: [], file, line };
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
    return { badges, traceBadges, orders };
}
