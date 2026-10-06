// trace.js — reading a SkePU trace file.  Pure; usable in the worker, in Node and in tests.
'use strict';

/**
 * Parse the text of a trace file.  A trace is an array of events, or an object
 * { header, data } (the header has metadata such as the hashes of the source files).
 *
 * Returns { header, events }.  The events are shallow copies in file order with an added
 * `idx` (their position in the trace), which is the stable identifier of an event: 'e<idx>'.
 * File paths are kept as they are in the file.
 */
export function parseTrace(text) {
    const raw = JSON.parse(text);
    let header = null, data;
    if (Array.isArray(raw)) {
        data = raw;
    } else if (raw && Array.isArray(raw.data)) {
        header = raw.header ?? null;
        data   = raw.data;
    } else {
        throw new Error('Not a SkePU trace: expected an array of events or an object { header, data }');
    }
    return { header, events: data.map((e, i) => ({ ...e, idx: i })) };
}
