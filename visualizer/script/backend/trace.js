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

// ── Utility: trace line-range computation ─────────────────────────────────────
// Mirrors compute_trace_line_ranges() in visualizer.py.
// Maps each top-level JSON array element to the (1-based inclusive) line span
// it occupies in the raw source text.

function bisectRight(arr, val) {
    let lo = 0, hi = arr.length;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (arr[mid] <= val) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/**
 * Return the index one past the end of the JSON value starting at text[pos].
 * Handles objects, arrays, strings, numbers, booleans, and null.
 * Mirrors Python's JSONDecoder.raw_decode().
 */
function jsonValueEnd(text, pos) {
    const ch = text[pos];
    if (ch === '"') {
        let i = pos + 1;
        while (i < text.length) {
            if (text[i] === '\\') { i += 2; continue; }
            if (text[i] === '"')  return i + 1;
            i++;
        }
        throw new SyntaxError('Unterminated JSON string at offset ' + pos);
    }
    if (ch === '{' || ch === '[') {
        const close = ch === '{' ? '}' : ']';
        let depth = 1, i = pos + 1;
        while (i < text.length) {
            const c = text[i];
            if (c === '"') {
                i++;
                while (i < text.length) {
                    if (text[i] === '\\') { i += 2; continue; }
                    if (text[i] === '"')  { i++; break; }
                    i++;
                }
                continue;
            }
            if (c === '{' || c === '[') depth++;
            else if ((c === '}' || c === ']') && --depth === 0) return i + 1;
            i++;
        }
        throw new SyntaxError('Unterminated JSON object/array at offset ' + pos);
    }
    // Number, boolean, null — scan to next delimiter.
    let i = pos;
    while (i < text.length && !/[\s,\]}\r\n]/.test(text[i])) i++;
    return i;
}

export function computeTraceLineRanges(jsonText) {
    // Build sorted array of character offsets where each line begins.
    const lineStarts = [0];
    for (let i = 0; i < jsonText.length; i++)
        if (jsonText[i] === '\n') lineStarts.push(i + 1);

    const charToLine = pos => bisectRight(lineStarts, pos);

    const ranges = [];
    let pos = jsonText.indexOf('[') + 1;
    while (pos < jsonText.length) {
        while (pos < jsonText.length && ' \t\n\r,'.includes(jsonText[pos])) pos++;
        if (pos >= jsonText.length || jsonText[pos] === ']') break;
        const start  = pos;
        const end    = jsonValueEnd(jsonText, pos);
        ranges.push([charToLine(start), charToLine(end - 1)]);
        pos = end;
    }
    return ranges;
}

// ── Utility: file path preprocessing ─────────────────────────────────────────
// Mirrors preprocess_file_paths() in visualizer.py.
// Strips the longest common directory prefix from all `file` fields so paths
// shown in the UI are relative to the source tree root.

export function preprocessFilePaths(events) {
    const paths = events.map(e => e.file).filter(f => f && f !== '');
    if (!paths.length) return;

    // Character-by-character common prefix (matches Python os.path.commonprefix).
    let prefix = paths[0];
    for (let i = 1; i < paths.length && prefix; i++) {
        const p = paths[i];
        let j = 0;
        while (j < prefix.length && j < p.length && prefix[j] === p[j]) j++;
        prefix = prefix.slice(0, j);
    }

    // Trim to directory boundary (mirrors os.path.dirname).
    const slash = Math.max(prefix.lastIndexOf('/'), prefix.lastIndexOf('\\'));
    const dir   = slash >= 0 ? prefix.slice(0, slash) : '';
    if (!dir) return;

    for (const ev of events) {
        if (ev.file && ev.file !== '' && ev.file.startsWith(dir))
            ev.file = ev.file.slice(dir.length);
    }
}
