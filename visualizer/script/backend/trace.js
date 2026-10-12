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

// The sections of a trace header that are not source files. The rest of the header maps the path of a
// source file to the hash of its contents (what the tracer writes, so that SkeVU can tell whether the source it
// has is the source that was run).
export const RESERVED_HEADER_KEYS = ['mpi', 'clock', 'instances'];

/** The source files of a header, as { path: hash }. */
export function headerSourceFiles(header) {
    const files = {};
    for (const [k, v] of Object.entries(header ?? {}))
        if (!RESERVED_HEADER_KEYS.includes(k) && typeof v === 'string') files[k] = v;
    return files;
}

/**
 * The table of the skeleton instances in a header: where each instance and its user functions are defined in the source.
 * Returns a Map from the object id of the instance (what the calls of the instance have) to
 *   { label, instance: { file, line, end_line, anonymous? }, user_functions: [{ name, kind, file, line, end_line }] }
 * (kind is "function" or "lambda", and a lambda has no name).  An entry that does not have the lines is left out: the
 * table is what the tracer wrote, but a header can come from anywhere.
 */
export function headerInstances(header) {
    const result = new Map();
    const table = header?.instances;
    if (!table || typeof table !== 'object') return result;
    const place = p => p && typeof p.file === 'string' && Number.isInteger(p.line);
    for (const [id, e] of Object.entries(table)) {
        if (!e || !place(e.instance) || !Array.isArray(e.user_functions)) continue;
        const instance = { ...e.instance, end_line: e.instance.end_line ?? e.instance.line };
        const user_functions = e.user_functions.filter(place).map(f => ({ name: '', kind: 'function', ...f, end_line: f.end_line ?? f.line }));
        result.set(Number(id), { label: e.label ?? '', instance, user_functions });
    }
    return result;
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

/**
 * The position of the '[' that starts the events: the start of the text for a trace that is an array, the
 * value of "data" for one that is { header, data } (wherever the key is: the header can have arrays).
 */
export function eventsArrayStart(text) {
    let pos = 0;
    const skip = () => { while (pos < text.length && ' \t\n\r'.includes(text[pos])) pos++; };
    skip();
    if (text[pos] === '[') return pos;
    if (text[pos] !== '{') throw new Error('Not a SkePU trace: expected an array of events or an object { header, data }');
    pos++;
    while (pos < text.length) {
        while (pos < text.length && ' \t\n\r,'.includes(text[pos])) pos++;
        if (text[pos] === '}') break;
        const keyEnd = jsonValueEnd(text, pos);
        const key = JSON.parse(text.slice(pos, keyEnd));
        pos = keyEnd;
        skip();
        if (text[pos] !== ':') throw new SyntaxError('Expected ":" at offset ' + pos);
        pos++;
        skip();
        if (key === 'data') return pos;
        pos = jsonValueEnd(text, pos);
    }
    throw new Error('Not a SkePU trace: the object has no "data" array');
}

export function computeTraceLineRanges(jsonText) {
    // Build sorted array of character offsets where each line begins.
    const lineStarts = [0];
    for (let i = 0; i < jsonText.length; i++)
        if (jsonText[i] === '\n') lineStarts.push(i + 1);

    const charToLine = pos => bisectRight(lineStarts, pos);

    const ranges = [];
    let pos = eventsArrayStart(jsonText) + 1;
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
