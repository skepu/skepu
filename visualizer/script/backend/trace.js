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

/**
 * The source files of a header, as { path: hash }: the hash of the contents of each source file that the events name
 * (what the tracer writes in `files`, so that SkeVU can tell whether the source it has is the source that was run).
 */
export function headerSourceFiles(header) {
    const files = {};
    if (header?.files && typeof header.files === 'object')
        for (const [path, hash] of Object.entries(header.files))
            if (typeof hash === 'string') files[path] = hash;
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

// ── The trace as the code view shows it ──────────────────────────────────────────────────────────────────────────
// A trace can be written compactly (the tracer writes one event on each line, or a whole trace on one line), which is not a text to read in
// the code view. The view shows a formatted copy of the text instead, and the line ranges of the events in that copy. The formatting changes
// white space only: every string and number is copied as it is written, so the values are the ones of the file.

/** Text with lines that are longer than this on average is not read as it is: it is formatted for the view. */
export const READABLE_AVERAGE_LINE = 120;
/** A text longer than this is shown as it is (a copy of it, formatted, would be much larger and take a long time). */
export const MAX_FORMATTED_CHARS = 64 * 1024 * 1024;
/** An array of values (no objects or arrays in it) that is shorter than this is kept on one line. */
const INLINE_ARRAY_CHARS = 100;

function lineCount(text) {
    let lines = 1;
    for (let at = text.indexOf('\n'); at >= 0; at = text.indexOf('\n', at + 1)) lines++;
    return lines;
}

const isSpace = code => code === 32 || code === 10 || code === 13 || code === 9;

/** The index one past the string that starts at text[at] (a quote). */
function stringEnd(text, at) {
    for (let i = at + 1; i < text.length; i++) {
        const code = text.charCodeAt(i);
        if (code === 92) i++;                 // (a backslash: the next character is part of the string)
        else if (code === 34) return i + 1;
    }
    throw new SyntaxError('Unterminated JSON string at offset ' + at);
}

/** If the array that starts at text[at] has only values in it (no objects or arrays) and is short, the index of its ']'; else -1. */
function inlineArrayEnd(text, at) {
    const limit = Math.min(text.length, at + INLINE_ARRAY_CHARS);
    for (let i = at + 1; i < limit; i++) {
        const code = text.charCodeAt(i);
        if (code === 34) { i = stringEnd(text, i) - 1; if (i >= limit) return -1; }
        else if (code === 123 || code === 91) return -1;     // { [
        else if (code === 93) return i;                       // ]
    }
    return -1;
}

/** The array of values text[at .. end] on one line: ", " between the values, the values as they are written. */
function inlineArray(text, at, end) {
    const out = [];
    let i = at + 1;
    while (i < end) {
        const code = text.charCodeAt(i);
        if (isSpace(code) || code === 44) { i++; continue; }
        let stop;
        if (code === 34) stop = stringEnd(text, i);
        else { stop = i; while (stop < end && !isSpace(text.charCodeAt(stop)) && text.charCodeAt(stop) !== 44) stop++; }
        out.push(text.slice(i, stop));
        i = stop;
    }
    return '[' + out.join(', ') + ']';
}

/** JSON text with two spaces of indentation, an object member or an array element on each line, and short arrays of values on one line. */
export function prettyPrintJson(text) {
    const out = [], indents = ['\n'];
    const indent = depth => indents[depth] ??= '\n' + '  '.repeat(depth);
    let depth = 0, i = 0;
    const n = text.length;
    while (i < n) {
        const code = text.charCodeAt(i);
        if (isSpace(code)) { i++; continue; }
        if (code === 34) { const end = stringEnd(text, i); out.push(text.slice(i, end)); i = end; }
        else if (code === 123 || code === 91) {                        // { [
            let next = i + 1;
            while (next < n && isSpace(text.charCodeAt(next))) next++;
            const close = code === 123 ? 125 : 93;
            if (text.charCodeAt(next) === close) { out.push(code === 123 ? '{}' : '[]'); i = next + 1; continue; }
            if (code === 91) {
                const end = inlineArrayEnd(text, i);
                if (end >= 0) { out.push(inlineArray(text, i, end)); i = end + 1; continue; }
            }
            out.push(text[i]); depth++; out.push(indent(depth)); i++;
        }
        else if (code === 125 || code === 93) { if (--depth < 0) throw new SyntaxError('Unbalanced JSON at offset ' + i); out.push(indent(depth), text[i]); i++; }
        else if (code === 44) { out.push(',', indent(depth)); i++; }
        else if (code === 58) { out.push(': '); i++; }
        else { let stop = i; while (stop < n && !isSpace(text.charCodeAt(stop)) && ',:]}'.indexOf(text[stop]) < 0) stop++; out.push(text.slice(i, stop)); i = stop; }
    }
    if (depth !== 0) throw new SyntaxError('Unbalanced JSON at the end');
    return out.join('');
}

/**
 * The text of a trace for the code view and the line ranges of its events in that text: the text as it is if it is readable, formatted
 * otherwise (a text that cannot be formatted, or is very large, is shown as it is).
 */
export function formatTraceForDisplay(text) {
    let shown = text;
    if (text.length <= MAX_FORMATTED_CHARS && text.length / lineCount(text) > READABLE_AVERAGE_LINE) {
        try { shown = prettyPrintJson(text); } catch (e) { shown = text; }
    }
    return { text: shown, ranges: computeTraceLineRanges(shown) };
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
