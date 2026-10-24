// test_trace_format.js — the trace as the code view shows it (script/backend/trace.js: formatTraceForDisplay): compact traces are formatted for reading,
// with white space changes only, and the line ranges are those of the text that is shown.
import { describe, it, expect } from 'vitest';
import { prettyPrintJson, formatTraceForDisplay, computeTraceLineRanges, READABLE_AVERAGE_LINE } from '../../script/backend/trace.js';

const events = [
    { object_id: 5, type: 'allocation', label: 'v, with "quotes", [brackets] and {braces}: x', time: 12, line: 3 },
    { object_id: 1000, type: 'skeleton_call', pattern: 'Map', label: 'f', start: 1, end: 9, elements: [2048, 1], file: '/a/b.cpp', line: 21,
      elwise_inputs: [5, 6], proxy_inputs: [], scalar_inputs: [], outputs: [7], parameters: { strides: [1, 1], mode: 'None' }, big: 12345678901234567890, f: 1.50, e: 1e-7 },
    { object_id: 5, type: 'deallocation', label: 'v\n\\', time: 40, line: 9 },
];
const header = { files: { '/a/b.cpp': 'abc123' }, instances: { 1000: { label: 'f', instance: { file: '/a/b.cpp', line: 20, end_line: 21 } } } };

// the text without its white space outside of strings
function stripped(text) {
    let out = '', inString = false;
    for (let i = 0; i < text.length; i++) {
        const c = text[i];
        if (inString) { out += c; if (c === '\\') out += text[++i]; else if (c === '"') inString = false; }
        else if (c === '"') { inString = true; out += c; }
        else if (!/\s/.test(c)) out += c;
    }
    return out;
}

describe('the formatter', () => {
    it('keeps every value as it is written (white space only changes)', () => {
        const compact = '{"header":{"n":12345678901234567890,"x":1.50,"e":1e-7,"s":"a\\"b, c: [d]"},"data":[{"a":[],"b":{},"c":[1,2,3]}]}';
        const pretty = prettyPrintJson(compact);
        expect(stripped(pretty)).toBe(stripped(compact));   // (white space outside of strings is the only difference)
        expect(JSON.parse(pretty)).toEqual(JSON.parse(compact));
        expect(pretty).toContain('12345678901234567890');   // (a number that a JSON.parse and stringify would change)
        expect(pretty).toContain('1.50');
        expect(pretty).toContain('1e-7');
    });

    it('puts an object member or an array element on each line, and short arrays of values on one', () => {
        const pretty = prettyPrintJson('[{"elements":[2048,1],"parameters":{"strides":[1,1]},"empty":[],"none":{}}]');
        expect(pretty).toBe('[\n  {\n    "elements": [2048, 1],\n    "parameters": {\n      "strides": [1, 1]\n    },\n    "empty": [],\n    "none": {}\n  }\n]');
    });

    it('does not take a bracket, a brace, a comma or a colon in a string for structure', () => {
        const text = JSON.stringify([{ s: 'a, b: [c] {d} "e" \\' }]);
        expect(JSON.parse(prettyPrintJson(text))).toEqual(JSON.parse(text));
    });

    it('does not accept text that is not balanced (the view then shows it as it is)', () => {
        expect(() => prettyPrintJson('[{"a":1}')).toThrow();
        expect(() => prettyPrintJson('[1]]')).toThrow();
        expect(() => prettyPrintJson('["unterminated')).toThrow();
    });
});

describe('the trace for the code view', () => {
    const oneLine = JSON.stringify({ header, data: events });
    const perLine = '{"header":' + JSON.stringify(header) + ',"data":[\n' + events.map(e => JSON.stringify(e)).join(',\n') + '\n]}';
    const indented = JSON.stringify({ header, data: events }, null, 4);

    it('is a trace that is readable as it is: the text is not changed', () => {
        const shown = formatTraceForDisplay(indented);
        expect(shown.text).toBe(indented);
        expect(shown.ranges).toEqual(computeTraceLineRanges(indented));
    });

    it('is formatted when it is on one line, and has a range for each event, each on lines of its own', () => {
        const shown = formatTraceForDisplay(oneLine);
        expect(shown.text).not.toBe(oneLine);
        expect(JSON.parse(shown.text)).toEqual(JSON.parse(oneLine));
        expect(shown.ranges).toHaveLength(events.length);
        const lines = shown.text.split('\n');
        shown.ranges.forEach(([first, last], i) => {
            expect(JSON.parse(lines.slice(first - 1, last).join('\n').replace(/,\s*$/, ''))).toEqual(events[i]);
        });
        for (let i = 1; i < shown.ranges.length; i++) expect(shown.ranges[i][0]).toBeGreaterThan(shown.ranges[i - 1][1]);
    });

    it('is formatted when the events are one on each line (a line of an event is long), and the ranges follow the formatted text', () => {
        expect(perLine.length / perLine.split('\n').length).toBeGreaterThan(READABLE_AVERAGE_LINE);
        const raw = computeTraceLineRanges(perLine);
        expect(raw.map(([a, b]) => b - a)).toEqual([0, 0, 0]);   // (in the file: a line for an event)
        const shown = formatTraceForDisplay(perLine);
        expect(shown.ranges[1][1] - shown.ranges[1][0]).toBeGreaterThan(5);   // (in the view: several lines)
        expect(JSON.parse(shown.text)).toEqual(JSON.parse(perLine));
    });

    it('is formatted once: the formatted text is readable, and is not changed by another pass', () => {
        const once = formatTraceForDisplay(oneLine);
        const twice = formatTraceForDisplay(once.text);
        expect(twice.text).toBe(once.text);
        expect(twice.ranges).toEqual(once.ranges);
    });

    it('is a bare array of events, as a trace can be', () => {
        const shown = formatTraceForDisplay(JSON.stringify(events));
        expect(shown.ranges).toHaveLength(events.length);
    });

    it('is shown as it is when it cannot be formatted (the ranges of a text that is not a trace are an error of the loader, not of the formatter)', () => {
        const truncated = oneLine.slice(0, oneLine.length - 30);
        expect(() => formatTraceForDisplay(truncated)).toThrow();   // (computeTraceLineRanges reports it, as it does for the raw text)
    });

    it('takes a large trace in a short time', () => {
        const many = Array.from({ length: 30000 }, (_, i) => ({ ...events[1], object_id: i, start: i, end: i + 5 }));
        const text = JSON.stringify({ header, data: many });
        const started = performance.now();
        const shown = formatTraceForDisplay(text);
        const took = performance.now() - started;
        expect(shown.ranges).toHaveLength(30000);
        expect(took).toBeLessThan(3000);
    });
});
