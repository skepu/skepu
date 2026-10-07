// tools.js — the tools of the SkeVU analysis server: name, description, input schema, handler.
// A handler gets the session (the loaded traces) and the arguments, and returns plain JSON.
'use strict';

import { readFileSync } from 'node:fs';
import { basename, resolve, sep } from 'node:path';
import { parseTrace } from '../script/backend/trace.js';
import * as A from '../script/backend/analysis.js';
import { criticalPath } from '../script/backend/graph_analysis.js';
import { fusionCandidates } from '../script/backend/fusion.js';

export class Session {
    constructor({ roots = null } = {}) {
        this.roots  = roots;            // directories that may be read, or null for any
        this.traces = new Map();        // handle -> { path, header, events, sources }
        this.seq    = 0;
    }

    read(path) {
        const p = resolve(path);
        if (this.roots && !this.roots.some(r => p === r || p.startsWith(r + sep)))
            throw new Error(`${path} is outside the allowed directories`);
        return readFileSync(p, 'utf8');
    }

    trace(handle) {
        const t = this.traces.get(handle);
        if (!t) throw new Error(`unknown trace handle '${handle}'; call load_trace first`);
        return t;
    }
}

const handle = { type: 'string', description: 'Handle returned by load_trace.' };
const limit  = { type: 'integer', minimum: 1, maximum: 500, description: 'Maximum number of rows (default 20).' };

export const TOOLS = [
    {
        name: 'load_trace',
        description: 'Load a SkePU trace file (JSON written by a program built with SKEPU_TRACING) and optionally its source files. Returns a handle for the other tools and a short summary. Times are microseconds.',
        inputSchema: {
            type: 'object', required: ['path'],
            properties: {
                path: { type: 'string', description: 'Path of the trace file.' },
                sources: { type: 'array', items: { type: 'string' }, description: 'Paths of source files (.cpp/.hpp) of the program, for source_context.' },
            },
        },
        run(session, { path, sources = [] }) {
            const { header, events } = parseTrace(session.read(path));
            const files = {};
            for (const s of sources) files[basename(s)] = session.read(s);
            const h = 't' + (++session.seq);
            session.traces.set(h, { path, header, events, sources: files });
            return { handle: h, ...A.summarize(events, header), sources: Object.keys(files) };
        },
    },
    {
        name: 'summarize',
        description: 'Overview of a loaded trace: events by type, time span, backends, regions, skeleton call time and transfer time by kind.',
        inputSchema: { type: 'object', required: ['handle'], properties: { handle } },
        run(session, { handle }) { const t = session.trace(handle); return A.summarize(t.events, t.header); },
    },
    {
        name: 'time_by',
        description: 'Where the time goes: events of a type (default skeleton calls) grouped by pattern, label, backend, region, file or file_line, with count, total, mean, max and share. Sorted by total time.',
        inputSchema: {
            type: 'object', required: ['handle'],
            properties: {
                handle,
                group_by: { type: 'string', enum: ['pattern', 'label', 'backend', 'region', 'file', 'file_line'] },
                type: { type: 'string', enum: ['skeleton_call', 'transfer', 'external'], description: 'Event type (default skeleton_call).' },
                top: limit,
            },
        },
        run(session, { handle, group_by = 'pattern', type = 'skeleton_call', top = 20 }) {
            return A.timeBy(session.trace(handle).events, group_by, { type, top });
        },
    },
    {
        name: 'transfers',
        description: 'Data transfers between host and device by kind (container, or the library\'s internal ones: reduce-temp = partial reduction results brought to the host, overlap-wrap = MapOverlap edge data, prng), the internal ones attributed to the skeleton call that caused them, and the containers that move most or change direction repeatedly.',
        inputSchema: {
            type: 'object', required: ['handle'],
            properties: { handle, kind: { type: 'string', description: 'Only this kind (container, reduce-temp, overlap-wrap, prng, ...) in the longest list.' }, limit },
        },
        run(session, { handle, kind = null, limit = 20 }) { return A.transfers(session.trace(handle).events, { kind, limit }); },
    },
    {
        name: 'find_events',
        description: 'Events matching a filter (type, label substring, file suffix, line, region, backend, pattern, minimum duration), paged. Sort by trace order or by duration.',
        inputSchema: {
            type: 'object', required: ['handle'],
            properties: {
                handle, type: { type: 'string' }, label: { type: 'string' }, file: { type: 'string' }, line: { type: 'integer' },
                region: { type: 'string' }, backend: { type: 'string' }, pattern: { type: 'string' },
                min_duration_us: { type: 'number' }, sort_by: { type: 'string', enum: ['trace', 'duration'] },
                limit: { type: 'integer', minimum: 1, maximum: 500 }, offset: { type: 'integer', minimum: 0 },
            },
        },
        run(session, { handle, ...q }) { return A.findEvents(session.trace(handle).events, q); },
    },
    {
        name: 'containers',
        description: 'Data containers: label, allocations, container transfers (count, time) and how many skeleton calls read and write each. Sorted by transfer time.',
        inputSchema: { type: 'object', required: ['handle'], properties: { handle, top: limit } },
        run(session, { handle, top = 20 }) { return A.containers(session.trace(handle).events, { top }); },
    },
    {
        name: 'critical_path',
        description: 'The longest path in time through the dependences between skeleton calls and data: its length, the work (sum of skeleton call times), the parallelism (work / path), and the timed nodes on the path with their share. The library\'s internal transfers are not part of the dependence graph.',
        inputSchema: { type: 'object', required: ['handle'], properties: { handle } },
        run(session, { handle }) { return criticalPath(session.trace(handle).events); },
    },
    {
        name: 'fusion_candidates',
        description: 'Skeleton calls that could be fused into one call, and why others could not. Serial fusion: a call and the only consumer of its output (read element by element, same backend and elements, no host access in between), as chains or trees; the intermediate containers need not be written and read in memory. Parallel fusion: independent calls over the same elements that could share one launch. Both are checked for conflicting events between the calls. Equal groups (the iterations of a loop) are one entry with a count. "blocked" says why pairs were not fusable: multiple-readers, host-access, proxy-access, pattern, backend-mismatch, size-mismatch, shared-prng, hazard, recompute, scalar-output. A Map fused into a MapOverlap, MapPool, MapPairs or MapPairsReduce recomputes its elements: the group has a "recompute" entry with the factor when it is known. Which patterns can be fused is an assumption, returned as "rules" and settable.',
        inputSchema: {
            type: 'object', required: ['handle'],
            properties: {
                handle,
                blocked_limit: { type: 'integer', minimum: 0, maximum: 200, description: 'Number of blocked pairs to list (default 30).' },
                include_overlap: { type: 'boolean', description: 'Let a call be in a serial and in a parallel group (default false).' },
                aggregate: { type: 'boolean', description: 'One entry for equal groups (default true).' },
                rules: {
                    type: 'object', description: 'Pattern names (Map, Reduce, MapReduce, MapOverlap, Scan, ...) that may be fused.',
                    properties: {
                        serialProducers: { type: 'array', items: { type: 'string' } }, serialConsumers: { type: 'array', items: { type: 'string' } },
                        serialConsumersRecompute: { type: 'array', items: { type: 'string' }, description: 'Consumers a Map can be fused into at the price of computing it more than once per element.' },
                        allowRecompute: { type: 'boolean', description: 'Report fusions that recompute (default true).' },
                        parallel: { type: 'array', items: { type: 'string' } },
                    },
                },
            },
        },
        run(session, { handle, ...options }) { return fusionCandidates(session.trace(handle).events, options); },
    },
    {
        name: 'source_context',
        description: 'The events attributed to a source line, and its source text (a few lines around it) if the file was given to load_trace.',
        inputSchema: {
            type: 'object', required: ['handle', 'file', 'line'],
            properties: { handle, file: { type: 'string', description: 'File name or path suffix.' }, line: { type: 'integer' }, context: { type: 'integer', description: 'Lines of source around the line (default 3).' } },
        },
        run(session, { handle, file, line, context = 3 }) {
            const t = session.trace(handle);
            const r = A.sourceLine(t.events, file, line);
            const text = t.sources[basename(file)];
            if (text) {
                const lines = text.split('\n');
                const from = Math.max(1, line - context), to = Math.min(lines.length, line + context);
                r.source = lines.slice(from - 1, to).map((s, i) => ({ line: from + i, text: s }));
            }
            return r;
        },
    },
];
