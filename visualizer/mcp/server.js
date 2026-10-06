#!/usr/bin/env node
// SkeVU analysis server: the Model Context Protocol (JSON-RPC 2.0, one message per line on
// stdin/stdout) over the headless analyses of the visualizer backend. No dependencies.
//
//   node mcp/server.js                         any readable file
//   SKEVU_MCP_ROOTS=/data/traces:/src node mcp/server.js    only below these directories
'use strict';

import { createInterface } from 'node:readline';
import { webcrypto } from 'node:crypto';
import { resolve } from 'node:path';
import { TOOLS, Session } from './tools.js';

if (!globalThis.crypto?.randomUUID) globalThis.crypto = webcrypto;   // Node < 19

const PROTOCOL_VERSIONS = ['2025-06-18', '2025-03-26', '2024-11-05'];
const roots = process.env.SKEVU_MCP_ROOTS ? process.env.SKEVU_MCP_ROOTS.split(':').filter(Boolean).map(r => resolve(r)) : null;
const session = new Session({ roots });

const send = msg => process.stdout.write(JSON.stringify({ jsonrpc: '2.0', ...msg }) + '\n');
const reply = (id, result) => send({ id, result });
const fail  = (id, code, message) => send({ id, error: { code, message } });

function callTool(params) {
    const tool = TOOLS.find(t => t.name === params?.name);
    if (!tool) return { error: { code: -32602, message: `Unknown tool: ${params?.name}` } };
    try {
        const out = tool.run(session, params.arguments ?? {});
        return { result: { content: [{ type: 'text', text: JSON.stringify(out) }], structuredContent: out } };
    } catch (e) {
        // A failure of the tool is a result the model can read, not a protocol error.
        return { result: { content: [{ type: 'text', text: String(e.message ?? e) }], isError: true } };
    }
}

function handle(msg) {
    const { id, method, params } = msg;
    switch (method) {
        case 'initialize': {
            const wanted = params?.protocolVersion;
            return reply(id, {
                protocolVersion: PROTOCOL_VERSIONS.includes(wanted) ? wanted : PROTOCOL_VERSIONS[0],
                capabilities: { tools: {} },
                serverInfo: { name: 'skevu-mcp', version: '0.1.0' },
                instructions: 'Analyses of SkePU execution traces. Call load_trace first; times are microseconds.',
            });
        }
        case 'ping':       return reply(id, {});
        case 'tools/list': return reply(id, { tools: TOOLS.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })) });
        case 'tools/call': { const r = callTool(params); return r.error ? fail(id, r.error.code, r.error.message) : reply(id, r.result); }
        default:
            if (id === undefined) return;                       // a notification (e.g. notifications/initialized)
            return fail(id, -32601, `Method not found: ${method}`);
    }
}

const rl = createInterface({ input: process.stdin });
rl.on('line', line => {
    if (!line.trim()) return;
    let msg;
    try { msg = JSON.parse(line); } catch { return fail(null, -32700, 'Parse error'); }
    try { handle(msg); } catch (e) { fail(msg.id ?? null, -32603, String(e.message ?? e)); }
});
