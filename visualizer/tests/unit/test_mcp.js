// test_mcp.js — the analysis server (mcp/server.js) over its JSON-RPC stdio protocol.
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT  = join(dirname(fileURLToPath(import.meta.url)), '../..');
const TRACE = join(ROOT, 'examples/dnn_cifar10/trace_dnn_cifar10_opencl.json');

function startServer(env = {}) {
    const proc = spawn(process.execPath, [join(ROOT, 'mcp/server.js')], { env: { ...process.env, ...env }, stdio: ['pipe', 'pipe', 'inherit'] });
    const pending = new Map();
    let seq = 0;
    createInterface({ input: proc.stdout }).on('line', line => {
        const m = JSON.parse(line);
        pending.get(m.id)?.(m);
        pending.delete(m.id);
    });
    return {
        request(method, params) {
            const id = ++seq;
            return new Promise(res => { pending.set(id, res); proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n'); });
        },
        notify(method, params) { proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n'); },
        async tool(name, args) { return (await this.request('tools/call', { name, arguments: args })).result; },
        close() { proc.kill(); },
    };
}

describe('the MCP server', () => {
    let s;
    beforeAll(() => { s = startServer(); });
    afterAll(() => s.close());

    it('initializes and lists its tools', async () => {
        const init = await s.request('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'test', version: '0' } });
        expect(init.result).toMatchObject({ protocolVersion: '2025-06-18', capabilities: { tools: {} }, serverInfo: { name: 'skevu-mcp' } });
        s.notify('notifications/initialized');
        const { result } = await s.request('tools/list');
        const names = result.tools.map(t => t.name);
        expect(names).toEqual(expect.arrayContaining(['load_trace', 'summarize', 'time_by', 'transfers', 'find_events', 'containers', 'critical_path', 'fusion_candidates', 'source_context']));
        for (const t of result.tools) expect(t.inputSchema.type).toBe('object');
    });

    it('loads a trace and answers questions about it', async () => {
        const loaded = await s.tool('load_trace', { path: TRACE });
        expect(loaded.isError).toBeUndefined();
        const { handle, events } = loaded.structuredContent;
        expect(events).toBe(812);

        const tr = (await s.tool('transfers', { handle, limit: 3 })).structuredContent;
        expect(tr.by_kind['reduce-temp'].count).toBe(24);
        expect(tr.longest).toHaveLength(3);

        const text = (await s.tool('time_by', { handle, group_by: 'pattern' })).content[0].text;
        expect(JSON.parse(text).rows[0].key).toBe('Map');

        const cp = (await s.tool('critical_path', { handle })).structuredContent;
        expect(cp.work_us).toBe(245525);

        const fu = (await s.tool('fusion_candidates', { handle })).structuredContent;
        expect(fu.serial_groups).toBeGreaterThan(50);
        expect(fu.groups[0]).toHaveProperty('occurrences');
    });

    it('reports a failing tool as a tool error, not a protocol error', async () => {
        const r = await s.tool('summarize', { handle: 'nope' });
        expect(r.isError).toBe(true);
        expect(r.content[0].text).toMatch(/unknown trace handle/);
        const bad = await s.tool('time_by', { handle: (await s.tool('load_trace', { path: TRACE })).structuredContent.handle, group_by: 'colour' });
        expect(bad.isError).toBe(true);
    });

    it('answers unknown methods and tools with protocol errors', async () => {
        expect((await s.request('resources/list')).error.code).toBe(-32601);
        expect((await s.request('tools/call', { name: 'nope', arguments: {} })).error.code).toBe(-32602);
    });
});

describe('the allowed directories', () => {
    it('keep the server to them', async () => {
        const s = startServer({ SKEVU_MCP_ROOTS: join(ROOT, 'tests') });
        try {
            const outside = await s.tool('load_trace', { path: TRACE });
            expect(outside.isError).toBe(true);
            expect(outside.content[0].text).toMatch(/outside the allowed/);
            const inside = await s.tool('load_trace', { path: join(ROOT, 'tests/fixtures/map_chain.json') });
            expect(inside.structuredContent.events).toBe(7);
        } finally { s.close(); }
    });
});
