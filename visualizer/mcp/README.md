# SkeVU analysis server (prototype)

A headless SkeVU as a [Model Context Protocol](https://modelcontextprotocol.io) server: tools that
load SkePU trace files and answer questions about them as small JSON documents, for an AI assistant
or any script. It uses the analyses in `script/backend/` (`trace.js`, `analysis.js`,
`graph_analysis.js`) and has no dependencies besides Node (19 or later; 18 works with the built-in
shim).

```bash
node mcp/server.js                                        # speaks MCP on stdin/stdout
SKEVU_MCP_ROOTS=/data/traces:/src/app node mcp/server.js  # reads files only below these directories
claude mcp add skevu -- node /path/to/visualizer/mcp/server.js   # for Claude Code
```

## Tools

All times are microseconds. Load a trace first and use the returned handle.

| Tool | Answers |
|---|---|
| `load_trace(path, sources?)` | Loads a trace (and source files) and returns a handle and an overview |
| `summarize(handle)` | Events by type, time span, backends, regions, skeleton and transfer time |
| `time_by(handle, group_by, type?, top?)` | Where the time goes, by pattern, label, backend, region, file or file_line |
| `transfers(handle, kind?, limit?)` | Transfers by kind, the library's internal ones attributed to the skeleton call that causes them, containers that move a lot or ping-pong |
| `find_events(handle, filters...)` | Events by type, label, file, line, region, backend, pattern, duration; paged |
| `containers(handle, top?)` | Allocations, transfers and the number of reads and writes of each container |
| `critical_path(handle)` | The longest path in time through the dependences, the work and the parallelism |
| `source_context(handle, file, line)` | The events of a source line and the source text around it |

Events have stable ids `e<n>` (their position in the trace), unlike the node ids of the graph in the UI.

### Internal transfers

The trace also holds the library's own transfers, which the graph view leaves out and which can be
most of the transfer time: `Internal (reduce-temp)` (partial results of a reduction brought to the
host), `Internal (overlap-wrap)` (edge data of a MapOverlap) and `[Internal PRNG]`. `transfers` reports
them by kind; this is read from the labels, since the tracer records no kind.

## Status

A prototype: the analyses work on the events and, for the critical path, on a dependence graph built
for the purpose; the UI's worker (`script/backend/visualizer.js`) is unchanged and does not use this
code yet. Open: tracer fields for the kind of a transfer and the skeleton call it belongs to, the
`{header, data}` trace format (read, but the source hashes are not checked), fusion candidates,
and sharing the pipeline with the worker.

`npm test` runs the tests (`tests/unit/test_analysis.js`, `test_mcp.js`).
