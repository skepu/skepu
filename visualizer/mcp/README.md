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
| `fusion_candidates(handle, ...)` | Calls that could be fused serially (producer and consumer) or in parallel (independent siblings), the intermediates avoided, and why other pairs cannot be fused; rules and audit in [FUSION.md](FUSION.md) |
| `critical_path(handle)` | The longest path in time through the dependences, the work and the parallelism |
| `source_context(handle, file, line)` | The events of a source line and the source text around it |

Events have stable ids `e<n>` (their position in the trace), unlike the node ids of the graph in the UI.

### Internal transfers

The trace also holds the library's own transfers, which the graph view leaves out and which can be
most of the transfer time: `Internal (reduce-temp)` (partial results of a reduction brought to the
host), `Internal (overlap-wrap)` (edge data of a MapOverlap) and `[Internal PRNG]`. `transfers` reports
them by kind; this is read from the labels, since the tracer records no kind.

## Status

A prototype. The UI's worker (`script/backend/visualizer.js`) and the server share the code that
reads traces (`trace.js`) and builds the analysed graph (`pipeline.js`); `tests/unit/test_worker_golden.js`
pins the worker's output on the shipped traces. Nodes have a key (`e<n>`, `e<n>.<k>`, `root`) that is
the same whenever a graph is built from the same events; the ids of nodes are random UUIDs.

Fusion candidates are decided by `script/backend/fusion.js` from the events and the containers they
read and write (rules, assumptions and the audit of the graph view's heuristic in [FUSION.md](FUSION.md)).
They are candidates: that fusing can be done does not say that it pays, and the pattern tables are
assumptions about what SkePU can fuse.

Open: tracer fields for the kind of a transfer and the skeleton call it belongs to, so that the
attribution does not rest on labels and order; the `{header, data}` trace format (read, but the
source hashes are not checked); comparing two traces (backends, before and after a change).

`npm test` runs the tests (`tests/unit/test_analysis.js`, `test_mcp.js`).
