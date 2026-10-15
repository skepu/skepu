# Fusion analysis: rules

`script/backend/fusion.js` finds the skeleton calls of a trace that could be fused into one call
(serial: producer and consumer; parallel: independent siblings), and reports why other pairs could not.
The analysis server and the views (View > Fusion analysis) use this detector.

## Rules

Everything is decided from the events of the trace, by the containers (object ids) they read and write.

**Serial.** A call A that writes a container C, and a call B that reads that version of C, are one fused
call if:

1. B is the only reader of that version of C: no other call, no host access (element access, snapshot,
   transfer, copy, external), until C is overwritten or freed. If C is never overwritten or freed in the trace it
   is *live-out*: still fusable, but C must still be written, so only the read is saved.
2. B reads C element by element, not through a proxy (random access).
3. The patterns are fusable (`rules.serialProducers` x `rules.serialConsumers`: a Map or a MapPairs into a Map,
   a Reduce or a MapReduce; a Map into a Reduce is a MapReduce) or fusable at the price of recomputation
   (below), A's output is not a scalar, they run on the same backend over the same elements, and do not draw
   from the same PRNG stream.
4. The fused call can be put at the place of the last call of the group (the earlier ones *sink*) or of the
   first (the later ones *hoist*): no event in between conflicts with a call that has to move past it
   (it writes what the call reads or writes, or reads what it writes). The intermediate containers are not
   written, so they do not conflict.
5. Inside the group the calls keep their order for each element, so a call that reads a container through a
   proxy must not share it with another call of the group that writes it.

Pairs that satisfy 1-3 are joined into groups (chains, fan-in and fan-out trees), a pair being added only if the
whole group still satisfies 4 and 5; a pair that fails may be added later when the calls in its way have joined.

**Recomputation.** A Map can also be fused into a call that reads its input with a neighbourhood or in pairs
(`rules.serialConsumersRecompute`: MapOverlap, MapPool, MapPairs, MapPairsReduce). The fused call computes the
Map's function once for each output that reads an element, so it trades memory traffic for work, and the group says
so (`recompute`, with the factor):

* MapPairs: the length of the other vector (from the sizes; the Map must be over one of the two vectors);
* MapOverlap: the number of outputs whose neighbourhood contains an element, the product over the dimensions of
  `ceil((2 * overlap + 1) / stride)`, from the `parameters` of the call;
* MapPool: the product of `ceil(pool_size / stride)`: **1 if the windows do not overlap, and then the fusion is
  free** (no `recompute` entry), as for a Map into a Map.

Without `parameters` in the trace (an older trace) the factor of a MapOverlap and a MapPool is not known, and a pool
is taken as a stencil. Output extents may differ from the input's in these consumers. `rules.allowRecompute = false`
leaves these out (reason `recompute`). There is no fused form of a Map into a Scan (no MapScan: its proxies and
uniform arguments), and a Reduce cannot be fused into what reads its scalar result. A Map or MapReduce with a stride
(`parameters.strides`) does not read and write the elements one to one and is not fused (`strided`).

**Parallel.** Independent calls of a fusable pattern (`rules.parallel`, default Map, Reduce, MapPairs) over the same
elements on the same backend, none of which writes what another reads or writes, and no two sharing a PRNG stream,
under rule 4. Grouping is greedy in trace order (first fit among the latest groups). Calls of serial groups are
left out unless `include_overlap`.

The result lists the groups with the intermediates (and how they end), the launches saved, the saved memory
traffic in elements (a write and a read per eliminated intermediate), and the calls that cannot be fused
with the reason (`multiple-readers`, `host-access`, `proxy-access`, `pattern`, `backend-mismatch`,
`size-mismatch`, `shared-prng`, `scalar-output`, `recompute`, `strided`, `hazard`). Equal groups of the iterations of a loop are one
entry with their number.

## What is assumed, and not checked

* Which patterns SkePU can fuse (the `rules` tables; `examples/fusion.cpp` has the `>>`, `||` and `^` that exist:
  Map or MapPairs into Map or Reduce, side by side Maps, Reduces and MapPairs). MapOverlap/MapPool as producers,
  and Scan, are left out. MapPairsReduce as a producer is not considered.
* That the Map into a MapPool/MapOverlap/MapPairs would be fused by SkePU at all; it is reported as possible, with
  its cost.
* That the trace has all the accesses: an access that leaves no event (the host reading a container that
  was already transferred) cannot be seen.
* That an element-wise call reads and writes only its own element.
* Tuning and execution plans, and the properties of the user functions (side effects other than the PRNG), are not in the trace.
* Fusing *can* be done does not say it *pays*: the time of the calls and the saved traffic are reported,
  not a prediction.

## In the views

With Fusion analysis on, each group is a compound 'fusion' node (the 'Fusion Suggestion' of the legend) around
its calls and the versions of the containers that fusion would not write; its info pane says the kind, the
calls, the containers not written or read, where the fused call would go and the launches saved. The info pane of
a call that could not be fused with a consumer of its output says why ('Not fused'). A group whose calls are in
different regions is not drawn, because that would take them out of their regions (the analysis server lists it).
With iterations coalesced, the groups of a loop are one node with the repeat count. `pipeline.js` (`applyFusion`)
does this.
