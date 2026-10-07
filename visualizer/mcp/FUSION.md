# Fusion analysis: rules and audit

`script/backend/fusion.js` finds the skeleton calls of a trace that could be fused into one call
(serial: producer and consumer; parallel: independent siblings), and reports why other pairs could not.
It replaces the heuristic that the graph view used (`findSerialFusions`, `findParallelFusions` in `graph.js`,
see the audit below): the analysis server and the views (View > Fusion analysis) now use this detector.

## Rules

Everything is decided from the events of the trace, by the containers (object ids) they read and write.

**Serial.** A call A that writes a container C, and a call B that reads that version of C, are one fused
call if:

1. B is the only reader of that version of C: no other call, no host access (element access, snapshot,
   transfer, external), until C is overwritten or freed. If C is never overwritten or freed in the trace it
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
so (`recompute`): for a MapPairs the factor is the length of the other vector (known from the trace; the Map must be
over one of the two vectors); for a MapOverlap each element is computed once for each output whose neighbourhood
contains it; a MapPool computes nothing twice if its windows do not overlap, and as a MapOverlap if they do (the trace
has neither the radius nor the windows, so the factor is not given). Output extents may differ from the input's
there. `rules.allowRecompute = false` leaves these out (reason `recompute`). There is no fused form of a Map into a Scan
(no MapScan: its proxies and uniform arguments), and a Reduce cannot be fused into what reads its scalar result.

**Parallel.** Independent calls of a fusable pattern (`rules.parallel`, default Map, Reduce, MapPairs) over the same
elements on the same backend, none of which writes what another reads or writes, and no two sharing a PRNG stream,
under rule 4. Grouping is greedy in trace order (first fit among the latest groups). Calls of serial groups are
left out unless `include_overlap`.

The result lists the groups with the intermediates (and how they end), the launches saved, the saved memory
traffic in elements (a write and a read per eliminated intermediate), and the calls that cannot be fused
with the reason (`multiple-readers`, `host-access`, `proxy-access`, `pattern`, `backend-mismatch`,
`size-mismatch`, `shared-prng`, `scalar-output`, `recompute`, `hazard`). Equal groups of the iterations of a loop are one
entry with their number.

## What is assumed, and not checked

* Which patterns SkePU can fuse (the `rules` tables; `examples/fusion.cpp` has the `>>`, `||` and `^` that exist:
  Map or MapPairs into Map or Reduce, side by side Maps, Reduces and MapPairs). MapOverlap/MapPool as producers,
  and Scan, are left out. MapPairsReduce as a producer is not considered.
* That the Map into a MapPool/MapOverlap/MapPairs would be fused by SkePU at all; it is reported as possible, with
  its cost.
* That a MapPool is traced as one: the tracer reports a pool as `MapOverlap 4D` today, so it is taken as a stencil
  (recomputing) unless the pattern is named `MapPool`.
* That the trace has all the accesses: an access that leaves no event (the host reading a container that
  was already transferred) cannot be seen.
* That an element-wise call reads and writes only its own element.
* Execution parameters (tuning, execution plans, thread counts) and user function properties
  (side effects other than the PRNG) are not in the trace.
* Fusing *can* be done does not say it *pays*: the time of the calls and the saved traffic are reported,
  not a prediction.

## Audit of the earlier heuristic of the graph view

Found by running it on small traces (`tests/unit/test_fusion.js` has the new behaviour of each case):

| Case | Graph view | Now |
|---|---|---|
| Producer's input is overwritten before the consumer (write after read) | fused | fused only by moving the consumer up; blocked if neither can move. The graph has no edges for write-after-read |
| Element counts differ | fused | blocked (`size-mismatch`) |
| Backends differ | not fused, by accident (no edge across memory spaces) | blocked (`backend-mismatch`) |
| Map into Scan | fused | blocked (`pattern`), table is settable |
| MapOverlap into Map | fused, as a chain with the stencil | blocked (`pattern`), table is settable |
| Both calls draw from one PRNG stream | fused | blocked (`shared-prng`) |
| The producer's two outputs go to two consumers | two overlapping groups with the producer in both | one group |
| Intermediate never freed (live-out) | not told | reported, saves only the read |
| Reasons for not fusing | none | per pair, with counts |
| Parallel fusion | never runs (`findFusions` does not compute the equivalence classes first) | runs |
| Parallel fusion when run by hand | fuses different element counts, different backends and a call that overwrites the other's input | checked |
| Fusion changes the graph that the analysis reads | yes (nodes are regrouped) | no, works on the events |

## In the views

With Fusion analysis on, each group is a compound 'fusion' node (the 'Fusion Suggestion' of the legend) around
its calls and the versions of the containers that fusion would not write; its info pane says the kind, the
calls, the containers not written or read, where the fused call would go and the launches saved. The info pane of
a call that could not be fused with a consumer of its output says why ('Not fused'). A group whose calls are in
different regions is not drawn, because that would take them out of their regions (the analysis server lists it).
With iterations coalesced, the groups of a loop are one node with the repeat count. `pipeline.js` (`applyFusion`)
does this.

`findFusions` and the functions it calls in `graph.js` are no longer used by the views; they are kept, with
their tests (`test_serial_fusion.js`), until it is decided to remove them.
