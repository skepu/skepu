#!/usr/bin/env python3
"""Checks the trace files of tests/mpi/trace_map.cpp, run on P processes: check_trace.py <directory> <P>.

The first process writes the trace of the program, the others lean files (see skepu3/mpi/trace.hpp). What is checked:
the files and their headers; that the processes agree on the calls (seq, pattern and signature) and on the collectives (comm_seq); that the
local elements of a call add up to the elements of the call; that the bytes sent by all processes in a collective are the bytes
received; that the number of events is the same for every call (O(P), nothing that grows with the problem); that the ids of a call
are of containers that were allocated; and that, on the clock of rank 0, a collective ends at about the same time everywhere.
"""
import hashlib, json, os, sys

def load(path):
    with open(path) as f:
        return json.load(f)

def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)

directory, P = sys.argv[1], int(sys.argv[2])
names = ["skepu_trace.json"] + ["skepu_trace.rank%d.json" % k for k in range(1, P)]
files = []
for k, name in enumerate(names):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        fail("missing " + path)
    t = load(path)
    if not isinstance(t, dict) or "header" not in t or "data" not in t:
        fail(name + ": not { header, data }")
    h = t["header"]
    if h["mpi"]["rank"] != k or h["mpi"]["ranks"] != P:
        fail(name + ": wrong rank/ranks in the header: %s" % h["mpi"])
    if h["mpi"]["role"] != ("full" if k == 0 else "lean"):
        fail(name + ": wrong role")
    if "start" not in h["clock"] or "end" not in h["clock"]:
        fail(name + ": no clock samples")
    files.append(t)

full = files[0]["data"]
calls0 = [e for e in full if e["type"] in ("skeleton_call", "external") and "seq" in e]   # (an external is a call too)
comms0 = [e for e in full if e["type"] == "communication"]
allocs = {e["object_id"]: e for e in full if e["type"] == "allocation"}
if not calls0:
    fail("no skeleton_call events on rank 0")

# nothing of the node-level library on rank 0, and nothing but lean events elsewhere
for e in full:
    if e["type"] not in ("allocation", "deallocation", "skeleton_call", "communication", "region", "external", "transfer", "snapshot", "element_access", "scalar_value"):
        fail("rank 0: unexpected event type " + e["type"])
    if e["type"] == "transfer":
        fail("rank 0: a node-level transfer event (those are not recorded in MPI programs)")
for k in range(1, P):
    for e in files[k]["data"]:
        if e["type"] not in ("rank_call", "rank_communication"):
            fail("rank %d: not a lean event: %s" % (k, e["type"]))
        if "object_id" in e or "object_ids" in e or "outputs" in e:
            fail("rank %d: a lean event has ids" % k)

# the calls
lean_calls = [{e["seq"]: e for e in files[k]["data"] if e["type"] == "rank_call"} for k in range(1, P)]
for c in calls0:
    seq = c["seq"]
    m = c["mpi"]
    total = m["local_elements"]
    for k in range(1, P):
        if seq not in lean_calls[k - 1]:
            fail("rank %d has no call seq %d" % (k, seq))
        l = lean_calls[k - 1][seq]
        pattern = c.get("pattern", "external")
        if l["pattern"] != pattern or l["signature"] != m["signature"]:
            fail("rank %d disagrees about call %d: %s vs %s" % (k, seq, l["pattern"], pattern))
        total += l["local_elements"]
        if not (l["start"] <= l["work_start"] <= l["work_end"] <= l["end"] or l["work_start"] == 0):
            fail("rank %d call %d: the work is not inside the call" % (k, seq))
    expected = 1
    for dimension in c.get("elements", [0]):
        expected *= dimension
    if c["type"] == "skeleton_call" and total != expected:
        fail("call %d (%s): local elements add up to %d, not %d" % (seq, c["pattern"], total, expected))
    if c["type"] == "external" and (m["work_start"] == 0 or total != 0):
        fail("external %d: the code runs on rank 0 and has no elements" % seq)
    if c["start"] > c["end"]:
        fail("call %d: start after end" % seq)
    for key in ("outputs", "elwise_inputs", "proxy_inputs"):
        if key == "outputs" and c.get("pattern") in ("Reduce 1D-Full", "MapReduce"):
            continue   # (the outputs are the ids of scalars, which are not allocated)
        for i in c[key]:
            if i not in allocs:
                fail("call %d: %s has the id %d, which was not allocated" % (seq, key, i))
for k in range(1, P):
    if len(lean_calls[k - 1]) != len(calls0):
        fail("rank %d has %d calls, rank 0 has %d" % (k, len(lean_calls[k - 1]), len(calls0)))

# the collectives
lean_comms = [{e["comm_seq"]: e for e in files[k]["data"] if e["type"] == "rank_communication"} for k in range(1, P)]
if not comms0:
    fail("no communication events on rank 0")
reasons = set()
spreads = []
for c in comms0:
    cs = c["comm_seq"]
    sent, received = c["bytes_sent"], c["bytes_received"]
    ends = [c["end"]]
    reasons.add(c["reason"].split(":")[0])
    if (c["seq"] != 0) != ("phase" in c) or c.get("phase", "before") not in ("before", "after"):
        fail("collective %d: phase %s for the call %d" % (cs, c.get("phase"), c["seq"]))
    for k in range(1, P):
        if cs not in lean_comms[k - 1]:
            fail("rank %d has no collective %d" % (k, cs))
        l = lean_comms[k - 1][cs]
        if l["operation"] != c["operation"] or l["seq"] != c["seq"] or l.get("phase") != c.get("phase"):
            fail("rank %d disagrees about collective %d" % (k, cs))
        sent += l["bytes_sent"]
        received += l["bytes_received"]
        # on the clock of rank 0
        off = files[k]["header"]["clock"]["start"]["offset_us"]
        ends.append(l["end"] + off)
    if sent != received:
        fail("collective %d (%s): %d bytes sent, %d received" % (cs, c["reason"], sent, received))
    spreads.append(max(ends) - min(ends))
for k in range(1, P):
    if len(lean_comms[k - 1]) != len(comms0):
        fail("rank %d has %d collectives, rank 0 has %d" % (k, len(lean_comms[k - 1]), len(comms0)))

# (with SKEPU_TRACE_TEST_SKEW_US the clocks of the processes start at different times: the raw times are far apart, the aligned ones are not)
skew = int(os.environ.get("SKEPU_TRACE_TEST_SKEW_US", "0"))
if skew > 0 and P > 1:
    raw = []
    for c in comms0:
        ends = [c["end"]] + [lean_comms[k - 1][c["comm_seq"]]["end"] for k in range(1, P)]
        raw.append(max(ends) - min(ends))
    if min(raw) < skew * (P - 1) / 2:
        fail("the test did not skew the clocks: the raw times are only %d us apart" % min(raw))
worst = max(spreads) if spreads else 0
if worst > 20000:
    fail("the collectives end up to %d us apart on the clock of rank 0 (the clocks are not aligned)" % worst)

# where the instances and their user functions are defined: rank 0 has a table with all the instances that made a call, and the other ranks do not
instances = files[0]["header"].get("instances", {})
for e in calls0:
    if e["type"] == "skeleton_call" and str(e["object_id"]) not in instances:
        fail("the instance %s of a call is not in the table of the instances" % e["label"])
for e in instances.values():
    if not e["user_functions"] or not e["instance"].get("line") or not all(f.get("line") for f in e["user_functions"]):
        fail("the instance %s has no lines" % e["label"])
for t in files[1:]:
    if "instances" in t["header"]:
        fail("a rank that is not the first has a table of the instances")

# the source files of the trace: rank 0 has the SHA-256 of each file that its events name (the file of a call is hashed by the precompiler, as compiled)
hashes = files[0]["header"].get("files", {})
named = set(e["file"] for e in files[0]["data"] if e.get("file"))
if named != set(hashes):
    fail("header.files has %s, the events name %s" % (sorted(hashes), sorted(named)))
for path, h in hashes.items():
    if h != hashlib.sha256(open(path, "rb").read()).hexdigest():
        fail("the hash of %s in the header is not the one of the file" % path)

# the containers are named after their variables (skepu-tool does it for the ones that the program does not name itself),
# also in the allocation events that came before the name
names = {e["label"] for e in allocs.values()}
for name in ("c", "row_totals", "col_totals", "scanned", "smoothed", "pairs_out", "pooled"):
    if name not in names:
        fail("no container named %s in the trace (the names: %s)" % (name, sorted(names)))
if any(label.startswith(("Vector #", "Matrix #")) for label in names):
    fail("a container has no name: %s" % sorted(names))

# a container that has only its initial value is not gathered (nothing is moved to read what every process knows)
for e in full:
    if e["type"] == "allocation" and e.get("label") in ("fresh", "fresh_matrix"):
        for c in comms0:
            if e["object_id"] in c["object_ids"] and c["operation"] == "allgather" and c["seq"] == 0:
                fail("the container %s was gathered before anything was written to it: %s" % (e["label"], c["reason"]))

# the data of an skepu::external call is not moved when the call ends: it goes where it is needed, when it is (a copy to all
# for the host, the tiles for a skeleton)
external_seqs = {c["seq"] for c in calls0 if c["type"] == "external"}
for c in comms0:
    if c["operation"] == "allgather" and c["seq"] in external_seqs:
        fail("collective %d: an external call has the container on every process, and the code runs on one: %s" % (c["comm_seq"], c["reason"]))
    if c["operation"] == "scatter" and c["seq"] in external_seqs:
        fail("collective %d: the container of an external call is scattered in the call: %s" % (c["comm_seq"], c["reason"]))
if not any(c["operation"] == "gather" and c["seq"] in external_seqs for c in comms0):
    fail("no gather to the first process for an external call")
if not any(c["operation"] == "bcast" for c in comms0):
    fail("no bcast: the host reads data that an external call wrote")

# the line numbers of the source are kept by skepu-tool where it replaces a skeleton declared over several lines
source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_map.cpp")).read().split("\n")
marked = [i + 1 for i, l in enumerate(source) if '"line-mark"' in l]
found = [e["line"] for e in full if e["type"] == "external" and e.get("label") == "line-mark"]
if found != marked:
    fail("the external 'line-mark' is on line %s in the trace, and on line %s in the source" % (found, marked))

# skepu::randomize::uniform is a skeleton call that writes the container, in a region of the library function, as in a single node
rnd_ids = [e["object_id"] for e in full if e["type"] == "allocation" and e.get("label") == "rnd"]
if not any(e["type"] == "skeleton_call" and e.get("label") == "randomize::uniform" and e["outputs"] == rnd_ids and e.get("region") == "randomize.uniform" for e in full):
    fail("no skeleton call 'randomize::uniform' in the region 'randomize.uniform' that writes the container rnd")

# the elements that were read (Vector::elem): of a container that was allocated, and given to a call as a scalar
accesses = [e for e in full if e["type"] == "element_access"]
if not accesses:
    fail("no element_access event on rank 0")
scalar_inputs = set()
for c in calls0:
    scalar_inputs.update(c.get("scalar_inputs", []))
for e in accesses:
    if e["source"] not in allocs:
        fail("element_access of the container %d, which was not allocated" % e["source"])
    if e["object_id"] not in scalar_inputs:
        fail("element_access %d is not used by a call" % e["object_id"])
for k in range(1, P):
    if any(e["type"] == "element_access" for e in files[k]["data"]):
        fail("rank %d has an element_access event" % k)

# the values of the scalars (scalar_value): events of the first process, a number or one of the strings of a number that is not finite, and the value of
# the output of a call for some of them (the others are scalars that the program or the library made)
call_outputs = set()
for c in calls0:
    call_outputs.update(c.get("outputs", []))
values = [e for e in full if e["type"] == "scalar_value"]
if not values:
    fail("no scalar_value event on rank 0")
for e in values:
    if not (isinstance(e["value"], (int, float)) or e["value"] in ("nan", "inf", "-inf")):
        fail("scalar_value %d has no number: %r" % (e["object_id"], e["value"]))
for k in range(1, P):
    if any(e["type"] == "scalar_value" for e in files[k]["data"]):
        fail("rank %d has a scalar_value event" % k)
if not any(e["object_id"] in call_outputs for e in values):
    fail("no scalar_value is the value of the output of a call")

# the snapshots (SKEPU_TRACE_SNAPSHOT of a distributed container): events of the first process, whose files have the whole container
snapshots = [e for e in full if e["type"] == "snapshot"]
for e in snapshots:
    path = os.path.join(directory, e["data_file"])
    if not os.path.exists(path):
        fail("snapshot %s: no file %s" % (e["label"], path))
    with open(path) as f:
        header = f.readline().split()
    size = 1
    for d in header[1:]:
        size *= int(d)
    if int(header[0]) not in (1, 2, 3, 4) or size != len(open(path).read().split("\n", 1)[1].split()):
        fail("snapshot %s: the file does not have all %s elements" % (e["label"], header[1:]))
for k in range(1, P):
    if any(e["type"] == "snapshot" for e in files[k]["data"]):
        fail("rank %d has a snapshot event" % k)

# the number of events: the same for every call
per_call = {}
for k in range(P):
    for e in files[k]["data"]:
        if e["type"] in ("skeleton_call", "external", "rank_call"):
            per_call[e["seq"]] = per_call.get(e["seq"], 0) + 1
if set(per_call.values()) != {P}:
    fail("not one event per process for each call: %s" % sorted(set(per_call.values())))

print("OK: %d processes, %d calls, %d collectives (%s), %d snapshots, collectives end within %d us on the clock of rank 0"
      % (P, len(calls0), len(comms0), ", ".join(sorted(reasons)), len(snapshots), worst))
