from flask import Flask, render_template, request, jsonify, json
import uuid, copy, math
import os, time, webbrowser
from collections import defaultdict


backendToMemSpace = {
    "CPU" : "CPU",
    "OpenMP" : "CPU",
    "OpenCL" : "GPU (OpenCL)",
    "CUDA" : "GPU (CUDA)"
}

allBackends  = [key for key in backendToMemSpace]
allMemSpaces = set([backendToMemSpace[key] for key in backendToMemSpace])

def makeMemSpaceDict():
    res = { "ANY" : None }
    for memspace in allMemSpaces:
        res[memspace] = None
    return res

class GraphSettings:

    def __init__(self):
        self.updates = True
        self.transfers = True
        self.antideps = True
        self.allocations = True
        self.deallocations = True
        self.regions = True

class DirectedGraph:


    def __init__(self):

        self._nodes = []
        self._edges = []

        self._regions = {}
        self._openRegions = []

        self._labelByID = {}

        self._nodeLabelIndex = {}
        self._nodeIdIndex = {}
        self._edgeIdIndex = {}

        self._latestProducerOfID = defaultdict(makeMemSpaceDict)
        self._latestConsumerOfID = defaultdict(makeMemSpaceDict)

        self.update_versions = defaultdict(int)

        self.comesBeforeCache = defaultdict(dict)
        self.equivalenceClasses = {}

        self.settings = GraphSettings()


    def addNode(self, node):
        self._nodes.append(node)
        self._nodeIdIndex[node.id] = node
        self._nodeLabelIndex[node.label] = node

    def addEdge(self, edge):
        sourceNode = self._nodeIdIndex.get(edge.source.id)
        targetNode = self._nodeIdIndex.get(edge.target.id)

        sourceNode.addOutgoingEdge(edge)
        targetNode.addIncomingEdge(edge)

        self._edges.append(edge)
        self._edgeIdIndex[edge.id] = edge

    def getNodeById(self, nid):
        return self._nodeIdIndex[nid]

    def getNodeByLabel(self, label):
        return self._nodeLabelIndex[label]

    def setLabelForID(self, label, id):
        self._labelByID[id] = label

    def labelForID(self, id):
        try:
            return self._labelByID[id]
        except KeyError:
            return ""

    def newVersionNumberForID(self, id):
        self.update_versions[id] += 1
        return self.update_versions[id]

    def depthByLabel(self, label):
        try:
            return self._nodeLabelIndex[label].depth
        except KeyError:
            return 0

    def getEdgeById(self, eid):
        return self._edgeIdIndex[eid]

    def getAllNodes(self):
        return self._nodes

    def getRootNodes(self):
        return [node for node in self._nodes if len(node.getIncomingEdges()) == 0]

    def getLeafNodes(self):
        return [node for node in self._nodes if len(node.getOutgoingEdges()) == 0]

    def getGlobalRegionNodes(self):
        return [node for node in self._nodes if not node.region]

    def getGlobalFreeNodes(self):
        return [node for node in self._nodes if not node.region and not isinstance(node, RegionNode)]

    def getAllRegions(self):
        return [node for node in self._nodes if isinstance(node, RegionNode)]

    # Graph builder

    def writeToID(self, id, node, backend):
        self.readFromID(id, node, backend)
        self._latestProducerOfID[id]["ANY"] = node
        if backend == "ALL":
            for memspace in allMemSpaces:
                self._latestProducerOfID[id][memspace] = node
        else:
            self._latestProducerOfID[id][backendToMemSpace[backend]] = node

    def readFromID(self, id, node, backend):
        self._latestConsumerOfID[id]["ANY"] = node
        if backend == "ALL":
            for memspace in allMemSpaces:
                self._latestConsumerOfID[id][memspace] = node
        else:
            self._latestConsumerOfID[id][backendToMemSpace[backend]] = node

    def transferIDMemspace(self, object_id, source_backend, target_backend):
        node = self._latestProducerOfID[object_id][backendToMemSpace[source_backend]]
        self._latestProducerOfID[object_id][backendToMemSpace[target_backend]] = node

    def producerForID(self, id, backend = "ANY"):
        memspace = backendToMemSpace[backend] if backend != "ANY" else "ANY"
        try:
            return self._latestProducerOfID[id][memspace]
        except KeyError:
            return None

    def accessorForID(self, id, backend = "ANY"):
        memspace = backendToMemSpace[backend] if backend != "ANY" else "ANY"
        try:
            return self._latestConsumerOfID[id][memspace]
        except KeyError:
            return None

    def visitNodes(self, f):
        pass

    def pruneEmptyRegions(self):
        """Remove RegionNodes that contain no nested nodes, iterating to
        convergence so that removing a leaf region can expose an empty parent."""
        changed = True
        while changed:
            changed = False
            for node in list(self._nodes):
                if isinstance(node, RegionNode) and len(node.nested_nodes) == 0:
                    self._nodes.remove(node)
                    self._nodeIdIndex.pop(node.id, None)
                    self._nodeLabelIndex.pop(node.label, None)
                    if node.region:
                        try:
                            node.region.nested_nodes.remove(node)
                        except ValueError:
                            pass
                    changed = True


    def nodePreceedes(self, n1, n2):
        for edge in n2.getIncomingEdges():
            if edge.source is n1:
                return True

        return False


    def comesBefore(self, n1, n2):

        def helper(n1, n2):
            for edge in n2.getIncomingEdges():
                if self.comesBefore(n1, edge.source):
                    return True
            return False

        if n1 in self.comesBeforeCache and n2 in self.comesBeforeCache[n1]:
            return self.comesBeforeCache[n1][n2]

        if n1 is n2:
            result = False
        elif self.nodePreceedes(n1, n2):
            result = True
        elif self.nodePreceedes(n2, n1):
            result = False
        else:
            result = helper(n1, n2)

        self.comesBeforeCache[n1][n2] = result
    #    print(f"{n1.label} comes before {n2.label}: {result}")
        return result

    def computeEquivalenceClasses(self):

        def eqClassesHelper(nodes):
            eqClasses = []
            for node in nodes:
                found = False
                for eqClass in eqClasses:
                    seemsEq = True
                    for otherNode in eqClass:
                        if self.comesBefore(node, otherNode) or self.comesBefore(otherNode, node):
                            seemsEq = False
                            break
                    if seemsEq:
                        eqClass.append(node)
                        found = True
                        break
                if not found:
                    eqClasses.append([node])

            return eqClasses


        if not self.settings.regions:
            self.equivalenceClasses = eqClassesHelper(self.getGlobalFreeNodes())
        else:
            for region in self.getAllRegions():
            #    print(f"## Computing EQ classes for {region.label}")
                self.equivalenceClasses[region] = eqClassesHelper(region.getDirectChildren())

            for region in self.equivalenceClasses:
                print(f"## EQ classes for {region.label}")

                for eqClass in self.equivalenceClasses[region]:
                    print(f"-- Class:")
                    for node in eqClass:
                        print(f"\t{node.label}")

    # Processing and analysis
    def computeKeyPaths(self):

        def helper(node, keypath):
            mykey = node.type + "_" + node.label
            if isinstance(node, ComputationNode):
                mykey += "["
                for id in node.elwise_inputs:
                    mykey += str(id) + "+"
                for id in node.proxy_inputs:
                    mykey += str(id) + "+"
                mykey += "]"

            node.keypath = keypath + "->" + mykey
            if isinstance(node, RegionNode):
                for nested_node in node.nested_nodes:
                    helper(nested_node, node.keypath)

        for node in self.getGlobalRegionNodes():
            helper(node, "[global]")

    def computeDepths(self):

        # Assumes no cycles, otherwise infinite recursion
        def helper(node, current_depth):
            node.depth = max(node.depth, current_depth)
            for child in node.getChildNodes():
                helper(child, current_depth + 1)


        for node in self._nodes:
            node.depth = 0

        for node in self.getRootNodes():
            helper(node, 0)


    def computeLiveness(self):

        for update in filter(lambda node: isinstance(node, UpdateNode), self._nodes):
            print(f"{update.label} version {update.version}")
            live = []
            next_update = None
            for node in self._nodes:
                if node.total_order > update.total_order:
                    if isinstance(node, UpdateNode) and node.label == update.label:
                        next_update = node
                        break

            for node in self._nodes:
                if node.total_order >= update.total_order and (next_update is None or node.total_order < next_update.total_order):
                    live.append(node.id)

            update.is_live = live



    def findCriticalPath(self):
        deepest_node = None
        for node in self._nodes:
            if node.type == "region":
                continue
            if not deepest_node or node.depth > deepest_node.depth:
                deepest_node = node

        current_node = deepest_node
        while True:
            current_node.is_critical_path = True
            max_parent_depth = -1
            max_edge = None
            for edge in current_node.getIncomingEdges():
                parent = edge.source
                if parent.depth > max_parent_depth:
                    max_parent_depth = parent.depth
                    max_edge = edge
                    current_node = parent

            if max_edge:
                max_edge.is_critical_path = True
            else:
                break


    def findFusions(self):
        self.findParallelFusions()
        return self.findSerialFusions()


    def findParallelFusions(self):
        def findFusionsInEqClass(eqClasses, region):
                size = 0
                fused = None
                root = None
                for node in eqClass:
                    if isinstance(node, CallNode) and node.pattern == "Map":
                        if not root:
                            root = node
                            continue
                        if not fused:
                            size = node.elements
                            fused = FusedNode(self, "Parallel Fusion", node.total_order, region)
                            root.fused = fused
                            root.region = fused

                        if node.region:
                            node.region.nested_nodes.remove(node)
                        node.fused = fused
                        node.region = fused


        if self.settings.regions:
            for region in self.equivalenceClasses:
                for eqClass in self.equivalenceClasses[region]:
                    findFusionsInEqClass(eqClass, region)
        else:
            for eqClass in self.equivalenceClasses:
                findFusionsInEqClass(eqClass, None)


        return []

    def findSerialFusions(self):
        fusions = defaultdict(set)

        def fusionHelper(node, fused):

            candidate = True
        #    for edge in node.getOutgoingEdges():
        #        if isinstance(edge, ProxyEdge):
        #            candidate = False

            if not isinstance(node, CallNode):
                if isinstance(node, UpdateNode):
                    candidate = node.fused is not None
                else:
                    candidate = False

            elif fused is not None and node.pattern != "Map":
                candidate = False

            if candidate:
                for parent in [edge.source for edge in node.getIncomingEdges() if isinstance(edge, ElwiseEdge)]:
                    if isinstance(parent, CallNode) or isinstance(parent, UpdateNode):
                        print(f"Detected serial fusion {node.label}, {parent.label}!")
                        fusions[node.id] = parent.id

                        fused = node.fused if node.fused else None

                        if not fused and node.fused:
                            fused = node.fused
                        elif not fused and parent.fused:
                            fused = parent.fused
                        elif not fused:
                            label = f"Serial Fusion" # {node.label} and {parent.label}"
                            fused = FusedNode(self, label, node.total_order, node.region)

                        node.fused = fused
                        parent.fused = fused

                        if node.region and isinstance(node.region, RegionNode):
                            node.region.nested_nodes.remove(node)
                        node.region = fused
                        if parent.region and isinstance(parent.region, RegionNode):
                            parent.region.nested_nodes.remove(parent)
                        parent.region = fused

                        if isinstance(parent, UpdateNode):
                            parent = parent.getIncomingEdges()[0].source

                        fusionHelper(parent, fused)
            else:
                for parent in [edge.source for edge in node.getIncomingEdges()]:
                    if not parent.fused:
                        fusionHelper(parent, None)



        for node in self.getLeafNodes():
            fusionHelper(node, None)


        print(fusions)
        result = []
        for key in fusions:
            result.append(fusions[key])

        return result


    def coalesceIterations(self):

        self.computeKeyPaths()

        unique_keypaths = set()
        for node in self._nodes:
            unique_keypaths.add(node.keypath)

        canonicals = {}
        for keypath in unique_keypaths:
            nodes = [node for node in self._nodes if node.keypath == keypath]
            canonicals[keypath] = nodes[0]

        for keypath in unique_keypaths:
            nodes = [node for node in self._nodes if node.keypath == keypath]
            rest_nodes = nodes[1:]

            for node in rest_nodes:
                self._nodes.remove(node)
                canonicals[keypath].iteration_count += 1
                canonicals[keypath].durations.extend(node.durations)
                canonicals[keypath].intervals.extend(node.intervals)

                for edge in node.getIncomingEdges():
                    edge.target = canonicals[edge.target.keypath]

                for edge in node.getOutgoingEdges():
                    edge.source = canonicals[edge.source.keypath]

        # After re-pointing all edges to canonical nodes, many edges share the
        # same (source, target) pair. Keep one per directed pair; accumulate
        # the absorbed count on the surviving canonical edge.
        canonical_edge = {}  # (source_id, target_id) -> surviving edge
        deduped = []
        for edge in self._edges:
            key = (edge.source.id, edge.target.id)
            if key not in canonical_edge:
                canonical_edge[key] = edge
                deduped.append(edge)
            else:
                canonical_edge[key].iteration_count += 1
        self._edges = deduped


    # Serialization

    def summary(self):
        print(f"Number of nodes: {len(self._nodes)}")
        print(f"Number of edges: {len(self._edges)}")

    def toCytoscape(self):

        cy_nodes = []
        cy_edges = []

        for node in self._nodes:
            cy_nodes.append({"data":node.toCytoscape()})

        for edge in self._edges:
            cy_edges.append({"data":edge.toCytoscape()})

        return (cy_nodes, cy_edges)



class Node:

    def __init__(self, graph, label, total_order, region=None):
        self.id = str(uuid.uuid4())
        self.label = label
        self.graph = graph
        self.total_order = total_order
        self._incoming = []
        self._outgoing = []
        self.depth = 0
        self.keypath = ""
        self.fused = None

        self.is_critical_path = False
        self.iteration_count = 1
        self.durations = []   # populated by ComputationNode; accumulated during coalesceIterations
        self.intervals = []   # list of (start, end) tuples; populated by ComputationNode
        self.region = None
        self.nesting_level = 0
        if graph.settings.regions:
            if isinstance(region, RegionNode):
                self.region = region
            elif region:
                self.region = graph.getNodeByLabel(region)
            if self.region:
                self.region.addNestedNode(self)
                self.nesting_level = self.region.nesting_level + 1

        graph.addNode(self)

    def addIncomingEdge(self, edge):
        assert(edge.target.id == self.id)
        self._incoming.append(edge)

    def addOutgoingEdge(self, edge):
        assert(edge.source.id == self.id)
        self._outgoing.append(edge)

    def getIncomingEdges(self):
        return self._incoming

    def getOutgoingEdges(self):
        return self._outgoing

    def getParentNodes(self):
        return [edge.source for edge in self._incoming]

    def getChildNodes(self):
        return [edge.target for edge in self._outgoing]

    def toCytoscape(self):
        cy_data = {}
        cy_data["id"] = self.id
        cy_data["label"] = self.label
        cy_data["total_order"] = self.total_order
        cy_data["type"] = self.type
        cy_data["is_critical_path"] = self.is_critical_path
        cy_data["dag_depth"] = self.depth
        cy_data["iteration_count"] = self.iteration_count
        if self.fused:
            cy_data["parent"] = self.fused.id
        elif self.region:
            cy_data["parent"] = self.region.id
            cy_data["region_depth"] = self.nesting_level
        return cy_data

    def infoData(self):
        info_data = {}
        info_data["internal"] = {}
        info_data["type"] = self.type
    #    info_data["On critical path"] = self.is_critical_path
    #    info_data["DAG depth"] = self.depth
        if self.iteration_count > 1:
            info_data["Repeat count"] = self.iteration_count
        if self.region:
            info_data["Region"] = graph.labelForID(self.region.id)
            info_data["Region Nesting Depth"] = self.nesting_level
        return info_data


class ComputationNode(Node):

    def __init__(self, graph, json_data, total_order):
        super(ComputationNode, self).__init__(graph, json_data["label"], total_order, json_data.get("region"))

        self.start = json_data["start"]
        self.end = json_data["end"]
        self.duration = self.end - self.start
        self.durations = [self.duration]
        self.intervals = [(self.start, self.end)]

        self.backend = json_data.get("backend", "CPU")
        self.file = json_data.get("file")
        self.line = json_data.get("line")

        self.elwise_inputs = json_data["elwise_inputs"]
        self.proxy_inputs = json_data["proxy_inputs"]
        self.scalar_inputs = json_data["scalar_inputs"]

        for id in self.elwise_inputs:
            ElwiseEdge(graph, graph.labelForID(id), id, self)
            graph.readFromID(id, self, self.backend)

        for id in self.proxy_inputs:
            ProxyEdge(graph, graph.labelForID(id), id, self)
            graph.readFromID(id, self, self.backend)

        for id in self.scalar_inputs:
            ScalarEdge(graph, graph.labelForID(id), id, self)
            graph.readFromID(id, self, "ALL")

        for id in json_data['outputs']:
            if graph.settings.antideps and id not in json_data["elwise_inputs"]:
                AntiDepEdge(graph, graph.labelForID(id), id, self)
            if graph.settings.updates:
                if not self.hasScalarOutput():
                    ElwiseEdge(graph, graph.labelForID(id), self, UpdateNode(graph, id, self.total_order + 1, self.region, self.backend))
                else:
                    ScalarEdge(graph, graph.labelForID(id), self, ScalarNode.fromID(graph, id, self.total_order + 1, self.region))
            else:
                graph.writeToID(id, self, self.backend)
        
        if "prng" in json_data:
            id = json_data["prng"]
            ElwiseEdge(graph, graph.labelForID(id), id, self)
        #    graph.readFromID(id, self, self.backend)
            graph.writeToID(id, self, self.backend)

    def hasScalarOutput(self):
        return False

    def toCytoscape(self):
        cy_data = super(ComputationNode, self).toCytoscape()
        cy_data["start"] = self.start
        cy_data["end"] = self.end
        cy_data["duration"] = self.duration
        cy_data["intervals"] = self.intervals
        cy_data["file"] = self.file
        cy_data["line"] = self.line
        cy_data["backend"] = self.backend
        return cy_data

    def infoData(self):
        info_data = super(ComputationNode, self).infoData()
        info_data["Duration"] = self.duration
        info_data["File"] = self.file
        info_data["Line"] = self.line
        info_data["Backend"] = self.backend
        info_data["durations"] = [{"x": i, "y": d} for i, d in enumerate(self.durations)]
        return info_data


class CallNode(ComputationNode):

    def __init__(self, graph, json_data, total_order):
        self.type = "skeleton_call"
        self.pattern = json_data["pattern"]
        self.elements = json_data["elements"]
        super(CallNode, self).__init__(graph, json_data, total_order)

    def hasScalarOutput(self):
        # Todo: Reduce2D special case
        return self.pattern in ["Reduce", "MapReduce"]

    def toCytoscape(self):
        cy_data = super(CallNode, self).toCytoscape()
        cy_data["pattern"] = self.pattern
        cy_data["elements"] = self.elements
        return cy_data

    def infoData(self):
        info_data = super(CallNode, self).infoData()
        info_data["Pattern"] = self.pattern
        info_data["Elements"] = self.elements
        return info_data


class ExternalNode(ComputationNode):

    def __init__(self, graph, json_data, total_order):
        super(ExternalNode, self).__init__(graph, json_data, total_order)
        self.type = "external"

    def toCytoscape(self):
        cy_data = super(ExternalNode, self).toCytoscape()
        return cy_data


class RegionNode(Node):

    def __init__(self, graph, json_data, total_order):
        if graph.settings.regions:
            super(RegionNode, self).__init__(graph, json_data["label"], total_order, json_data.get("region"))
            graph.setLabelForID(json_data["label"], self.id)
            self.type = "region"
            self.region_depth = json_data["region_depth"]
            self.nested_nodes = []
            self.region_start = None   # computed by computeExtent()
            self.region_end   = None

    def addNestedNode(self, node):
        self.nested_nodes.append(node)

    def getDirectChildren(self):
        return [node for node in self.nested_nodes if not isinstance(node, RegionNode)]

    def computeExtent(self):
        """Recursively derive start/end from all nested computation nodes."""
        starts, ends = [], []
        for node in self.nested_nodes:
            if isinstance(node, RegionNode):
                node.computeExtent()
                if node.region_start is not None: starts.append(node.region_start)
                if node.region_end   is not None: ends.append(node.region_end)
            elif isinstance(node, ComputationNode):
                starts.append(node.start)
                ends.append(node.end)
            elif hasattr(node, 'timestamp') and node.timestamp is not None:
                starts.append(node.timestamp)
                ends.append(node.timestamp)
        self.region_start = min(starts) if starts else None
        self.region_end   = max(ends)   if ends   else None

    def toCytoscape(self):
        cy_data = super(RegionNode, self).toCytoscape()
        cy_data["region_depth"]  = self.region_depth
        cy_data["nesting_level"] = self.nesting_level
        cy_data["region_start"]  = self.region_start
        cy_data["region_end"]    = self.region_end
        return cy_data


class FusedNode(Node):

    def __init__(self, graph, label, total_order, region):
        super(FusedNode, self).__init__(graph, label, total_order, region)
        self.type = "fusion"
        graph.setLabelForID(label, self.id)

    def addNestedNode(self, node):
        self.nested_nodes.append(node)

    def toCytoscape(self):
        cy_data = super(FusedNode, self).toCytoscape()
        return cy_data

    def infoData(self):
        info_data = super(FusedNode, self).toCytoscape()
        return info_data


class DataNode(Node):

    def __init__(self, graph, label, total_order, region):
        super(DataNode, self).__init__(graph, label, total_order, region)
        self.backend = "ANY"

    def toCytoscape(self):
        cy_data = super(DataNode, self).toCytoscape()
        return cy_data


class AllocationNode(DataNode):

    def __init__(self, graph, json_data, total_order):
        graph.setLabelForID(json_data["label"], json_data["object_id"])
        if graph.settings.allocations:
            super(AllocationNode, self).__init__(graph, json_data["label"], total_order, json_data.get("region"))
            self.type = "allocation"
            self.timestamp = json_data.get("time")
            graph.writeToID(json_data["object_id"], self, "ALL")

    def toCytoscape(self):
        cy_data = super(AllocationNode, self).toCytoscape()
        cy_data["timestamp"] = getattr(self, "timestamp", None)
        return cy_data

    def infoData(self):
        info_data = super(AllocationNode, self).infoData()
        info_data["location"] = backendToMemSpace["CPU"]
        return info_data


class DeallocationNode(DataNode):

    def __init__(self, graph, json_data, total_order):
        if graph.settings.deallocations:
            super(DeallocationNode, self).__init__(graph, json_data["label"], total_order, json_data.get("region"))
            self.type = "deallocation"
            self.timestamp = json_data.get("time")
            self.backend = "ANY"
            AntiDepEdge(graph, json_data["label"], graph.accessorForID(json_data["object_id"]), self)

    def toCytoscape(self):
        cy_data = super(DeallocationNode, self).toCytoscape()
        cy_data["timestamp"] = getattr(self, "timestamp", None)
        return cy_data


class UpdateNode(DataNode):

    def __init__(self, graph, id, total_order, region, backend):
        label = graph.labelForID(id)
        super(UpdateNode, self).__init__(graph, label, total_order, region)
        self.type = "container_update"
        self.version = graph.newVersionNumberForID(id)
        self.backend = backend
        self.is_live = []
        graph.writeToID(id, self, backend)

    def toCytoscape(self):
        cy_data = super(UpdateNode, self).toCytoscape()
        cy_data["version"] = self.version
        return cy_data

    def infoData(self):
        info_data = super(UpdateNode, self).infoData()
        info_data["Version"] = self.version
        info_data["Location"] = backendToMemSpace[self.backend]
        info_data["internal"]["is_live"] = self.is_live
        return info_data


class ScalarNode(DataNode):

    def __init__(self, graph, label, total_order, region):
        super(ScalarNode, self).__init__(graph, label, total_order, region)
        self.type = "scalar"

    def fromEventData(graph, json_data, total_order, region=None):
        label = json_data["label"] + "[" + str(json_data["index"]) + "]"
        if graph.settings.scalars:
            node = ScalarNode(graph, label, total_order, region)
            ScalarEdge(graph, label, json_data["source"], node)
            graph.readFromID(json_data["source"], node, "ALL")
            graph.writeToID(json_data["object_id"], node, "ALL")
        else:
            graph.setLabelForID(label, json_data["object_id"])
            graph.writeToID(json_data["object_id"], graph.producerForID(json_data["source"]), "ALL")

    def fromID(graph, object_id, total_order, region):
        label = "scalar" # + str(object_id)
        node = ScalarNode(graph, label, total_order, region)
        graph.setLabelForID(label, object_id)
        graph.writeToID(object_id, node, "ALL")
        return node

    def toCytoscape(self):
        cy_data = super(ScalarNode, self).toCytoscape()
        return cy_data

    def infoData(self):
        info_data = super(ScalarNode, self).infoData()
        return info_data


class TransferNode(DataNode):
    def __init__(self, graph, json_data, total_order):

        if "internal" in json_data["label"].lower():
            return

        self.direction = json_data["direction"]
        object_id = json_data["object_id"]
        trace_backend = json_data["backend"]
        target_backend = "CPU" if self.direction == "device-to-host" else trace_backend
        source_backend = trace_backend if self.direction == "device-to-host" else "CPU"

        if graph.settings.transfers:
            super(TransferNode, self).__init__(graph, json_data["label"], total_order, json_data.get("region"))
            self.type = "transfer"
            self.start = json_data["start"]
            self.end = json_data["end"]
            self.duration = self.end - self.start
            self.timestamp = json_data.get("time")
            self.depth = graph.depthByLabel(json_data["label"]) + 1
            self.backend = source_backend
            ProxyEdge(graph, json_data["label"], object_id, self)
            graph.writeToID(object_id, self, target_backend)
        else:
            graph.transferIDMemspace(object_id, source_backend, target_backend)

    def toCytoscape(self):
        cy_data = super(TransferNode, self).toCytoscape()
        cy_data["direction"] = self.direction
        cy_data["timestamp"] = self.timestamp
        cy_data["start"] = self.start
        cy_data["end"] = self.end
        cy_data["duration"] = self.duration
        return cy_data

    def infoData(self):
        info_data = super(TransferNode, self).infoData()
        info_data["Direction"] = self.direction
        info_data["Duration"] = self.duration
        info_data["Backend"] = self.backend
        return info_data




class Edge:

    def __init__(self, graph, source_id, target):
        self.id = str(uuid.uuid4())
        self.label = ""
        self.target = target
        if isinstance(source_id, Node):
            self.source = source_id
        else:
            self.source = graph.producerForID(source_id, target.backend)

        self.is_critical_path = False
        self.iteration_count = 1

        if self.source and self.target:
            graph.addEdge(self)

    def toCytoscape(self):
        cy_data = {}
        cy_data["id"] = self.id
        cy_data["label"] = self.label
        cy_data["source"] = self.source.id
        cy_data["target"] = self.target.id
        cy_data["is_critical_path"] = self.is_critical_path
        cy_data["iteration_count"] = self.iteration_count
        return cy_data


class ElwiseEdge(Edge):

    def __init__(self, graph, label, source_id, target):
        super(ElwiseEdge, self).__init__(graph, source_id, target)
        self.label = label

    def toCytoscape(self):
        cy_data = super(ElwiseEdge, self).toCytoscape()
        cy_data["access_mode"] = "elwise"
        cy_data["type"] = "forward-dep"
        return cy_data


class ProxyEdge(Edge):

    def __init__(self, graph, label, source_id, target):
        super(ProxyEdge, self).__init__(graph, source_id, target)
        self.label = label

    def toCytoscape(self):
        cy_data = super(ProxyEdge, self).toCytoscape()
        cy_data["access_mode"] = "proxy"
        cy_data["type"] = "forward-dep"
        return cy_data


class ScalarEdge(Edge):

    def __init__(self, graph, label, source_id, target):
        super(ScalarEdge, self).__init__(graph, source_id, target)
        self.label = label

    def toCytoscape(self):
        cy_data = super(ScalarEdge, self).toCytoscape()
        cy_data["access_mode"] = "scalar"
        cy_data["type"] = "forward-dep"
        return cy_data


class AntiDepEdge(Edge):

    def __init__(self, graph, label, source_id, target):
        super(AntiDepEdge, self).__init__(graph, source_id, target)
        self.label = label

    def toCytoscape(self):
        cy_data = super(AntiDepEdge, self).toCytoscape()
        cy_data["access_mode"] = "proxy"
        cy_data["type"] = "anti-dep"
        return cy_data







app = Flask(__name__)
event_data = None
unique_rows = {}
keypath_iterations = {}
depths = {}
graph = None

# ── Examples ──────────────────────────────────────────────────────────────────

EXAMPLES_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples')
EXAMPLES_FILE = os.path.join(EXAMPLES_DIR, 'examples.json')

def load_examples():
    """Return the list of example descriptors from examples/examples.json,
    or an empty list if the file is absent or malformed."""
    try:
        with open(EXAMPLES_FILE, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []



def preprocess_file_paths(event_data):
    # Find longest common path prefix
    file_paths = []
    for event in event_data:
        if "file" in event and event["file"] != "":
            file_paths.append(event["file"])
    path_prefix = os.path.dirname(os.path.commonprefix(file_paths))
#    print("Common file path prefix: ", path_prefix)
    
    # Filter path fields to remove common prefix
    for event in event_data:
        if "file" in event and event["file"] != "":
            event["file"] = event["file"].removeprefix(path_prefix)


@app.route('/graph')
def request_graph():
    global event_data
    global graph

    request_time = time.time()
    graph = DirectedGraph()

    # parameters
    graph.settings.antideps = request.args.get('anti_deps') == "true"
    graph.settings.allocations = request.args.get('container_allocations') == "true"
    graph.settings.deallocations = request.args.get('container_deallocations') == "true"
    graph.settings.transfers = request.args.get('container_transfers') == "true"
    graph.settings.updates = request.args.get('data_as_edges') != "true"
    graph.settings.scalars = request.args.get('show_scalars') == "true"
    graph.settings.regions = request.args.get('show_regions') == "true"
    oalesce_edges = request.args.get('coalesce_region_deps') == "true"
    coalesce_iterations = request.args.get('collapse_iteration') == "true"
    fusion_analysis = request.args.get('fusion_analysis') == "true"
    
    preprocess_file_paths(event_data)

    cntr = 0
    for event in event_data:
        cntr += 1
        order = cntr * 100

        if event["type"] == "skeleton_call":
            CallNode(graph, event, order)
        elif event["type"] == "external":
            ExternalNode(graph, event, order)
        elif event["type"] == "allocation":
            AllocationNode(graph, event, order)
        elif event["type"] == "deallocation":
            DeallocationNode(graph, event, order)
        elif event["type"] == "element_access":
            ScalarNode.fromEventData(graph, event, order)
        elif event["type"] == "transfer":
            TransferNode(graph, event, order)
        elif event["type"] == "region":
            RegionNode(graph, event, order)
    
    graph.summary()
#    print("Commputing depths ...")
#    graph.computeDepths()
    print("Finding critical path ...")
    graph.findCriticalPath()
    print("Computing keypaths ...")
    graph.computeKeyPaths()
#    graph.computeLiveness()
#    print("Computing equivalence classes ...")
#    graph.computeEquivalenceClasses()

    fusion_hints = graph.findFusions() if fusion_analysis else []

    if coalesce_iterations:
        graph.coalesceIterations()

    # Compute region extents (start/end times) for Gantt timeline display.
    # Only root regions are called explicitly; computeExtent() recurses into nested ones.
    if graph.settings.regions:
        for node in graph.getAllNodes():
            if isinstance(node, RegionNode) and node.region is None:
                node.computeExtent()

    # Drop empty RegionNodes (no nested nodes) before serialisation.
    if graph.settings.regions:
        graph.pruneEmptyRegions()

    (nodes, edges) = graph.toCytoscape()

#    print(nodes)
#    print(edges)
#    print(fusion_hints)
    response_time = time.time()

    print(f"Graph building took {response_time - request_time} seconds.")

    return {
        "nodes": nodes,
        "edges": edges,
        "event_count": len(event_data),
        "fusion_hints" : fusion_hints,
        "request_time" : request_time,
        "response_time" : response_time
    }

# CG with 6 skeletons in iterative loop.
# 75 iterations take 35 seconds to render
# 100 iterations causes stack overflow in dagre layout engine



@app.route('/get_data')
def get_data():
    global graph

    node_id = request.args.get('id')
    info_data = graph.getNodeById(node_id).infoData()
    return info_data




    row = unique_rows[node_id]
    row["dag_depth"] = depths[node_id]

    if "duration" in row:
        iterations = keypath_iterations[row["keypath"]]

        durations = []
        i = 0

        total_duration = 0
        for iter in iterations:
            durations.append({"x": "2024-01-0" + str(i+2), "y": iter["duration"]})
            total_duration += iter["duration"]
            i += 1
        row["duration_average"] = total_duration / len(iterations)

        stddev = 0
        for iter in iterations:
            stddev += pow(row["duration_average"] - iter["duration"], 2)
        row["duration_stddev"] = math.sqrt(stddev / len(iterations))
        row["duration_stddev_rel"] = str(row["duration_stddev"] / row["duration_average"] * 100) + " %"

        row["duration_dev"] = row["duration_average"] - iter["duration"]
        row["duration_dev_rel"] = str(row["duration_dev"] / row["duration_average"] * 100) + " %"

        row["durations"] = durations

    return row










# Create the correct container format
@app.route('/timeline')
def timeline():
    global event_data
    global unique_rows
    global keypath_iterations

    nodes = []
    groups = []
    edges = []
    latest_of_label = {}
    previous_of_label = {}
    singleton_of_keypath = {}
    unique_rows = {}
    keypath_iterations = {}
    event_count = len(event_data)

    # parameters
    render_anti_deps = request.args.get('anti_deps') == "true"
    render_container_allocations = request.args.get('container_allocations') == "true"
    data_as_edges = request.args.get('data_as_edges') == "true"
    show_regions = True #request.args.get('show_regions') == "true"
    coalesce_region_deps = request.args.get('coalesce_region_deps') == "true"
    coalesce_iterations = request.args.get('collapse_iteration') == "true"

    def find_keypath(elem):
        path = elem["label"]
        if "region" in elem:
            return find_keypath(unique_rows[latest_of_label[elem["region"]]]) + "->" + path
        else:
            return path

    for row in event_data:
        label = row["label"]
        row["version"] = unique_rows[latest_of_label[label]]["version"] + 1 if label in latest_of_label else 0

        keypath = find_keypath(row)
        row["keypath"] = keypath
        if keypath not in keypath_iterations:
            keypath_iterations[keypath] = [row]
        else:
            keypath_iterations[keypath].append(row)

        row["iterations"] = len(keypath_iterations[keypath])

        if coalesce_iterations:
            first = False
            if keypath not in singleton_of_keypath:
                singleton_of_keypath[keypath] = row
                first = True
                node_id = str(uuid.uuid4())
                unique_rows[node_id] = row
                if label in latest_of_label: previous_of_label[label] = latest_of_label[label]
                latest_of_label[label] = node_id
            else:
                singleton_of_keypath[keypath]["iterations"] = row["iterations"]
                node_id = latest_of_label[label]
                continue
        else:
            node_id = str(uuid.uuid4())
            unique_rows[node_id] = row
            if label in latest_of_label: previous_of_label[label] = latest_of_label[label]
            latest_of_label[label] = node_id




        if row["type"] == "skeleton_call" or row["type"] == "external":
            duration = row["end"] - row["start"]
            row["duration"] = duration
            node_data = {'id': node_id, 'content': label, 'type': 'background', 'node_type' : row["type"], 'duration': duration, 'start' : row['start'], 'end': row['end']}
            if "file" in row: node_data['file'] = row["file"]
            if "line" in row: node_data['line'] = row["line"]
            if "elements" in row: node_data['elements'] = row["elements"]
            if "region" in row and show_regions:
                node_data["group"] = latest_of_label[row["region"]]
            nodes.append(node_data)

            def handle_input(input_label, access_mode):
                if not (unique_rows[latest_of_label[input_label]]["type"] == "allocation" and not render_container_allocations):

                    source = latest_of_label[input_label]
                    target = node_id

                    if coalesce_region_deps and "region" not in unique_rows[source] and "region" in unique_rows[target]:
                        target = latest_of_label[unique_rows[target]["region"]]

                    elif coalesce_region_deps and "region" in unique_rows[source] and "region" not in unique_rows[target]:
                        source = latest_of_label[unique_rows[source]["region"]]

                    elif coalesce_region_deps and "region" in unique_rows[source] and "region" in unique_rows[target]:
                        if unique_rows[latest_of_label[unique_rows[source]["region"]]]["region_depth"] == unique_rows[latest_of_label[unique_rows[target]["region"]]]["region_depth"]:
                            if unique_rows[source]["region"] != unique_rows[target]["region"]:
                                source = latest_of_label[unique_rows[source]["region"]]
                                target = latest_of_label[unique_rows[target]["region"]]

                            elif unique_rows[source]["iterations"] != unique_rows[target]["iterations"]:
                                source = previous_of_label[unique_rows[source]["region"]]
                                target = latest_of_label[unique_rows[target]["region"]]

                #    edges.append({'data': {'source': source, 'target': target, 'type': 'forward-dep', 'access_mode': access_mode}})

            for input_label in row["elwise_inputs"]: handle_input(input_label, "elwise")
            if "proxy_inputs" in row:
                for input_label in row["proxy_inputs"]:  handle_input(input_label, "proxy")


            for output_label in row['outputs']:

                if data_as_edges:
                    latest_of_label[output_label] = node_id

                else:

                    old_version = latest_of_label[output_label] if output_label in latest_of_label else None

                    new_output = str(uuid.uuid4())
                    unique_rows[new_output] = copy.deepcopy(unique_rows[latest_of_label[output_label]])
                    unique_rows[new_output]["type"] = "container_update"
                    unique_rows[new_output]["keypath"] = (find_keypath(unique_rows[latest_of_label[row["region"]]]) + "->" if "region" in row else "") + unique_rows[new_output]["label"]
                    unique_rows[new_output]["iterations"] = row["iterations"]
                    unique_rows[new_output]["version"] += 1
                    if "region" in row and show_regions:
                        unique_rows[new_output]["region"] = row["region"]
                    latest_of_label[output_label] = new_output
                    new_node_data = {'id': new_output, 'label': output_label, 'type' : 'container_update'}

                    if "region" in row and show_regions:
                        new_node_data["parent"] = latest_of_label[row["region"]]

                #    nodes.append({'data' : new_node_data})
                #    edges.append({'data': {'source': node_id, 'target': new_output, 'type': 'forward-dep', 'access_mode': 'elwise'}})


                # Identify anti-dependences
                if render_anti_deps and old_version and output_label not in row["inputs"]\
                    and not (unique_rows[old_version]["type"] == "allocation" and not render_container_allocations):
                #    edges.append({'data': {'source': old_version, 'target': new_output, 'type': 'anti-dep'}})
                    pass


        elif row["type"] == "allocation":
            if render_container_allocations:
                node_data = {'id': node_id, 'label': label, 'type' : row["type"]}
                if "region" in row and show_regions:
                    node_data["parent"] = latest_of_label[row["region"]]
            #    nodes.append({'data' : node_data})


        elif row["type"] == "region" and show_regions:
            node_data = {'id': node_id, 'content': label, 'node_type' : row["type"], 'region_depth': row["region_depth"]}
            if "region" in row:
                node_data["parent"] = latest_of_label[row["region"]]
            groups.append(node_data)


    return { "nodes": nodes, "groups": groups, "edges": edges, "event_count": event_count }











# Render the first page to upload files
@app.route('/')
def csv():
    return render_template('files.html', examples=load_examples())


# Being able to upload the CSV file and save variables as needed
@app.route('/upload', methods=['POST'])
def upload():
    global event_data
    if request.method == 'POST':
        trace_filename = None
        json_file = request.files.get('json_file')
        if json_file:
            trace_filename = os.path.basename(json_file.filename.replace('\\', '/'))
            event_data = json.load(json_file)

        trace_json = json.dumps(event_data, indent=2) if event_data else None

        cpp_files = {}
        for f in request.files.getlist('cpp_files'):
            if f and f.filename:
                basename = os.path.basename(f.filename.replace('\\', '/'))
                cpp_files[basename] = f.read().decode('utf-8')

    return main_page(cpp_files=cpp_files, event_data=event_data, trace_json=trace_json, trace_filename=trace_filename)


# Load a built-in example by index into the global event_data and render main.html.
@app.route('/load_example')
def load_example():
    global event_data
    index = request.args.get('index', type=int, default=0)
    examples = load_examples()
    if index < 0 or index >= len(examples):
        return "Example not found", 404

    ex = examples[index]
    trace_dict = ex['trace']                               # dict: {backend: rel_path}
    backend = request.args.get('backend', default=next(iter(trace_dict)))
    if backend not in trace_dict:
        return "Backend not available for this example", 404

    trace_path = os.path.join(EXAMPLES_DIR, trace_dict[backend])
    with open(trace_path, encoding='utf-8') as f:
        event_data = json.load(f)

    trace_filename = os.path.basename(trace_path)
    trace_json_str = json.dumps(event_data, indent=2)

    cpp_files = {}
    for rel_path in ex.get('cpp_files', []):
        full_path = os.path.join(EXAMPLES_DIR, rel_path)
        if os.path.exists(full_path):
            cpp_files[os.path.basename(full_path)] = open(full_path, encoding='utf-8').read()

    return main_page(
        cpp_files=cpp_files,
        event_data=event_data,
        trace_json=trace_json_str,
        trace_filename=trace_filename,
    )


# Route to the main-page of the website
@app.route('/main')
def main_page(cpp_files=None, event_data=None, trace_json=None, trace_filename=None):
    return render_template(
        'main.html',
        data=event_data,
        cpp_files=cpp_files or {},
        trace_json=trace_json,
        trace_filename=trace_filename,
        examples=load_examples(),
    )

if __name__ == '__main__':
#    pid = os.fork()
#    if pid != 0:
    app.run(host = "localhost", port = 5001, debug = True)
#    else:
    #    time.sleep(0.3)
    #    webbrowser.open('localhost:5001')
