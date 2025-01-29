from flask import Flask, render_template, request, jsonify, json
import uuid, copy, math
import os, time, webbrowser



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

        self._nodeLabelIndex = {}
        self._nodeIdIndex = {}
        self._edgeIdIndex = {}

        self._latestProducerOfLabel = {}

        self.settings = GraphSettings()


    def addNode(self, node):
        self._nodes.append(node)
        self._nodeIdIndex[node.id] = node
        self._nodeLabelIndex[node.label] = node

        print(f"Added node with label {node.label} ID {node.id}")

    def addEdge(self, edge):
        sourceNode = self._nodeIdIndex.get(edge.source.id)
        targetNode = self._nodeIdIndex.get(edge.target.id)

        print(f"Adding edge from node with label {edge.source.label} to {edge.target.label}")

        sourceNode.addOutgoingEdge(edge)
        targetNode.addIncomingEdge(edge)

        self._edges.append(edge)
        self._edgeIdIndex[edge.id] = edge

    def getNodeById(self, nid):
        return self._nodeIdIndex[nid]

    def getNodeByLabel(self, label):
        print(f"Get node with label {label}")
        return self._nodeLabelIndex[label]

    def depthByLabel(self, label):
        try:
            return self._nodeLabelIndex[label].depth
        except KeyError:
            return 0

    def getEdgeById(self, eid):
        return self._edgeIdIndex[eid]

    def getRootNodes(self):
        return [node for node in self._nodes if len(node.getIncomingEdges()) == 0]

    def getLeafNodes(self):
        return [node for node in self._nodes if len(node.getOutgoingEdges()) == 0]

    def getGlobalRegionNodes(self):
        return [node for node in self._nodes if not node.region]

    # Graph builder

    def writeToLabel(self, label, node):
        self._latestProducerOfLabel[label] = node

    def producerForLabel(self, label):
        try:
            return self._latestProducerOfLabel[label]
        except KeyError:
            return None



    def visitNodes(self, f):
        pass


    # Processing and analysis
    def computeKeyPaths(self):

        def helper(node, keypath):
            mykey = node.label
            if isinstance(node, ComputationNode):
                mykey += "["
                for label in node.elwise_inputs:
                    mykey += label + "+"
                for label in node.proxy_inputs:
                    mykey += label + "+"
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
            print(f"Visiting {node.label} of ID {node.id} at depth {current_depth}")
            node.depth = max(node.depth, current_depth)
            for child in node.getChildNodes():
                helper(child, current_depth + 1)


        for node in self._nodes:
            node.depth = 0

        for node in self.getRootNodes():
            helper(node, 0)


    def findCriticalPath(self):
        deepest_node = None
        for node in self._nodes:
            if node.type == "region":
                continue
            if not deepest_node or node.depth > deepest_node.depth:
                deepest_node = node

        print(f"Deepest node: {deepest_node.label} ID: {deepest_node.id}")

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
                print(f"Critical path: {current_node.label} ID: {current_node.id}")
            else:
                break

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


                for edge in node.getIncomingEdges():
                #    if edge in self._edges:
                #        self._edges.remove(edge)
                    edge.target = canonicals[edge.target.keypath]


                for edge in node.getOutgoingEdges():
                #    if edge in self._edges:
                #        self._edges.remove(edge)
                    edge.source = canonicals[edge.source.keypath]


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

    def __init__(self, graph, label, region=None):
        self.id = str(uuid.uuid4())
        self.label = label
        self.graph = graph
        self._incoming = []
        self._outgoing = []
        self.depth = 0
        self.keypath = ""

        self.is_critical_path = False
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
        cy_data["type"] = self.type
        cy_data["is_critical_path"] = self.is_critical_path
        cy_data["dag_depth"] = self.depth
        if self.region:
            cy_data["parent"] = self.region.id
            cy_data["region_depth"] = self.nesting_level
        return cy_data

    def infoData(self):
        info_data = {}
        info_data["type"] = self.type
        info_data["is_critical_path"] = self.is_critical_path
        info_data["dag_depth"] = self.depth
        info_data["keypath"] = self.keypath
        if self.region:
            info_data["parent"] = self.region.id
            info_data["region_depth"] = self.nesting_level
        return info_data


class ComputationNode(Node):

    def __init__(self, graph, json_data):
        super(ComputationNode, self).__init__(graph, json_data["label"], json_data.get("region"))

        self.start = json_data["start"]
        self.end = json_data["end"]
        self.duration = self.end - self.start

        self.file = json_data.get("file")
        self.line = json_data.get("line")

        self.elwise_inputs = json_data["elwise_inputs"]
        self.proxy_inputs = json_data["proxy_inputs"]

        for label in json_data["elwise_inputs"]:
            ElwiseEdge(graph, label, self)

        for label in json_data["proxy_inputs"]:
            ProxyEdge(graph, label, self)

        for label in json_data['outputs']:
            if graph.settings.antideps:
                AntiDepEdge(graph, label, self)
            if graph.settings.updates:
                ElwiseEdge(graph, self, UpdateNode(graph, label, self.region))
            else:
                graph.writeToLabel(label, self)

    def toCytoscape(self):
        cy_data = super(ComputationNode, self).toCytoscape()
        cy_data["start"] = self.start
        cy_data["end"] = self.end
        cy_data["duration"] = self.duration
        cy_data["file"] = self.file
        cy_data["line"] = self.line
        return cy_data

    def infoData(self):
        info_data = super(ComputationNode, self).infoData()
        info_data["duration"] = self.duration
        info_data["file"] = self.file
        info_data["line"] = self.line
        return info_data


class CallNode(ComputationNode):

    def __init__(self, graph, json_data):
        super(CallNode, self).__init__(graph, json_data)
        self.type = "skeleton_call"
        self.pattern = json_data["pattern"]
        self.elements = json_data["elements"]

    def toCytoscape(self):
        cy_data = super(CallNode, self).toCytoscape()
        cy_data["pattern"] = self.pattern
        cy_data["elements"] = self.elements
        return cy_data


class ExternalNode(ComputationNode):

    def __init__(self, graph, json_data):
        super(ExternalNode, self).__init__(graph, json_data)
        self.type = "external"

    def toCytoscape(self):
        cy_data = super(ExternalNode, self).toCytoscape()
        return cy_data


class RegionNode(Node):

    def __init__(self, graph, json_data):
        if graph.settings.regions:
            super(RegionNode, self).__init__(graph, json_data["label"], json_data.get("region"))
            self.type = "region"
            self.region_depth = json_data["region_depth"]
            self.nested_nodes = []
            graph.addNode(self)

    def addNestedNode(self, node):
        self.nested_nodes.append(node)

    def toCytoscape(self):
        cy_data = super(RegionNode, self).toCytoscape()
        cy_data["region_depth"] = self.region_depth
        return cy_data



class DataNode(Node):

    def __init__(self, graph, label, region):
        super(DataNode, self).__init__(graph, label, region)

    def toCytoscape(self):
        cy_data = super(DataNode, self).toCytoscape()
        return cy_data


class AllocationNode(DataNode):

    def __init__(self, graph, json_data):
        if graph.settings.allocations:
            super(AllocationNode, self).__init__(graph, json_data["label"], json_data.get("region"))
            self.type = "allocation"
            graph.writeToLabel(json_data["label"], self)

    def toCytoscape(self):
        cy_data = super(AllocationNode, self).toCytoscape()
        return cy_data


class DeallocationNode(DataNode):

    def __init__(self, graph, json_data):
        if graph.settings.deallocations:
            super(DeallocationNode, self).__init__(graph, json_data["label"], json_data.get("region"))
            self.type = "deallocation"
            ProxyEdge(graph, json_data["label"], self)

    def toCytoscape(self):
        cy_data = super(DeallocationNode, self).toCytoscape()
        return cy_data


class UpdateNode(DataNode):
    def __init__(self, graph, label, region):
        super(UpdateNode, self).__init__(graph, label, region)
        self.type = "container_update"
        graph.writeToLabel(label, self)

    def toCytoscape(self):
        cy_data = super(UpdateNode, self).toCytoscape()
        return cy_data


class TransferNode(DataNode):
    def __init__(self, graph, json_data):
        if graph.settings.transfers:
            super(TransferNode, self).__init__(graph, json_data["label"], json_data.get("region"))
            ProxyEdge(graph, json_data["label"], self)
            graph.writeToLabel(json_data["label"], self)
            self.type = "transfer"
            self.direction = json_data["direction"]
            self.depth = graph.depthByLabel(json_data["label"]) + 1

    def toCytoscape(self):
        cy_data = super(TransferNode, self).toCytoscape()
        cy_data["direction"] = self.direction
        return cy_data





class Edge:

    def __init__(self, graph, source, target):
        self.id = str(uuid.uuid4())
        self.label = "dummy"
        self.target = target
        if isinstance(source, Node):
            self.source = source
        else:
            self.source = graph.producerForLabel(source)

        self.is_critical_path = False


        if self.source and self.target:
            print(f"Edge from {self.source.id} to {self.target.id}")
            graph.addEdge(self)

    def toCytoscape(self):
        cy_data = {}
        cy_data["id"] = self.id
        cy_data["label"] = self.label
        cy_data["source"] = self.source.id
        cy_data["target"] = self.target.id
        cy_data["is_critical_path"] = self.is_critical_path
        return cy_data


class ElwiseEdge(Edge):

    def __init__(self, graph, source_label, target):
        super(ElwiseEdge, self).__init__(graph, source_label, target)

    def toCytoscape(self):
        cy_data = super(ElwiseEdge, self).toCytoscape()
        cy_data["access_mode"] = "elwise"
        cy_data["type"] = "forward-dep"
        return cy_data


class ProxyEdge(Edge):

    def __init__(self, graph, source_label, target):
        super(ProxyEdge, self).__init__(graph, source_label, target)

    def toCytoscape(self):
        cy_data = super(ProxyEdge, self).toCytoscape()
        cy_data["access_mode"] = "proxy"
        cy_data["type"] = "forward-dep"
        return cy_data


class AntiDepEdge(Edge):

    def __init__(self, graph, source_label, target):
        super(AntiDepEdge, self).__init__(graph, source_label, target)

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

# Render the first page to upload files
@app.route('/')
def csv():
    return render_template('files.html')


# Being able to upload the CSV file and save variables as needed
@app.route('/upload', methods=['POST'])
def upload():
    global event_data
    if request.method == 'POST':
        json_file = request.files.get('json_file')
        if json_file:
            event_data = json.load(json_file)

        cpp_file = request.files.get('cpp_file')
        if cpp_file:
            cpp_code = cpp_file.read().decode('utf-8')

    return main_page(cpp_code=cpp_code, event_data=event_data)











@app.route('/graph')
def request_graph():
    global event_data
    global graph
    graph = DirectedGraph()

    # parameters
    graph.settings.antideps = request.args.get('anti_deps') == "true"
    graph.settings.allocations = request.args.get('container_allocations') == "true"
    graph.settings.deallocations = request.args.get('container_deallocations') == "true"
    graph.settings.transfers = request.args.get('container_transfers') == "true"
    graph.settings.updates = request.args.get('data_as_edges') != "true"
    graph.settings.regions = request.args.get('show_regions') == "true"
    oalesce_edges = request.args.get('coalesce_region_deps') == "true"
    coalesce_iterations = request.args.get('collapse_iteration') == "true"

    for event in event_data:
        if event["type"] == "skeleton_call":
            CallNode(graph, event)
        elif event["type"] == "external":
            ExternalNode(graph, event)
        elif event["type"] == "allocation":
            AllocationNode(graph, event)
        elif event["type"] == "deallocation":
            DeallocationNode(graph, event)
        elif event["type"] == "transfer":
            TransferNode(graph, event)
        elif event["type"] == "region":
            RegionNode(graph, event)

    graph.summary()
    graph.computeDepths()
    graph.findCriticalPath()
    graph.computeKeyPaths()

    if coalesce_iterations:
        graph.coalesceIterations()


    (nodes, edges) = graph.toCytoscape()

#    print(nodes)
#    print(edges)

    return { "nodes": nodes, "edges": edges, "event_count": len(event_data), "fusion_hints" : [] }




@app.route('/get_data')
def get_data():
    global graph

    node_id = request.args.get('id')

    info_data = graph.getNodeById(node_id).infoData()
    print(info_data)
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
@app.route('/graph_old')
def request_graph_old():
    global event_data
    global unique_rows
    global keypath_iterations
    global depths

    nodes = []
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
    render_container_transfers = request.args.get('container_transfers') == "true"
    data_as_edges = request.args.get('data_as_edges') == "true"
    show_regions = request.args.get('show_regions') == "true"
    coalesce_region_deps = request.args.get('coalesce_region_deps') == "true"
    coalesce_iterations = request.args.get('collapse_iteration') == "true"

    def find_keypath(elem):
        path = elem["label"]
        if "region" in elem:
            return find_keypath(unique_rows[latest_of_label[elem["region"]]]) + "->" + path
        else:
            return path

    for row in event_data:
        if row["type"] == "transfer" and not render_container_transfers:
            continue

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

        depths[node_id] = 0


        if row["type"] == "skeleton_call" or row["type"] == "external":
            duration = row["end"] - row["start"]
            row["duration"] = duration
            node_data = {'id': node_id, 'label': label, 'type' : row["type"], 'duration': duration}
            if "pattern" in row: node_data['pattern'] = row["pattern"]
            if "file" in row: node_data['file'] = row["file"]
            if "line" in row: node_data['line'] = row["line"]
            if "elements" in row: node_data['elements'] = row["elements"]
            if "region" in row and show_regions:
                node_data["parent"] = latest_of_label[row["region"]]


            def handle_input(input_label, access_mode):
                if not (unique_rows[latest_of_label[input_label]]["type"] == "allocation" and not render_container_allocations)\
                    and not (unique_rows[latest_of_label[input_label]]["type"] == "transfer" and not render_container_transfers):

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

                    depths[node_id] = max(depths[node_id], depths[source] + 1)
                    depth_delta = depths[target] - depths[source]
                    edges.append({'data': {'source': source, 'target': target, 'type': 'forward-dep', 'access_mode': access_mode,
                    'depth_delta' : depth_delta}})

            if "elwise_inputs" in row:
                for input_label in row["elwise_inputs"]:
                    handle_input(input_label, "elwise")

            if "proxy_inputs" in row:
                for input_label in row["proxy_inputs"]:
                    handle_input(input_label, "proxy")


            for output_label in row['outputs']:

                if data_as_edges:
                    latest_of_label[output_label] = node_id

                else:

                    old_version = latest_of_label[output_label] if output_label in latest_of_label else None

                    new_output = str(uuid.uuid4())
                    depths[new_output] = depths[node_id] + 1
                    unique_rows[new_output] = copy.deepcopy(unique_rows[latest_of_label[output_label]])
                    unique_rows[new_output]["type"] = "container_update"
                    unique_rows[new_output]["keypath"] = (find_keypath(unique_rows[latest_of_label[row["region"]]]) + "->" if "region" in row else "") + unique_rows[new_output]["label"]
                    unique_rows[new_output]["iterations"] = row["iterations"]
                    unique_rows[new_output]["version"] += 1
                    if "region" in row and show_regions:
                        unique_rows[new_output]["region"] = row["region"]
                    latest_of_label[output_label] = new_output
                    new_node_data = {'id': new_output, 'label': output_label, 'type' : 'container_update', 'version': unique_rows[new_output]["version"], 'dag_depth' : depths[new_output]}

                    if "region" in row and show_regions:
                        new_node_data["parent"] = latest_of_label[row["region"]]

                    depth_delta = depths[new_output] - depths[node_id]

                    nodes.append({'data' : new_node_data})
                    edges.append({'data': {'source': node_id, 'target': new_output, 'type': 'forward-dep', 'access_mode': 'elwise',
                    'depth_delta' : depth_delta}})


                # Identify anti-dependences
                if render_anti_deps and old_version and not (output_label in row["elwise_inputs"] or output_label in row["proxy_inputs"])\
                    and not (unique_rows[old_version]["type"] == "allocation" and not render_container_allocations):#\
                #    and not (unique_rows[old_version]["type"] == "transfer" and not render_container_transfers):
                    edges.append({'data': {'source': old_version, 'target': new_output, 'type': 'anti-dep'}})

            node_data['dag_depth'] = depths[node_id]
            nodes.append({'data' : node_data})

        elif row["type"] == "allocation" or row["type"] == "deallocation":
            if render_container_allocations:
                node_data = {'id': node_id, 'label': label, 'type' : row["type"]}
                if "region" in row and show_regions:
                    node_data["parent"] = latest_of_label[row["region"]]

                if row["type"] == "deallocation" and label in previous_of_label:
                    source = previous_of_label[label]
                    depths[node_id] = max(depths[node_id], depths[source] + 1)
                    edges.append({'data': {'source': source, 'target': node_id, 'type': 'forward-dep', 'access_mode': 'proxy'}})

                node_data['dag_depth'] = depths[node_id]
                nodes.append({'data' : node_data})

        elif row["type"] == "transfer":
            if render_container_transfers:
                node_data = {'id': node_id, 'label': label, 'type' : row["type"], 'direction' : row["direction"]}
                if "region" in row and show_regions:
                    node_data["parent"] = latest_of_label[row["region"]]

            #    if row["type"] == "deallocation" and label in previous_of_label:
                source = previous_of_label[label]
                if not (unique_rows[source]["type"] == "allocation" and not render_container_allocations):

                    depths[node_id] = max(depths[node_id], depths[source] + 1)
                    edges.append({'data': {'source': source, 'target': node_id, 'type': 'forward-dep', 'access_mode': 'proxy'}})

                node_data['dag_depth'] = depths[node_id]
                nodes.append({'data' : node_data})


        elif row["type"] == "region":
            if show_regions:
                node_data = {'id': node_id, 'label': label, 'type' : row["type"], 'region_depth': row["region_depth"]}
                if "region" in row:
                    node_data["parent"] = latest_of_label[row["region"]]
                nodes.append({'data' : node_data})

    # Find critical path
    for e in edges:
        e["data"]["is_critical_path"] = False

    largest_depth = 0
    largest_node_id = None
    for n in nodes:
        if n["data"]["type"] == "region": continue
        depth = n["data"]["dag_depth"]
        if depth > largest_depth:
            largest_depth = depth
            largest_node_id = n["data"]["id"]

    while True:
        id = largest_node_id
        max_parent_depth = -1
        max_parent_node = None
        max_edge = None
        for e in edges:
            if e["data"]["target"] == id:
                for n in nodes:
                    if n["data"]["id"] == e["data"]["source"]:
                        depth = n["data"]["dag_depth"]
                        if depth > max_parent_depth:
                            max_parent_depth = depth
                            max_parent_node = n
                            max_edge = e
                            largest_node_id = n["data"]["id"]

        if max_edge:
            max_edge["data"]["is_critical_path"] = True
        else:
            break
    # End find critical path



    # Start find hints
    fusion_hints = []
    for n in nodes:
        id = n["data"]["id"]
        chain = []
        if n["data"]["type"] == "skeleton_call" and n["data"]["pattern"] == "Map":
            for e in edges:
                if e["data"]["target"] == id:
                    for p in nodes:
                        if p["data"]["type"] == "skeleton_call" and p["data"]["id"] == e["data"]["source"]:
                            fusion_hints.append([id, p["data"]["id"]])

    # End find hints


    return { "nodes": nodes, "edges": edges, "event_count": event_count, "fusion_hints" : fusion_hints }























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










 # Route to the main-page of the website
@app.route('/main')
def main_page(cpp_code=None, event_data=None):
    return render_template('main.html', data=event_data, cpp_code=cpp_code)

if __name__ == '__main__':
#    pid = os.fork()
#    if pid != 0:
    app.run(host = "localhost", port = 5001, debug = True)
#    else:
    #    time.sleep(0.3)
    #    webbrowser.open('localhost:5001')
