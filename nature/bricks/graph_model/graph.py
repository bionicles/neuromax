import networkx as nx
import random

from tools.screenshot_graph import screenshot_graph
from tools.safe_sample import safe_sample
from tools.log import log

# appearance
STYLE = "filled"
# initial
MIN_INITIAL_BOXES, MAX_INITIAL_BOXES = 1, 2
# evolution
MIN_MUTATIONS, MAX_MUTATIONS = 1, 2
MUTATION_OPTIONS = ["insert_motifs"]
MIN_MIN_P_INSERT, MAX_MIN_P_INSERT = 0, 0.1
MIN_MAX_P_INSERT, MAX_MAX_P_INSERT = 0.11, 1
# regulons
MIN_MIN_LAYERS, MAX_MIN_LAYERS, MIN_MAX_LAYERS, MAX_MAX_LAYERS = 1, 2, 3, 4
MIN_MIN_NODES, MAX_MIN_NODES, MIN_MAX_NODES, MAX_MAX_NODES = 1, 2, 3, 4


class Graph:
    def __init__(self, agent, brick_id):
        log("Graph.__init__")
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        self.brick_id = brick_id + "_graph"
        self.code_spec = agent.code_spec
        self.agent = agent
        self.get_graph()
        screenshot_graph(self.G, ".", "G")

    def get_graph(self):
        self.min_p_insert = self.pull_numbers(
            "min_p_insert", MIN_MIN_P_INSERT, MAX_MIN_P_INSERT)
        self.max_p_insert = self.pull_numbers(
            "max_p_insert", MIN_MAX_P_INSERT, MAX_MAX_P_INSERT)
        self.min_layers = self.pull_numbers(
            "min_layers", MIN_MIN_LAYERS, MAX_MIN_LAYERS)
        self.max_layers = self.pull_numbers(
            "max_layers", MIN_MAX_LAYERS, MAX_MAX_LAYERS)
        self.min_nodes = self.pull_numbers(
            "min_nodes", MIN_MIN_NODES, MAX_MIN_NODES)
        self.max_nodes = self.pull_numbers(
            "max_nodes", MIN_MAX_NODES, MAX_MAX_NODES)
        self.get_initial_graph()
        self.evolve_initial_graph()

    def get_initial_graph(self):
        """Create a graph connecting inputs to outputs with a black box."""
        self.G = nx.MultiDiGraph()
        self.G.add_node("source", label="SOURCE",
                        style=STYLE, color="gold", shape="cylinder",
                        node_type="source", output=None)
        self.G.add_node("sink", label="SINK",
                        style=STYLE, color="gold", shape="cylinder",
                        node_type="sink", output=None)
        n_initial_boxes = self.pull_numbers("n_initial_boxes",
                                            MIN_INITIAL_BOXES,
                                            MAX_INITIAL_BOXES)
        self.n_initial_boxes = n_initial_boxes
        for box_number in range(n_initial_boxes):
            self.G.add_node(box_number, label=box_number,
                            style=STYLE, color="black", shape="square",
                            node_type="brick", output=None)
        for input_number in range(self.n_in):
            input_key = f"input_{input_number}"
            self.G.add_node(input_key, label=input_key,
                            style=STYLE, color="blue", shape="circle",
                            node_type="input", output=None)
            self.G.add_edge("source", input_key)
            [self.G.add_edge(input_key, box_number)
             for box_number in range(n_initial_boxes)]
        for output_number in range(self.n_in):
            output_key = f"output_{output_number}"
            self.G.add_node(output_key, label=output_key,
                            style=STYLE, color="red", shape="triangle",
                            node_type="output", output=None)
            self.G.add_edge(output_key, "sink")
            [self.G.add_edge(box_number, output_key)
             for box_number in range(n_initial_boxes)]

    def evolve_initial_graph(self):
        self.n_mutations = self.pull_numbers("n_mutations",
                                             MIN_MUTATIONS, MAX_MUTATIONS)
        for mutation_number in range(self.n_mutations):
            mutation_name = self.pull_choices("mutations", MUTATION_OPTIONS, 1)
            getattr(Graph, mutation_name)(self)

    def get_regulon(self, parent=None, layers=None):
        """Build a subgraph to replace a node."""
        num_layers = safe_sample(self.min_layers, self.max_layers)
        M = nx.MultiDiGraph()
        ids = []
        for layer_number in range(num_layers):
            n_nodes = safe_sample(self.min_nodes, self.max_nodes)
            log(f"add layer {layer_number} with {n_nodes} nodes")
            ids_of_nodes_in_this_layer = []
            for node_number in range(n_nodes):
                if parent:
                    node_id = f"{parent}.{layer_number}.{node_number}"
                else:
                    node_id = f"{layer_number}.{node_number}"
                log(f"add node {node_id}")
                M.add_node(node_id, label="",
                           style=STYLE, shape="square", color="black",
                           node_type="brick", output=None)
                ids_of_nodes_in_this_layer.append(node_id)
            ids.append(ids_of_nodes_in_this_layer)
        for predecessor_layer_number, predecessor_node_ids in enumerate(ids):
            for successor_layer_number, successor_node_ids in enumerate(ids):
                if predecessor_layer_number >= successor_layer_number:
                    continue
                for predecessor_node_id in predecessor_node_ids:
                    for successor_node_id in successor_node_ids:
                        log(f"{predecessor_node_id}--->{successor_node_id}")
                        M.add_edge(predecessor_node_id, successor_node_id)
        return M, ids

    def insert_motif(self, id=None):
        """Replace a node with a subgraph."""
        id = random.choice(list(self.G.nodes())) if id is None else id
        if self.is_blacklisted(id):
            return
        M, new_ids = self.get_regulon(parent=id)
        self.G = nx.compose(self.G, M)
        # connect the node's predecessors to all nodes in first layer
        predecessors = self.G.predecessors(id)
        successors = new_ids[0]
        for predecessor in predecessors:
            for successor in successors:
                log(f"{predecessor}--->{successor}")
                self.G.add_edge(predecessor, successor)
        # connect the last layer of the motif to the node's successors
        predecessors = new_ids[-1]
        successors = list(self.G.successors(id))
        for predecessor in predecessors:
            for successor in successors:
                log(f"{predecessor}--->{successor}")
                self.G.add_edge(predecessor, successor)
        self.G.remove_node(id)

    def insert_motifs(self):
        """with probability P, replace nodes with random feedforward regulons"""
        nodes = self.G.nodes(data=True)
        if self.count_boxes() < self.n_initial_boxes:
            p_insert = 1.
        else:
            p_insert = safe_sample(self.min_p_insert, self.max_p_insert)
        for node in nodes:
            try:
                if node[1]["shape"] is "square":
                    if random.random() < p_insert:
                        self.insert_motif(id=node[0])
            except Exception as e:
                log('exception in recurse', e)

    def is_blacklisted(self, id):
        log(f"is {id} blacklisted?")
        try:
            node = self.G.node[id]
            if node["shape"] != "square":
                log("yup!")
                return True
            else:
                log("nope!")
                return False
        except Exception as e:
            log("nope!", e)
            return False

    def count_boxes(self):
        return len([node for node in list(self.G.nodes(data=True))
                    if node[1]["shape"] == "square"])
