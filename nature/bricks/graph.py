import networkx as nx
import random

from tools import screenshot_graph, safe_sample, log

# appearance
STYLE = "filled"
# initial
N_INITIAL_BOXES = 4
# evolution
MIN_MUTATIONS, MAX_MUTATIONS = 1, 2
MUTATION_OPTIONS = ["insert_motifs"]
# regulons ... dat imprecise probability
MIN_LAYERS, MAX_LAYERS = 1, 3
MIN_NODES, MAX_NODES = 1, 3
MIN_P_INSERT, MAX_P_INSERT = 0.5, 0.5


def add_node(G, id, color, shape, node_type, out=None, spec=None):
    G.add_node(
        id, label=id,
        style=STYLE, color=color, shape=shape,
        node_type=node_type, out=out, spec=spec)


class Graph:
    def __init__(self, agent, in_specs=None, out_specs=None):
        super(Graph, self).__init__()
        self.agent = agent
        self.get_initial_graph()
        if in_specs and out_specs:
            self.get_task_graph(in_specs, out_specs)
        else:
            self.get_internal_graph()
            self.evolve_initial_graph()
        screenshot_graph(self.G, ".", "G")

    def get_initial_graph(self):
        self.G = nx.MultiDiGraph()
        add_node(self.G, "source", "gold", "cylinder", "source")
        add_node(self.G, "sink", "gold", "cylinder", "sink")

    def get_task_graph(self, in_specs, out_specs):
        add_node(self.G, 0, "black", "square", "brick")
        for n, in_spec in enumerate(in_specs):
            in_key = f"input_{n}"
            add_node(self.G, in_key, "blue", "circle", "input", spec=in_spec)
            self.G.add_edge("source", in_key)
            self.G.add_edge(in_key, 0)
        for n, out_spec in enumerate(out_specs):
            out_key = f"out_{n}"
            add_node(self.G, out_key, "red", "triangle", "out", spec=out_spec)
            self.G.add_edge(out_key, "sink")
            self.G.add_edge(0, out_key)

    def get_internal_graph(self):
        """Create a graph connecting inputs to outs with a black box."""
        for box_number in range(N_INITIAL_BOXES):
            add_node(self.G, box_number, "black", "square", "brick")
        add_node(self.G, "input", "blue", "circle", "input")
        self.G.add_edge("source", "input")
        [self.G.add_edge("input", box) for box in range(N_INITIAL_BOXES)]
        add_node(self.G, "out", "red", "triangle", "out")
        self.G.add_edge("out", "sink")
        [self.G.add_edge(box, "out") for box in range(N_INITIAL_BOXES)]

    def evolve_initial_graph(self):
        self.n_mutations = self.pull_numbers("n_mutations",
                                             MIN_MUTATIONS, MAX_MUTATIONS)
        for mutation_number in range(self.n_mutations):
            mutation_name = self.pull_choices("mutations", MUTATION_OPTIONS, 1)
            getattr(Graph, mutation_name)(self)

    def get_regulon(self, parent=None, layers=None):
        """Build a subgraph to replace a node."""
        num_layers = safe_sample(MIN_LAYERS, MAX_LAYERS)
        M = nx.MultiDiGraph()
        ids = []
        for layer_number in range(num_layers):
            n_nodes = safe_sample(MIN_NODES, MAX_NODES)
            # log(f"add layer {layer_number} with {n_nodes} nodes")
            ids_of_nodes_in_this_layer = []
            for node_number in range(n_nodes):
                if parent:
                    node_id = f"{parent}.{layer_number}.{node_number}"
                else:
                    node_id = f"{layer_number}.{node_number}"
                # log(f"add node {node_id}")
                add_node(M, node_id, "square", "black", "brick")
                ids_of_nodes_in_this_layer.append(node_id)
            ids.append(ids_of_nodes_in_this_layer)
        for predecessor_layer_number, predecessor_node_ids in enumerate(ids):
            for successor_layer_number, successor_node_ids in enumerate(ids):
                if predecessor_layer_number >= successor_layer_number:
                    continue
                for predecessor_node_id in predecessor_node_ids:
                    for successor_node_id in successor_node_ids:
                        # log(f"{predecessor_node_id}--->{successor_node_id}")
                        M.add_edge(predecessor_node_id, successor_node_id)
        return M, ids

    def insert_motif(self, id=None):
        """Replace a node with a subgraph."""
        id = random.choice(list(self.G.nodes())) if id is None else id
        if self.is_blacklisted(id):
            return
        M, new_ids = self.get_regulon(parent=id)
        self.G = nx.compose(self.G, M)
        # connect predecessors first layer
        predecessors = self.G.predecessors(id)
        successors = new_ids[0]
        for predecessor in predecessors:
            for successor in successors:
                # log(f"{predecessor}--->{successor}")
                self.G.add_edge(predecessor, successor)
        # connect last layer to successors
        predecessors = new_ids[-1]
        successors = list(self.G.successors(id))
        for predecessor in predecessors:
            for successor in successors:
                # log(f"{predecessor}--->{successor}")
                self.G.add_edge(predecessor, successor)
        self.G.remove_node(id)

    def insert_motifs(self):
        """with probability P, replace nodes with random feedforward regulons"""
        nodes = self.G.nodes(data=True)
        p_insert = safe_sample(MIN_P_INSERT, MAX_P_INSERT)
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
            if self.G.node[id]["shape"] != "square":
                return True
            else:
                return False
        except Exception as e:
            log(e, color="red")
            return False

    def count_boxes(self):
        return len(
            [node for node in list(self.G.nodes(data=True))
             if node[1]["shape"] == "square"])
