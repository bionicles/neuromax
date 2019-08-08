# graph_model.py - bion
# why?: learn to map N inputs to M outputs with graph GP
from tensorflow_addons import InstanceNormalization
import tensorflow as tf
import networkx as nx
import random

from bricks.transformer import Transformer
from bricks.kernel_conv import KConvSet
from bricks.get_mlp import get_mlp
from tools import log, safe_sample

K = tf.keras
L = K.layers

# initial
MIN_INITIAL_BOXES, MAX_INITIAL_BOXES = 1, 4
# evolution
MIN_MUTATIONS, MAX_MUTATIONS = 1, 4
MUTATION_OPTIONS = ["insert_motifs"]
MIN_MIN_P_INSERT, MAX_MIN_P_INSERT, MIN_MAX_P_INSERT, MAX_MAX_P_INSERT = 0, 0.49, 0.51, 1
# regulons
MIN_MIN_LAYERS, MAX_MIN_LAYERS, MIN_MAX_LAYERS, MAX_MAX_LAYERS = 1, 2, 3, 4
MIN_MIN_NODES, MAX_MIN_NODES, MIN_MAX_NODES, MAX_MAX_NODES = 1, 2, 3, 4
# bricks
ACTIVATION_OPTIONS = ["tanh"]
BRICK_OPTIONS = ["conv1d", "transformer", "dnc", "mlp", "k_conv"]


class GraphModel:
    """
    GraphModel maps [n_in x code_shape] -> [n_out x code_shape]
    because we need to be able to handle multiple inputs
    """

    def __init__(self, agent, n_in, code_shape, n_out):
        self.agent = agent
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        self.n_in, self.code_shape, self.n_out = n_in, code_shape, n_out
        self.get_graph()
        self.make_model()

    def call(self, inputs):
        return self.model(inputs)

    def get_graph(self):
        self.min_layers = self.pull_numbers("min_layers", MIN_MIN_LAYERS, MAX_MIN_LAYERS)
        self.max_layers = self.pull_numbers("max_layers", MIN_MAX_LAYERS, MAX_MAX_LAYERS)
        self.min_nodes = self.pull_numbers("min_nodes", MIN_MIN_NODES, MAX_MIN_NODES)
        self.max_nodes = self.pull_numbers("max_nodes", MIN_MAX_NODES, MAX_MAX_NODES)
        self.min_p_insert = self.pull_numbers("min_p_insert", MIN_MIN_P_INSERT, MAX_MIN_P_INSERT)
        self.max_p_insert = self.pull_numbers("max_p_insert", MIN_MAX_P_INSERT, MAX_MAX_P_INSERT)
        self.get_initial_graph()
        self.evolve_initial_graph()
        self.differentiate()

    def get_initial_graph(self):
        """Create a graph connecting inputs to outputs with a black box."""
        self.G = nx.MultiDiGraph()
        self.G.add_node("source", label="SOURCE",
                        shape="cylinder", color="gold")
        self.G.add_node("sink", label="SINK",
                        shape="cylinder", color="gold")
        n_initial_boxes = self.pull_numbers("n_initial_boxes",
                                            MIN_INITIAL_BOXES,
                                            MAX_INITIAL_BOXES)
        self.n_initial_boxes = n_initial_boxes
        for box_number in range(n_initial_boxes):
            self.G.add_node(box_number, label=box_number,
                            shape="square", color="black")
        for input_number in range(self.n_in):
            input_key = f"input_{input_number}"
            self.G.add_node(input_key, label=input_key,
                            shape="circle", color="blue")
            self.G.add_edge("source", input_key)
            [self.G.add_edge(input_key, box_number)
             for box_number in range(n_initial_boxes)]
        for output_number in range(self.n_in):
            output_key = f"output_{output_number}"
            self.G.add_node(output_key, label=output_key,
                            shape="triangle", color="red")
            self.G.add_edge(output_key, "sink")
            [self.G.add_edge(box_number, output_key)
             for box_number in range(n_initial_boxes)]

    def evolve_initial_graph(self):
        self.n_mutations = self.pull_numbers("n_mutations",
                                             MIN_MUTATIONS, MAX_MUTATIONS)
        self.mutations = self.pull_choices("mutations",
                                           MUTATION_OPTIONS, self.n_mutations)
        [locals()[mutation]() for mutation in self.mutations]

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
                M.add_node(node_id, label="", style="filled",
                           shape="square", color="black")
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

    def make_model(self):
        """Build the keras model described by a graph."""
        self.outputs = [self.get_output(id) for id in list(self.G.predecessors("sink"))]
        self.inputs = [self.G.node[id]['brick'] for id in list(self.G.successors('source'))]
        self.model = K.Model(self.inputs, self.outputs)

    def get_output(self, id):
        """
        Get the output of a node in a computation graph.
        Pull inputs from predecessors.
        """
        node = self.G.node[id]
        keys = node.keys()
        node_type = node["node_type"]
        log('get output for', node)
        if node["output"] is not None:
            return node["output"]
        else:
            parent_ids = list(self.G.predecessors(id))
            if node_type is not "input":
                inputs = [self.get_output(parent_id) for parent_id in parent_ids]
                inputs = L.Concatenate()(inputs) if len(inputs) > 1 else inputs[0]
                if node_type is "recurrent":
                    output = inputs
                else:
                    brick = self.build_brick(id, inputs)
                    log("got brick", brick)
                    output = brick(inputs)
                    if "output_shape" not in keys and "gives_feedback" not in keys:
                        try:
                            output = L.Add()([inputs, output])
                        except Exception as e:
                            log("error adding inputs to output", e)
                        try:
                            output = L.Concatenate()([inputs, output])
                        except Exception as e:
                            log("error concatenating inputs to output", e)
            else:
                output = self.build_brick(id)
            if "output_shape" not in keys:
                output = InstanceNormalization()(output)
            self.G.node[id]["output"] = output
            log("got output", output)
            return output

    def build_brick(self, id, inputs=None):
        """Make the keras operation to be executed at a given node."""
        global G
        node = G.node[id]
        log('build brick for', node)
        brick_type = self.agent.pull_choices(f"{id}_brick_type", BRICK_OPTIONS)
        d_in = d_out = inputs.shape[-1]
        if brick_type == "input":
            brick = L.Input(self.agent.code_spec.shape)
        if brick_type == "sepconv1d":
            filters = self.pull_numbers(f"{id}_filters", 1, 32)
            activation = self.pull_choices(f"{id}_activation", ACTIVATION_OPTIONS)
            brick = L.SeparableConv1D(filters, 1, activation=activation)
        if brick_type in ["deep", "wide_deep"]:
            brick = get_mlp(self.agent, d_in, d_out, -1)
        if "k_conv" in brick_type:
            set_size = brick_type[-1]
            brick = KConvSet(self.agent, d_in, d_out, set_size)
        if brick_type == "transformer":
            brick = Transformer(self.agent)
        G.node[id]['brick'] = brick
        return brick

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
        return len([node for node in list(self.G.nodes(data=True)) if node[1]["shape"] == "square"])
