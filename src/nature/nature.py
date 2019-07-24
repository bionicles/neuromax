# nature.py - bion and kamel, july 2019
# why?: experiment faster with recursive neural architecture search
import tensorflow as tf
import networkx as nx
import random
from datetime import datetime
from .conv_kernel import NoisyDropConnectDense, SelfAttention, KConvSet
B, L, K = tf.keras.backend, tf.keras.layers, tf.keras

IMAGE_PATH = "archive/nets"
IMAGE_SIZE = "1024x512"
DTYPE = tf.float32
DEBUG = True


def log(*args):
    if DEBUG:
        print(*args)

tasks = {
    "Molecules-v0": {
        "intelligence": "bodily-kinesthetic",
        "goal": "simulate atoms in quantum, chemical, and proteins",
        "observation_space": ("atoms", (None, 10), tf.float32),
        "action_space": ("forces", (None, 3), tf.float32)
    },
}

def get_initial_graph(tasks):
    """Create a graph connecting task inputs to outputs with a black box."""
    G = nx.MultiDiGraph()
    G.add_node("source", label="", shape="cylinder", style="filled", color="gold")
    G.add_node("input", label="", shape="circle", style="filled", color="blue")
    G.node["input"]["node_type"] = "input"
    G.node["input"]["input_shape"] = tasks["Molecules-v0"]["observation_space"][1]
    G.add_node(1, label="", shape="square", style="filled", color="black")
    G.add_node("output", label="", shape="triangle", style="filled", color="red")
    G.node["output"]["node_type"] = "output"
    G.add_node("sink", label="", shape="cylinder", style="filled", color="gold")
    G.add_edge("source", "input")
    G.add_edge("input", 1)
    G.add_edge(1, "output")
    G.add_edge("output", "sink")
    return G


def screenshot(G, step):
    """Make a png image of a graph."""
    log("SCREENSHOT", step)
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update(rankdir="LR")
    A.draw(path=f"{IMAGE_PATH}/{step}.png", prog="dot")


def get_regulon(hp, parent=None, layers=None):
    """Build a subgraph to replace a node."""
    num_layers = random.randint(hp.min_layers, hp.max_layers)
    M = nx.MultiDiGraph()
    ids = []
    for layer_number in range(num_layers):
        n_nodes = random.randint(hp.min_nodes, hp.max_nodes)
        log(f"add layer {layer_number} with {n_nodes} nodes")
        ids_of_nodes_in_this_layer = []
        for node_number in range(n_nodes):
            if parent:
                node_id = f"{parent}.{layer_number}.{node_number}"
            else:
                node_id = f"{layer_number}.{node_number}"
            log(f"add node {node_id}")
            M.add_node(node_id, label="", style="filled", shape="square", color="black")
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


def insert_motif(node_id, hp):
    """Replace a node with a subgraph."""
    global G
    M, new_ids = get_regulon(hp, parent=node_id)
    G = nx.compose(G, M)
    # connect the node's predecessors to all nodes in first layer
    predecessors = G.predecessors(node_id)
    successors = new_ids[0]
    for predecessor in predecessors:
        for successor in successors:
            log(f"{predecessor}--->{successor}")
            G.add_edge(predecessor, successor)
    # connect the last layer of the motif to the node's successors
    predecessors = new_ids[-1]
    successors = list(G.successors(node_id))
    for predecessor in predecessors:
        for successor in successors:
            log(f"{predecessor}--->{successor}")
            G.add_edge(predecessor, successor)
    G.remove_node(node_id)


def recurse(hp):
    """Build a graph with recursive motif insertion."""
    for step in range(hp.recursions):
        nodes = G.nodes(data=True)
        for node in nodes:
            try:
                if node[1]["shape"] is "square":
                    if random.random() < hp.p_insert:
                        insert_motif(node[0], hp)
            except Exception as e:
                log('exception in recurse', e)
        screenshot(G, f"{step+1}")


def differentiate(hp):
    """Assign ops and arguments to nodes."""
    for node in G.nodes(data=True):
        node_id, node_data = node
        node[1]["output"] = None
        node[1]["op"] = None
        if node_data["shape"] is "square":
            node_type = random.choice(['sepconv1D', 'dense', 'NoisyDropConnectDense', 'SelfAttention', "GRU", "KConvSet"])
            activation = "tanh"
            label = f"{node_type} {activation}"
            node[1]["activation"] = activation
            node[1]["node_type"] = node_type
            node[1]["label"] = label
            log(f"setting {node_id} to {label}")
            if node_type is 'sepconv1D':
                node[1]["filters"] = random.randint(hp.min_filters, hp.max_filters)
                node[1]["kernel_size"] = 1
            if node_type is 'dense' or "NoisyDropConnectDense" or "GRU":
                node[1]["units"] = random.randint(hp.min_units, hp.max_units)
            if node_type is "NoisyDropConnectDense":
                node[1]["stddev"] = hp.stddev
            if node_type is "KConvSet":
                node[1]['hp'] = hp


def make_model():
    """Build the keras model described by a graph."""
    model_outputs = [get_output(id) for id in list(G.predecessors("sink"))]
    model_inputs = [G.node[id]['op'] for id in list(G.successors('source'))]
    model_outputs = L.SeparableConv1D(3, 1, activation="tanh")(model_outputs[0])
    model = K.Model(model_inputs, model_outputs)
    model.summary()
    K.utils.plot_model(model, "archive/nets/model.png", rankdir="LR")
    return model


def get_output(id):
    """
    Get the output of a node in a computation graph.
    Pull inputs from predecessors.
    """
    global G
    node = G.node[id]
    node_type = node["node_type"]
    log('get output for', node)
    if node["output"] is not None:
        return node["output"]
    elif node_type is "input" and node["op"] is not None:
        return node["op"]
    else:
        parent_ids = list(G.predecessors(id))
        if node_type is not "input":
            inputs = [get_output(parent_id) for parent_id in parent_ids]
            inputs = L.Concatenate()(inputs) if len(inputs) > 1 else inputs[0]
            if node_type is "output":
                return inputs
            op = build_op(id, inputs)
            log("got op", op)
            output = op(inputs)
        else:
            output = build_op(id)
        try:
            output = L.Add()([inputs, output])
        except Exception as e:
            log("failed to make residual connection at node", id, e)
        output = L.BatchNormalization()(output)
        G.node[id]["output"] = output
        log("got output", output)
        return output


def build_op(id, inputs=None):
    """Make the keras operation to be executed at a given node."""
    global G
    node = G.node[id]
    log('build op for', node)
    node_type = node["node_type"]
    if node_type is "dense":
        op = L.Dense(node['units'], node['activation'])
    if node_type is "sepconv1D":
        op = L.SeparableConv1D(node['filters'], node['kernel_size'], activation=node['activation'])
    if node_type is "input":
        op = L.Input(node['input_shape'])
    if node_type is "NoisyDropConnectDense":
        op = NoisyDropConnectDense(units=node['units'], activation=node['activation'], stddev=node['stddev'])
    if node_type is "KConvSet":
        hp = node['hp']
        d_in = inputs.shape[-1]
        d_out = random.randint(hp.min_units, hp.max_units)
        N = random.randint(1, 2) # SET SIZE -- TODO: DEBUG 3
        op = KConvSet(hp, d_in, d_out, N)
    if node_type is 'SelfAttention':
        op = SelfAttention()
    if node_type is 'GRU':
        op = L.GRU(node["units"], node['activation'], return_sequences=True)
    if node_type is 'BatchNormalization':
        op = L.BatchNormalization()
    G.node[id]['op'] = op
    log("built op", op)
    return op


def get_agent(trial_number, hp, d_in=10, d_out=3):
    """Build a model given hyperparameters and input/output shapes."""
    global G
    G = get_initial_graph(tasks)
    screenshot(G, '0')
    recurse(hp)
    differentiate(hp)
    screenshot(G, hp.recursions + 1)
    model = make_model()
    [log(item) for item in hp.items()]
    return model, str(datetime.now()).replace(" ", "_")
