# nature.py - bion and kamel, july 2019
# why?: experiment faster with recursive neural architecture search
import tensorflow as tf
import networkx as nx
import random
from .conv_kernel import NoisyDropConnectDense, SelfAttention, KernelConvSet
B, L, K = tf.keras.backend, tf.keras.layers, tf.keras

IMAGE_PATH = "../../archive/nets"
IMAGE_SIZE = "1024x512"
DTYPE = tf.float32

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
    print("SCREENSHOT", step)
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update(rankdir="LR")
    A.draw(path=f"{IMAGE_PATH}/{step}.png", prog="dot")


def get_regulon(parent=None, layers=None):
    """Build a subgraph to replace a node."""
    num_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    M = nx.MultiDiGraph()
    ids = []
    for layer_number in range(num_layers):
        n_nodes = random.randint(MIN_NODES, MAX_NODES)
        print(f"add layer {layer_number} with {n_nodes} nodes")
        ids_of_nodes_in_this_layer = []
        for node_number in range(n_nodes):
            if parent:
                node_id = f"{parent}.{layer_number}.{node_number}"
            else:
                node_id = f"{layer_number}.{node_number}"
            print(f"add node {node_id}")
            M.add_node(node_id, label="", style="filled", shape="square", color="black")
            ids_of_nodes_in_this_layer.append(node_id)
        ids.append(ids_of_nodes_in_this_layer)
    for predecessor_layer_number, predecessor_node_ids in enumerate(ids):
        for successor_layer_number, successor_node_ids in enumerate(ids):
            if predecessor_layer_number >= successor_layer_number:
                continue
            for predecessor_node_id in predecessor_node_ids:
                for successor_node_id in successor_node_ids:
                    print(f"{predecessor_node_id}--->{successor_node_id}")
                    M.add_edge(predecessor_node_id, successor_node_id)
    return M, ids


def insert_motif(node_id, motif):
    """Replace a node with a subgraph."""
    global G
    M, new_ids = get_regulon(node_id)
    G = nx.compose(G, M)
    # connect the node's predecessors to all nodes in first layer
    predecessors = G.predecessors(node_id)
    successors = new_ids[0]
    for predecessor in predecessors:
        for successor in successors:
            print(f"{predecessor}--->{successor}")
            G.add_edge(predecessor, successor)
    # connect the last layer of the motif to the node's successors
    predecessors = new_ids[-1]
    successors = list(G.successors(node_id))
    for predecessor in predecessors:
        for successor in successors:
            print(f"{predecessor}--->{successor}")
            G.add_edge(predecessor, successor)
    G.remove_node(node_id)


def recurse():
    """Build a graph with recursive motif insertion."""
    for step in range(STEPS):
        nodes = G.nodes(data=True)
        for node in nodes:
            try:
                if node[1]["shape"] is "square":
                    if random.random() < P_INSERT:
                        insert_motif(node[0], "regulon")
            except Exception as e:
                print('exception in recurse', e)
        screenshot(G, f"{step+1}")


def differentiate():
    """Assign ops and arguments to nodes."""
    for node in G.nodes(data=True):
        node_id, node_data = node
        node[1]["output"] = None
        node[1]["op"] = None
        if node_data["shape"] is "square":
            node_type = random.choice(['conv1D', 'dense', 'NoisyDropConnectDense', 'SelfAttention', 'LSTM', 'BatchNormalization'])  # 'KernelConvSet'])
            if node_type is 'conv1D':
                activation = random.choice(["relu", "sigmoid"])
                label = f"{node_type} {activation}"
                filters, kernel_size = random.randint(MIN_FILTER, MAX_FILTER), KERNEL_SIZE
                print(f"setting {node_id} to {label}")
                node[1]["activation"] = activation
                node[1]["node_type"] = node_type
                node[1]["filters"] = filters
                node[1]["kernel_size"] = kernel_size
                node[1]["label"] = label
                node[1]["color"] = "yellow" if activation is "relu" else "green"

            if node_type is 'dense' or "NoisyDropConnectDense" or "LSTM":
                activation = random.choice(["linear", "tanh", "sigmoid"])
                label = f"{node_type} {activation}"
                print(f"setting {node_id} to {label}")
                node[1]["activation"] = activation
                node[1]["node_type"] = node_type
                node[1]["label"] = label
                node[1]["color"] = "yellow" if activation is "linear" else "green"
                node[1]["units"] = 64
                if node_type is "NoisyDropConnectDense":
                    node[1]["stddev"] = 0.01
            if node_type is 'SelfAttention':
                node[1]["d_features"] = 10
            if node_type is 'BatchNormalization':
                node[1]["node_type"] = node_type
# if node_type is "KernelConvSet":
#     node[1]["d_features"] = 10
#     node[1]["d_output"] = 5


def make_model():
    """Build the keras model described by a graph."""
    model_outputs = [get_output(id) for id in list(G.predecessors("sink"))]
    model_inputs = [G.node[id]['op'] for id in list(G.successors('source'))]
    model = K.Model(model_inputs, model_outputs)
    model.summary()
    K.utils.plot_model(model, "../../archive/nets/model.png", rankdir="LR")
    return model


def get_output(id):
    """
    Get the output of a node in a computation graph.
    Pull inputs from predecessors.
    """
    global G
    node = G.node[id]
    node_type = node["node_type"]
    print('get output for', node)
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
            op = build_op(id)
            print("got op", op)
            output = op(inputs)
        else:
            output = build_op(id)
        G.node[id]["output"] = output
        print("got output", output)
        return output


def build_op(id):
    """Make the keras operation to be executed at a given node."""
    global G
    node = G.node[id]
    print('build op for', node)
    node_type = node["node_type"]
    if node_type is "dense":
        op = L.Dense(node['units'], node['activation'])
    if node_type is "conv1D":
        op = L.Conv1D(node['filters'], node['kernel_size'], activation=node['activation'])
    if node_type is "input":
        op = L.Input(node['input_shape'])
    if node_type is "NoisyDropConnectDense":
        op = NoisyDropConnectDense(units=node['units'], activation=node['activation'], stddev=node['stddev'])
    if node_type is 'SelfAttention':
        op = SelfAttention(node['d_features'])
    if node_type is 'LSTM':
        op = L.LSTM(node["units"], node['activation'], return_sequences=True)
    if node_type is 'BatchNormalization':
        op = L.BatchNormalization()
    G.node[id]['op'] = op
    print("built op", op)
    return op


def get_agent(trial_number, hp, d_in=10, d_out=3):
    """Build a model given hyperparameters and input/output shapes."""
    global G
    G = get_initial_graph(tasks)
    screenshot(G, '0')
    recurse()
    differentiate()
    screenshot(G, hp.recursions + 1)
    return make_model()


if __name__ == "__main__":
    get_agent()
