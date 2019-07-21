import tensorflow as tf
import networkx as nx
import random

B, L, K = tf.keras.backend, tf.keras.layers, tf.keras

MIN_LAYERS, MAX_LAYERS = 1, 3
MIN_NODES, MAX_NODES = 1, 3
P_INSERT = 0.64
STEPS = 2

IMAGE_SIZE = "1024x512"
DTYPE = tf.float32

tasks = {
    "Molecules-v0": {
        "intelligence": "bodily-kinesthetic",
        "goal": "simulate atoms in quantum, chemical, and proteins",
        "observation_space": [
            ("atoms", (None, None, 10), tf.float32)
        ],
        "action_space": [
            ("forces", (None, None, 3), tf.float32)
        ]
    },
}


def get_initial_graph(tasks):
    G = nx.MultiDiGraph()
    G.add_node("source", label="", shape="cylinder", style="filled", color="gold")
    G.add_node("input", label="", shape="circle", style="filled", color="blue")
    G.add_node(1, label="", shape="square", style="filled", color="black")
    G.add_node("output", label="", shape="triangle", style="filled", color="red")
    G.add_node("sink", label="", shape="cylinder", style="filled", color="gold")
    G.add_edge("source", "input")
    G.add_edge("input", 1)
    G.add_edge(1, "output")
    G.add_edge("output", "sink")
    return G


def screenshot(G, step):
    print("SCREENSHOT", step)
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update(rankdir="LR")
    A.draw(path="./nets/{}.png".format(step), prog="dot")


def get_regulon(parent=None, layers=None):
    num_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    M = nx.MultiDiGraph()
    ids = []
    for layer_number in range(num_layers):
        n_nodes = random.randint(MIN_NODES, MAX_NODES)
        print(f"add layer {layer_number} with {n_nodes} nodes")
        ids_of_nodes_in_this_layer = []
        for node_number in range(n_nodes):
            if parent:
                node_id = "{}.{}.{}".format(parent, layer_number, node_number)
            else:
                node_id = "{}.{}".format(layer_number, node_number)
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
    global G
    # get a motif and add it to G
    Gi, new_ids = get_regulon(node_id)
    G = nx.compose(G, Gi)
    # connect the node's predecessors to all nodes in first layer
    predecessors = G.predecessors(node_id)
    successors = new_ids[0]
    for predecessor in predecessors:
        for successor in successors:
            print("adding edge from", predecessor, "to", successor)
            G.add_edge(predecessor, successor)
    # connect the last layer of the motif to the node's successors
    predecessors = new_ids[-1]
    successors = list(G.successors(node_id))
    for predecessor in predecessors:
        for successor in successors:
            print("adding edge from", predecessor, "to", successor)
            G.add_edge(predecessor, successor)
    G.remove_node(node_id)


def recurse():
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


# # we decide what boxes will do:
def differentiate():
    for node in G.nodes(data=True):
        node_id, node_data = node
        if node_data["shape"] is "square":
            layer = "dense"
            activation = random.choice(["linear", "tanh"])
            label = f"{layer} {activation}"
            print(f"setting {node_id} to {label}")
            node[1]["activation"] = activation
            node[1]["layer"] = layer
            node[1]["label"] = label
            node[1]["color"] = "yellow" if activation is "linear" else "green"
            node[1]["units"] = 64
            node[1]['tf_layer'] = get_layer("dense", activation=node[1]['activation'], units=node[1]['units'])
        if node_data["shape"] is "triangle":
            node[1]['activation'] = "sigmoid"
            node[1]['units'] = 3
            node[1]['tf_layer'] = get_layer("dense", activation=node[1]['activation'], units=node[1]['units'])
        if node_data["shape"] is "circle":
            node[1]["input_shape"] = (5, )
            node[1]['tf_layer'] = get_layer("input", shape=node[1]["input_shape"])
    return G


def make_model():
    model_outputs = [get_output(id, G) for id in list(G.predecessors("sink"))]
    model_inputs = [G.node[id]['op'] for id in list(G.successors('source'))]
    model = K.Model(model_inputs, model_outputs)
    model.summary()
    K.utils.plot_model(model, "./nets/model.png", rankdir="LR")
    return model


def get_output(id):
    global G
    node = G.node[id]
    if node["output_tensor"]:
        return node["output_tensor"]
    elif node["node_type"] is "input" and node["op"]:
        return node["op"]
    else:
        parent_ids = list(G.predecessors(id))
        inputs = [get_output(parent_id, G) for parent_id in parent_ids]
        output_tensor = build_op(id, G)(inputs)
        G.node[id]["output_tensor"] = output_tensor
        return output_tensor


def build_op(id):
    global G
    node = G.node[id]
    node_type = node["node_type"]
    if node_type is "dense":
        op = L.Dense(node['units'], node['activation'])
    if node_type is "input":
        op = L.Input(node['input_shape'])
    G.node[id]['op'] = op
    return op


def main():
    global G
    G = get_initial_graph(tasks)
    screenshot(G, "kamel_is_cool")
    recurse()
    differentiate()
    screenshot(G, STEPS + 1)
    # make_model(G2)


if __name__ == "__main__":
    main()
