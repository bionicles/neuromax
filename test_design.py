import tensorflow as tf
import networkx as nx
import random

B, L, K = tf.keras.backend, tf.keras.layers, tf.keras

STEPS, MIN_LAYERS, MAX_LAYERS, MIN_NODES, MAX_NODES = 2, 1, 3, 1, 2
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
    G = nx.MultiDiGraph()
    names = []
    for layer_number in range(num_layers):
        num_nodes = random.randint(MIN_NODES, MAX_NODES)
        print("layer number", layer_number, "has", num_nodes, "nodes")
        names_of_nodes_in_this_layer = []
        for node_number in range(num_nodes):
            if parent is not None:
                node_name = "{}.{}.{}".format(parent, layer_number, node_number)
            else:
                node_name = "{}.{}".format(layer_number, node_number)
            print("add node", node_name)
            G.add_node(node_name, label="", style="filled", shape="square", color="black")
            names_of_nodes_in_this_layer.append(node_name)
        names.append(names_of_nodes_in_this_layer)
    for predecessor_layer_number, predecessor_node_names in enumerate(names):
        for successor_layer_number, successor_node_names in enumerate(names):
            if predecessor_layer_number >= successor_layer_number:
                continue
            print("predecessor", predecessor_layer_number, predecessor_node_names)
            print("successor", successor_layer_number, successor_node_names)
            for predecessor_node_name in predecessor_node_names:
                for successor_node_name in successor_node_names:
                    print("adding edge from", predecessor_node_name, "to", successor_node_name)
                    G.add_edge(predecessor_node_name, successor_node_name)
    return G, names


def insert_motif(G, name, motif):
    # get a motif
    Gi, new_names = get_regulon(name)
    # add the motif to the graph
    Gn = nx.compose(G, Gi)
    # point the prececessors of the node to replace at all nodes in first layer
    predecessors = Gn.predecessors(name)
    successors = new_names[0]
    for predecessor in predecessors:
        for successor in successors:
            print("adding edge from", predecessor, "to", successor)
            Gn.add_edge(predecessor, successor)
    # point the last node in the motif at the successors of the node to replace
    predecessors = new_names[len(new_names)-1]
    successors = list(G.successors(name))
    for predecessor in predecessors:
        for successor in successors:
            print("adding edge from", predecessor, "to", successor)
            Gn.add_edge(predecessor, successor)
    # remove the node
    Gn.remove_node(name)
    return Gn


# # we decide what boxes will do:
# def differentiate_boxes(G):
#     for node in G.nodes(data=True):
#         node_id, node_data = node
#         if node_data["shape"] is "square":
#             layer = "dense"
#             activation = random.choice(["linear", "tanh"])
#             label = f"{layer} {activation}"
#             print(f"setting {node_id} to {label}")
#             node[1]["activation"] = activation
#             node[1]["layer"] = layer
#             node[1]["label"] = label
#             node[1]["color"] = "yellow" if activation is "linear" else "green"
#             node[1]["units"] = 64
#             node[1]['tf_layer'] = get_layer("dense", activation=node[1]['activation'], units=node[1]['units'])
#         if node_data["shape"] is "triangle":
#             node[1]['activation'] = "sigmoid"
#             node[1]['units'] = 3
#             node[1]['tf_layer'] = get_layer("dense", activation=node[1]['activation'], units=node[1]['units'])
#         if node_data["shape"] is "circle":
#             node[1]["input_shape"] = (5, )
#             node[1]['tf_layer'] = get_layer("input", shape=node[1]["input_shape"])
#     return G


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
    differentiate()
    # screenshot(G2, "architecture_search_is_fun")
    # make_model(G2)


if __name__ == "__main__":
    main()
