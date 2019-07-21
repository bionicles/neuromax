import networkx as nx
import random
import tensorflow as tf
B = tf.keras.backend
L = tf.keras.layers
K = tf.keras
import time

def get_graph():
    G = nx.MultiDiGraph()
    G.add_node("input1", label="", shape="circle", style="filled", color="blue")
    G.add_node("input2", label="", shape="circle", style="filled", color="blue")
    G.add_node("dense1", label="", shape="square", style="filled", color="black")
    G.add_node("dense2", label="", shape="square", style="filled", color="black")
    G.add_node("dense3", label="", shape="square", style="filled", color="black")
    G.add_node("dense4", label="", shape="square", style="filled", color="black")
    G.add_node("output1", label="", shape="triangle", style="filled", color="red")
    G.add_node("output2", label="", shape="triangle", style="filled", color="red")
    G.add_edge("input1", "dense1")
    G.add_edge("input2", "dense2")
    G.add_edge("dense1", "dense2")
    G.add_edge("dense2", "dense3")
    G.add_edge("dense1", "dense3")
    G.add_edge("dense3", "output1")
    G.add_edge("dense2", "dense4")
    G.add_edge("dense4", "output2")
    return G

def differentiate_boxes(G):
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
def get_layer(layer_name, shape=None, activation=None, units=None):
    if layer_name is "input":
        return K.Input(shape)
    if layer_name is "dense":
        return L.Dense(units=units, activation=activation)
G = get_graph()
G = differentiate_boxes(G)

def make_model(graph):
    inputs, outputs = [], []
    for layer_key in dict(graph.nodes(data=True)).keys():
        input_layers = list(graph.predecessors(layer_key))
        if 'input' in layer_key:
            inputs.append(dict(G.nodes(data=True))[layer_key]['tf_layer'])
        if len(input_layers)>1:
            print("connecting", input_layers, "to", layer_key)
            concat = L.Concatenate()([dict(G.nodes(data=True))[L]['tf_layer'] for L in input_layers])
            dict(G.nodes(data=True))[layer_key]['tf_layer'] = dict(G.nodes(data=True))[layer_key]['tf_layer'](concat)
        if len(input_layers)==1:
            print("connecting", input_layers, "to", layer_key)
            dict(G.nodes(data=True))[layer_key]['tf_layer'] = dict(G.nodes(data=True))[layer_key]['tf_layer'](dict(G.nodes(data=True))[input_layers[0]]['tf_layer'])
        if 'output' in layer_key:
            outputs.append(dict(G.nodes(data=True))[layer_key]['tf_layer'])
    model = K.Model(inputs, outputs)
    model.summary()
    K.utils.plot_model(model, "./nets/model.png", rankdir="LR")
    return model
print(dict(G.nodes(data=True))['input1'])
make_model(G)

def screenshot(G, step):
    print("SCREENSHOT", step)
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update(rankdir="LR")
    A.draw(path="./nets/{}.png".format(step), prog="dot")


screenshot(G, "kamel_is_cool")
