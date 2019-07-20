import networkx as nx
import tensorflow as tf
import random


def get_graph():
    G = nx.MultiDiGraph()
    G.add_node("input", label="", shape="circle", style="filled", color="blue")
    G.add_node(1, label="", shape="square", style="filled", color="black")
    G.add_node(2, label="", shape="square", style="filled", color="black")
    G.add_node(3, label="", shape="square", style="filled", color="black")
    G.add_node("output", label="", shape="triangle", style="filled", color="red")
    G.add_edge("input", 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(1, 3)
    G.add_edge(3, "output")
    return G


G = get_graph()


def screenshot(G, step):
    print("SCREENSHOT", step)
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update(rankdir="LR")
    A.draw(path="./nets/{}.png".format(step), prog="dot")


screenshot(G, "kamel_is_cool")


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
        print(node_id, ":", node_data)
    return G


G2 = differentiate_boxes(G)
screenshot(G2, "architecture_search_is_fun")
