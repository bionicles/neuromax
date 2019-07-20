import networkx as nx
# import tensorflow as tf


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
