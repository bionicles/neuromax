import tensorflow as tf
import networkx as nx

from nature import Brick, add_node, get_output
from tools import screenshot_graph, show_model

K = tf.keras

N_INITIAL_BOXES = 1


def Graph(in_specs, out_specs):
    G = nx.MultiDiGraph()
    add_node(G, "source", "gold", "cylinder", "source")
    add_node(G, "sink", "gold", "cylinder", "sink")
    for i in range(N_INITIAL_BOXES):
        add_node(G, i, "black", "square", "brick")
    for n, in_spec in enumerate(in_specs):
        in_key = f"input_{n}"
        add_node(G, in_key, "blue", "circle", "input", spec=in_spec)
        G.add_edge("source", in_key)
        [G.add_edge(in_key, i) for i in range(N_INITIAL_BOXES)]
    for n, out_spec in enumerate(out_specs):
        out_key = f"output_{n}"
        add_node(G, out_key, "red", "triangle", "output", spec=out_spec)
        G.add_edge(out_key, "sink")
        [G.add_edge(i, out_key) for i in range(N_INITIAL_BOXES)]
    return G


def Model(G, agent):
    outputs = [get_output(G, agent, id, task_model=True)
               for id in list(G.predecessors("sink"))]
    inputs = [G.node[id]['input'] for id in list(G.successors('source'))]
    return K.Model(inputs, outputs)


def TaskModel(agent, in_specs, out_specs):
    G = Graph(in_specs, out_specs)
    screenshot_graph(G, ".", 'brick')
    model = Model(G, agent)
    show_model(model, "task_model")
    call = model.call
    return Brick(agent, in_specs, out_specs, G, model, call)
