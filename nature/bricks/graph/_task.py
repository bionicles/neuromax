import tensorflow as tf
import networkx as nx

from nature import add_node, get_output, screenshot_graph
from tools import safe_sample, show_model, log

K = tf.keras
L = K.layers

MIN_STACKS, MAX_STACKS = 1, 8


def TaskGraph(in_specs, out_specs):
    G = nx.MultiDiGraph()
    add_node(G, "source", "gold", "cylinder", "source")
    add_node(G, "sink", "gold", "cylinder", "sink")
    n_stacks = safe_sample(MIN_STACKS, MAX_STACKS)
    for i in range(n_stacks):
        add_node(G, i, "black", "square", "brick")
    for n, in_spec in enumerate(in_specs):
        in_key = f"input_{n}"
        add_node(G, in_key, "blue", "circle", "input", spec=in_spec)
        G.add_edge("source", in_key)
        [G.add_edge(in_key, i) for i in range(n_stacks)]
    for n, out_spec in enumerate(out_specs):
        out_key = f"output_{n}"
        add_node(G, out_key, "red", "triangle", "output", spec=out_spec)
        G.add_edge(out_key, "sink")
        [G.add_edge(i, out_key) for i in range(n_stacks)]
    return G


def Model(G, agent):
    outputs = [get_output(G, agent, i) for i in list(G.predecessors("sink"))]
    inputs = [G.node[i]['input'] for i in list(G.successors('source'))]
    log('outputs', outputs, color="green", debug=True)
    return K.Model(inputs, outputs)


def TaskModel(agent, in_specs, out_specs):
    G = TaskGraph(in_specs, out_specs)
    screenshot_graph(G, ".", 'brick')
    model = Model(G, agent)
    show_model(model, "task_model")
    return G, model
