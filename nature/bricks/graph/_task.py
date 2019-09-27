import tensorflow as tf
import networkx as nx
from nature import add_node, get_output, screenshot_graph
from tools import safe_sample, plot_model, log

K, L = tf.keras, tf.keras.layers
MIN_STACKS, MAX_STACKS = 6, 6


def TaskGraph(AI, in_specs, out_specs):
    G = nx.MultiDiGraph()
    add_node(G, "critic", "red", "triangle", "critic", spec=AI.loss_spec)
    add_node(G, "source", "gold", "cylinder", "source")
    add_node(G, "merge_0", "black", "circle", "merge")
    add_node(G, "merge_1", "black", "circle", "merge")
    add_node(G, "merge_2", "black", "circle", "merge")
    add_node(G, "sink", "gold", "cylinder", "sink")
    G.add_edges_from([
        ('merge_0', 'merge_1'), ('merge_1', 'merge_2'), ('merge_2', 'critic')])
    n_stacks = safe_sample(MIN_STACKS, MAX_STACKS)
    for i in range(n_stacks):
        add_node(G, i, "black", "square", "brick")
        G.add_edges_from([("merge_0", i), (i, 'merge_1')])
    for n, in_spec in enumerate(in_specs):
        in_key = f"input_{n}"
        add_node(G, in_key, "blue", "circle", "input", spec=in_spec, n=n)
        G.add_edge("source", in_key)
        G.add_edge(in_key, "merge_0")
    for n, out_spec in enumerate(out_specs):
        out_key = f"output_{n}"
        add_node(G, out_key, "red", "triangle", "output", spec=out_spec, n=n)
        G.add_edges_from([
            ('merge_1', out_key), (out_key, "merge_2"), (out_key, 'sink')])
    G.add_edge("critic", "sink")
    return G


def Model(G, AI):
    outputs = [get_output(G, AI, i) for i in list(G.predecessors("sink"))]
    inputs = [G.node[i]['input'] for i in list(G.successors('source'))]
    log('outputs', outputs, color="green", debug=True)
    return K.Model(inputs, outputs)


def TaskModel(AI, in_specs, out_specs):
    G = TaskGraph(AI, in_specs, out_specs)
    screenshot_graph(G, ".", 'graph')
    model = Model(G, AI)
    plot_model(model, "model")
    return G, model
