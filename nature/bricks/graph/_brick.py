import tensorflow as tf
import networkx as nx

from nature import Brick, get_output, add_node, mutate
from tools import screenshot_graph, get_spec

L = tf.keras.layers

N_INITIAL_BOXES = 4


def Graph(spec):
    G = nx.MultiDiGraph()
    for box_number in range(N_INITIAL_BOXES):
        add_node(G, box_number, "black", "square", "brick")
    add_node(G, "source", "blue", "circle", "inputs", spec=spec)
    [G.add_edge("source", box) for box in range(N_INITIAL_BOXES)]
    add_node(G, "sink", "red", "triangle", "output", spec=spec)
    [G.add_edge(box, "sink") for box in range(N_INITIAL_BOXES)]
    return mutate(G)


def GraphBrick(agent, inputs):
    spec = get_spec(format="code", shape=inputs.shape)
    G = Graph(spec=spec)
    screenshot_graph(G, ".", 'brick')
    concat = L.Concatenate(-1)
    resizer = Resizer(agent, spec.shape)

    def call(self, x):
        y = [get_output(G, agent, i)
                   for i in list(G.predecessors("sink"))]
        y = concat(y)
        return resizer(y)

    return Brick(agent, inputs, G, call)
