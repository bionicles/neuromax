import tensorflow as tf
import networkx as nx

from nature import get_output, add_node, mutate, Resizer, screenshot_graph
from tools import get_spec

L = tf.keras.layers

N_INITIAL_BOXES = 4


def BrickGraph(spec, inputs):
    G = nx.MultiDiGraph()
    for box_number in range(N_INITIAL_BOXES):
        add_node(G, box_number, "black", "square", "brick")
    add_node(G, "source", "blue", "circle", "input",
             spec=spec, output=inputs)
    [G.add_edge("source", box) for box in range(N_INITIAL_BOXES)]
    add_node(G, "sink", "red", "triangle", "output", spec=spec)
    [G.add_edge(box, "sink") for box in range(N_INITIAL_BOXES)]
    return mutate(G)


class GraphBrick(L.Layer):
    def __init__(self, agent, inputs):
        super(GraphBrick, self).__init__()
        spec = get_spec(format="code", shape=inputs.shape)
        self.G = BrickGraph(spec, inputs)
        screenshot_graph(self.G, ".", 'brick')
        self.concat = L.Concatenate(-1)
        self.resizer = Resizer(spec.shape)
        self.agent = agent
        self.built = True

    def call(self, x):
        y = [get_output(self.G, self.agent, i)
             for i in list(self.G.predecessors("sink"))]
        y = self.concat(y)
        return self.resizer(y)
