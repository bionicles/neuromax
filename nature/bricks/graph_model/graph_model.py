# graph_model.py - bion
# why?: learn to map N inputs to M outputs with graph GP
import tensorflow_addons as tfa
import tensorflow as tf
# TODO: fix attention and DNC memory bricks
# from nature.bricks.multi_head_attention import MultiHeadAttention
from nature.bricks.graph_model.graph import Graph
from nature.bricks.dnc.cell import DNC_Cell
from nature.bricks.k_conv import KConvSet1D
from nature.bricks.kernel import get_kernel

from tools.get_unique_id import get_unique_id
from tools.show_model import show_model
from tools.log import log

InstanceNormalization = tfa.layers.InstanceNormalization
K = tf.keras
L = K.layers

# bricks
BRICK_OPTIONS = ["conv1d", "kernel", "k_conv", "dnc"]
# these should get pulled by bricks
MIN_FILTERS, MAX_FILTERS = 4, 32
ACTIVATION_OPTIONS = ["tanh"]
BATCH_SIZE = 8

class GraphModel:
    """
    GraphModel maps [n_in x code_shape] -> [n_out x code_shape]
    because we need to be able to handle multiple inputs
    """

    def __init__(self, agent, n_in=None, code_shape=None, n_out=None):
        log("\nGraphModel.__init__")
        self.agent = agent
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        self.name = get_unique_id("GraphModel")
        self.graph = Graph(agent, self.name)
        self.G = self.graph.G
        self.make_model()
        show_model(self.model, ".", "M", "png")

    def make_model(self):
        """Build the keras model described by a graph."""
        self.outputs = [self.get_output(id)
                        for id in list(self.G.predecessors("sink"))]
        self.inputs = [self.G.node[id]['brick']
                       for id in list(self.G.successors('source'))]
        self.model = K.Model(self.inputs, self.outputs)
        self.__call__ = self.model.__call__

    def get_output(self, id):
        """
        Get the output of a node in a computation graph.
        Pull inputs from predecessors.
        """
        node = self.G.node[id]
        keys = node.keys()
        node_type = node["node_type"]
        log('get output for', node)
        if node["output"] is not None:
            return node["output"]
        else:
            parent_ids = list(self.G.predecessors(id))
            if node_type is not "input":
                inputs = [self.get_output(parent_id) for parent_id in parent_ids]
                log("inputs:", inputs)
                inputs = L.Concatenate()(inputs) if len(inputs) > 1 else inputs[0]
                if node_type is "recurrent":
                    output = inputs
                else:
                    brick = self.build_brick(id, inputs)
                    log(f"got brick", brick)
                    output = brick(inputs, initial_state=self.G.node[id]['initial_state']) if "cell" in brick.__init__.__code__.co_varnames else brick(inputs)
                    if "output_shape" not in keys and "gives_feedback" not in keys:
                        try:
                            output = L.Add()([inputs, output])
                        except Exception as e:
                            log("error adding inputs to output", e)
                        try:
                            output = L.Concatenate()([inputs, output])
                        except Exception as e:
                            log("error concatenating inputs to output", e)
            else:
                output = self.build_brick(id)
            try:
                output = InstanceNormalization()(output)
            except Exception as e:
                log(f"can't use instance norm\n", e)
            self.G.node[id]["output"] = output
            log(f"got {node_type} output", output)
            return output

    def build_brick(self, id, inputs=None):
        """Make the keras operation to be executed at a given node."""
        node = self.G.node[id]
        log('build brick for', node)
        if node["node_type"] == "input":
            brick = L.Input(self.agent.code_spec.shape)
            brick_type = "input"
        else:
            d_in = d_out = inputs.shape[-1]
            brick_type = self.agent.pull_choices(
                f"{id}_brick_type", BRICK_OPTIONS)
            log("building a ", brick_type, "brick")
        if brick_type == "conv1d":
            filters = self.pull_numbers(f"{id}_filters", MIN_FILTERS, MAX_FILTERS)
            activation = self.pull_choices(f"{id}_activation",
                                           ACTIVATION_OPTIONS)
            brick = L.SeparableConv1D(filters, 1, activation=activation)
        if brick_type == "kernel":
            brick = get_kernel(self.agent, id, d_in, d_out, -1)
        if "k_conv" in brick_type:
            brick = KConvSet1D(self.agent, id, d_in, d_out, None)
        # if brick_type == "attention":
        #     brick = MultiHeadAttention(self.agent, id)
        if brick_type == "dnc":
            dnc_cell = DNC_Cell(self.agent, id, d_out)
            initial_state = dnc_cell.get_initial_state(batch_size=BATCH_SIZE)
            self.G.node[id]['initial_state'] = initial_state
            brick = L.RNN(dnc_cell, return_sequences=True)
        self.G.node[id]['brick_type'] = brick_type
        self.G.node[id]['brick'] = brick
        log("built a", brick_type)
        return brick
