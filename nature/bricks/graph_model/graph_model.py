# graph_model.py - bion
# why?: learn to map N inputs to M outputs with graph GP
import tensorflow as tf

from nature.bricks.residual_dense import get_residual_block_out, get_dense_block_out
from nature.bricks.norm_preact import get_norm_preact_out
from nature.bricks.get_mlp import get_mlp_out
from nature.bricks.swag import get_swag_out
from .graph import Graph

from tools.get_unique_id import get_unique_id
from tools.show_model import show_model
from tools.log import log

K = tf.keras
L = K.layers

BRICKS = ["swag", "mlp", "k_conv"]


class GraphModel(L.Layer):
    """
    GraphModel maps [n_in x code_shape] -> [n_out x code_shape]
    because we need to be able to handle multiple inputs
    """

    def __init__(self, agent):
        super(GraphModel, self).__init__()
        log("GraphModel init")
        self.code_shape = agent.code_spec.shape
        self.batch_size = agent.batch_size
        self.agent = agent
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        self.graph = Graph(agent, get_unique_id("GraphModel"))
        self.G = self.graph.G
        self.built = False
        self.build([agent.code_spec.shape for i in range(agent.n_in)])
        show_model(self.model, ".", "M", "png")

    def get_out(self, id):
        """
        Get the output of a node in a computation graph.
        Pull inputs from predecessors.
        """
        node = self.G.node[id]
        node_type = node["node_type"]
        if node["out"] is not None:
            return node["out"]
        else:
            if node_type is "input":
                out = L.Input(
                    self.code_shape, batch_size=self.batch_size)
            else:
                parent_ids = list(self.G.predecessors(id))
                inputs = [self.get_out(parent_id) for parent_id in parent_ids]
                if len(inputs) > 1:
                    inputs = L.Concatenate(-1)(inputs)
                else:
                    inputs = inputs[0]
                inputs = get_norm_preact_out(self.agent, id, inputs)
                d_out = inputs.shape[-1]
                brick_type = self.agent.pull_choices(
                    f"{id}_brick_type", BRICKS)
                if brick_type == "residual":
                    out, brick = get_residual_block_out(
                        self.agent, id, inputs, units=d_out, return_brick=True)
                if brick_type == "dense":
                    out, brick = get_dense_block_out(
                        self.agent, id, inputs, units=d_out, return_brick=True)
                if brick_type == "mlp":
                    out, brick = get_mlp_out(
                        self.agent, id, inputs,
                        last_layer=(d_out, "tanh"), return_brick=True)
                if brick_type == "swag":
                    out, brick = get_swag_out(
                        self.agent, id, inputs, units=d_out, return_brick=True)
                self.G.node[id]['brick_type'] = brick_type
            self.G.node[id]["out"] = out
            return out

    def build(self, input_shapes):
        """Build the keras model described by a graph."""
        if self.built:
            return self
        self.outs = [self.get_out(id)
                     for id in list(self.G.predecessors("sink"))]
        self.inputs = [self.G.node[id]['brick']
                       for id in list(self.G.successors('source'))]
        self.model = K.Model(self.inputs, self.outs)
        self.built = True
        return self

    def call(self, codes):
        log("")
        log("GraphModel call", color="blue")
        log("code spec", self.agent.code_spec, color="blue")
        log(f"got {len(codes)} codes", color="yellow")
        log("")
        return self.model(codes)
