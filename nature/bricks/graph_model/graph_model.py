# graph_model.py - bion
# why?: learn to map N inputs to M outputs with graph GP
from tensorflow_addons.layers import InstanceNormalization
import tensorflow_probability as tfp
import tensorflow as tf

from nature.bricks.graph_model.graph import Graph
from nature.bricks.multiply import get_multiply
from nature.bricks.k_conv import KConvSet1D
from nature.bricks.get_mlp import get_mlp
from nature.bricks.conv import get_conv_1D
from nature.bricks.swag import get_swag

from tools.get_unique_id import get_unique_id
from tools.show_model import show_model
from tools.log import log

tfpl = tfp.layers
K = tf.keras
L = K.layers

# bricks "conv1d", , "multiply",
BRICK_OPTIONS = ["swag", "mlp", "k_conv"]


class GraphModel(L.Layer):
    """
    GraphModel maps [n_in x code_shape] -> [n_out x code_shape]
    because we need to be able to handle multiple inputs
    """

    def __init__(self, agent):
        super(GraphModel, self).__init__()
        log("GraphModel init")
        self.agent = agent
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        self.graph = Graph(agent, get_unique_id("GraphModel"))
        self.G = self.graph.G
        self.built = False
        self.build([agent.code_spec.shape for i in range(agent.n_in)])
        show_model(self.model, ".", "M", "png")

    def get_output(self, id):
        """
        Get the output of a node in a computation graph.
        Pull inputs from predecessors.
        """
        node = self.G.node[id]
        keys = node.keys()
        node_type = node["node_type"]
        if node["output"] is not None:
            return node["output"]
        else:
            parent_ids = list(self.G.predecessors(id))
            if node_type is not "input":
                inputs = [self.get_output(parent_id) for parent_id in parent_ids]
                inputs = L.Concatenate(1)(inputs) if len(inputs) > 1 else inputs[0]
                inputs = InstanceNormalization()(inputs)
                if node_type is "recurrent":
                    output = inputs
                else:
                    brick = self.build_brick(id, inputs)
                    output = brick(inputs)
                    if "output_shape" not in keys and "gives_feedback" not in keys:
                        try:
                            output = L.Add()([inputs, output])
                        except Exception as e:
                            log("error adding inputs to output", e, color="red")
                            try:
                                output = L.Concatenate(1)([inputs, output])
                            except Exception as e:
                                log("error concatenating in+out", e, color="red")
            else:
                output = self.build_brick(id)
            self.G.node[id]["output"] = output
            return output

    def build_brick(self, id, inputs=None):
        """Make the keras operation to be executed at a given node."""
        node = self.G.node[id]
        # log('build brick for', node)
        if node["node_type"] == "input":
            brick = L.Input(self.agent.code_spec.shape,
                            batch_size=self.agent.batch_size)
            brick_type = "input"
        else:
            d_in = d_out = inputs.shape[-1]
            brick_type = self.agent.pull_choices(
                f"{id}_brick_type", BRICK_OPTIONS)
            # log("building a ", brick_type, "brick")
        if brick_type == "conv1d":
            brick = get_conv_1D(self.agent, id, d_out)
        if brick_type == "mlp":
            brick = get_mlp(self.agent, id, inputs.shape,
                            last_layer=(d_out, "tanh"))
        if brick_type == "multiply":
            brick = get_multiply(self.agent, id, d_out, inputs.shape)
        if brick_type == "swag":
            brick = get_swag(self.agent, id, d_out, inputs.shape)
        if "k_conv" in brick_type:
            brick = KConvSet1D(self.agent, id, d_in, d_out, None)
        # if brick_type == "attention":
        #     brick = MultiHeadAttention(self.agent, id)
        # if brick_type == "dnc":
        #     brick = DNC_RNN(self.agent, id)
        self.G.node[id]['brick_type'] = brick_type
        self.G.node[id]['brick'] = brick
        # log("built a", brick_type)
        return brick

    def build(self, input_shapes):
        """Build the keras model described by a graph."""
        if self.built:
            return self
        self.outputs = [self.get_output(id)
                        for id in list(self.G.predecessors("sink"))]
        self.inputs = [self.G.node[id]['brick']
                       for id in list(self.G.successors('source'))]
        self.model = K.Model(self.inputs, self.outputs)
        self.built = True
        return self

    def call(self, codes):
        log("")
        log("GraphModel call", color="blue")
        log("code spec", self.agent.code_spec, color="blue")
        log(f"got {len(codes)} codes", color="yellow")
        log("")
        return self.model(codes)
