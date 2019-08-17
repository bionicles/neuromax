# graph_model.py - bion
# why?: learn to map N inputs to M outputs with graph GP
import tensorflow_probability as tfp
# import tensorflow_addons as tfa
import tensorflow as tf
from nature.bricks.graph_model.graph import Graph
from nature.bricks.conv_1D import get_conv_1D
from nature.bricks.k_conv import KConvSet1D
from nature.bricks.get_mlp import get_mlp
# from nature.bricks.kernel import get_kernel

from tools.get_unique_id import get_unique_id
from tools.show_model import show_model
from tools.log import log

# InstanceNormalization = tfa.layers.InstanceNormalization
tfpl = tfp.layers
K = tf.keras
L = K.layers

# bricks
BRICK_OPTIONS = ["conv1d", "mlp"]


class GraphModel(L.Layer):
    """
    GraphModel maps [n_in x code_shape] -> [n_out x code_shape]
    because we need to be able to handle multiple inputs
    """

    def __init__(self, agent):
        super(GraphModel, self).__init__()
        log("\nGraphModel.__init__")
        self.agent = agent
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        self.graph = Graph(agent, get_unique_id("GraphModel"))
        self.G = self.graph.G
        self.build([agent.code_spec.shape for i in range(agent.n_in)])
        show_model(self.model, ".", "M", "png")

    def build(self, input_shapes):
        """Build the keras model described by a graph."""
        self.outputs = [self.get_output(id)
                        for id in list(self.G.predecessors("sink"))]
        self.inputs = [self.G.node[id]['brick']
                       for id in list(self.G.successors('source'))]
        self.model = K.Model(self.inputs, self.outputs)
        return self
        # self.__call__ = self.model.__call__

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
                inputs = L.Concatenate(1)(inputs) if len(inputs) > 1 else inputs[0]
                if node_type is "recurrent":
                    output = inputs
                else:
                    brick = self.build_brick(id, inputs)
                    log(f"got brick", brick)
                    output = brick(inputs)
                    if "output_shape" not in keys and "gives_feedback" not in keys:
                        try:
                            output = L.Add()([inputs, output])
                        except Exception as e:
                            log("error adding inputs to output", e, color="red")
                        try:
                            output = L.Concatenate(1)([inputs, output])
                        except Exception as e:
                            log("error concatenating inputs to output", e, color="red")
            else:
                output = self.build_brick(id)
            # if node_type is not "input":
            #     try:
            #         output = InstanceNormalization()(output)
            #     except Exception as e:
            #         log(f"can't use instance norm\n", e)
            self.G.node[id]["output"] = output
            log(f"got {node_type} node output", output)
            return output

    def build_brick(self, id, inputs=None):
        """Make the keras operation to be executed at a given node."""
        node = self.G.node[id]
        log('build brick for', node)
        if node["node_type"] == "input":
            brick = L.Input(self.agent.code_spec.shape,
                            batch_size=self.agent.batch_size)
            brick_type = "input"
        else:
            d_in = d_out = inputs.shape[-1]
            brick_type = self.agent.pull_choices(
                f"{id}_brick_type", BRICK_OPTIONS)
            log("building a ", brick_type, "brick")
        if brick_type == "conv1d":
            brick = get_conv_1D(self.agent, id, d_out)
        if brick_type == "mlp":
            brick = get_mlp(inputs.shape, last_layer=(d_out, "tanh"))
        if "k_conv" in brick_type:
            brick = KConvSet1D(self.agent, id, d_in, d_out, None)
        # if brick_type == "attention":
        #     brick = MultiHeadAttention(self.agent, id)
        # if brick_type == "dnc":
        #     brick = DNC_RNN(self.agent, id)
        self.G.node[id]['brick_type'] = brick_type
        self.G.node[id]['brick'] = brick
        log("built a", brick_type)
        return brick

    def call(self, codes):
        print("")
        log("GraphModel.__call__", color="blue")
        log(self.agent.code_spec, color="blue")
        [log(c, color="yellow") for c in codes]
        return self.model(codes)
