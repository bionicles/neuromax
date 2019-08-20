
# from tools import get_unique_id
# from tools import show_model
# from tools import log
#
# class GraphModel(L.Layer):
#     """
#     GraphModel maps [n_in x code_shape] -> [n_out x code_shape]
#     because we need to be able to handle multiple inputs
#     """
#
#     def __init__(self, agent):
#         super(GraphModel, self).__init__()
#         log("GraphModel init")
#         self.code_shape = agent.code_spec.shape
#         self.batch_size = agent.batch_size
#         self.agent = agent
#         self.pull_numbers = agent.pull_numbers
#         self.pull_choices = agent.pull_choices
#         self.graph = Graph(agent, get_unique_id("GraphModel"))
#         self.G = self.graph.G
#         self.built = False
#         self.build([agent.code_spec.shape for i in range(agent.n_in)])
#         show_model(self.model, ".", "M", "png")
#
#     def get_out(self, id):
#         """
#         Get the output of a node in a computation graph.
#         Pull inputs from predecessors.
#         """
#         node = self.G.node[id]
#         node_type = node["node_type"]
#         if node["out"] is not None:
#             return node["out"]
#         else:
#             if node_type is "input":
#                 out = L.Input(
#                     self.code_shape, batch_size=self.batch_size)
#             else:
#                 parent_ids = list(self.G.predecessors(id))
#                 inputs = [self.get_out(parent_id) for parent_id in parent_ids]
#                 if len(inputs) > 1:
#                     inputs = L.Concatenate(-1)(inputs)
#                 else:
#                     inputs = inputs[0]
#                 inputs = use_norm_preact(self.agent, id, inputs)
#                 d_out = inputs.shape[-1]
#                 brick_type = self.agent.pull_choices(
#                     f"{id}_brick_type", BRICKS)
#                 if brick_type == "residual":
#                     out, brick = use_residual_block(
#                         self.agent, id, inputs, units=d_out, return_brick=True)
#                 if brick_type == "dense":
#                     out, brick = use_dense_block(
#                         self.agent, id, inputs, units=d_out, return_brick=True)
#                 if brick_type == "mlp":
#                     out, brick = use_mlp(
#                         self.agent, id, inputs,
#                         last_layer=(d_out, "tanh"), return_brick=True)
#                 if brick_type == "swag":
#                     out, brick = use_swag(
#                         self.agent, id, inputs, units=d_out, return_brick=True)
#                 self.G.node[id]['brick_type'] = brick_type
#             self.G.node[id]["out"] = out
#             return out
#
#     def build(self, input_shapes):
#         """Build the keras model described by a graph."""
#         if self.built:
#             return self
#         self.outs = [self.get_out(id)
#                      for id in list(self.G.predecessors("sink"))]
#         self.inputs = [self.G.node[id]['brick']
#                        for id in list(self.G.successors('source'))]
#         self.model = K.Model(self.inputs, self.outs)
#         self.built = True
#         return self
#
#     def call(self, codes):
#         log("")
#         log("GraphModel call", color="blue")
#         log("code spec", self.agent.code_spec, color="blue")
#         log(f"got {len(codes)} codes", color="yellow")
#         log("")
#         return self.model(codes)
