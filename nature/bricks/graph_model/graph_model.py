# graph_model.py - bion
# why?: learn to map N inputs to M outputs with graph GP
import tensorflow as tf

from nature import use_residual_block, use_dense_block, use_input
from nature import use_norm_preact, use_swag, use_mlp
from tools import make_uuid
from .graph import Graph

K = tf.keras
L = K.layers

BRICKS = ["swag", "mlp", "residual", "dense"]


def get_out(G, agent, id):
    """
    Get the output of a node in a computation graph.
    Pull inputs from predecessors.
    """
    if id is "source":
        return
    node = G.node[id]
    node_type = node["node_type"]
    if node["out"]:
        return node["out"]
    if node_type is "input":
        brick = out = use_input(agent, id, None)
    else:
        parent_ids = list(G.predecessors(id))
        inputs = [get_out(parent_id) for parent_id in parent_ids]
        inputs = L.Concatenate(-1)(inputs) if len(inputs) > 1 else inputs[0]
        G.node[id]['inputs'] = inputs
        out = use_norm_preact(agent, id, inputs)
        d_out = inputs.shape[-1]
        brick_type = agent.pull_choices(f"{id}_brick_type", BRICKS)
        if brick_type == "residual":
            out, brick = use_residual_block(
                agent, id, out, units=d_out, return_brick=True)
        if brick_type == "dense":
            out, brick = use_dense_block(
                agent, id, out, units=d_out, return_brick=True)
        if brick_type == "mlp":
            out, brick = use_mlp(
                agent, id, out, last_layer=(d_out, "tanh"), return_brick=True)
        if brick_type == "swag":
            out, brick = use_swag(
                agent, id, out, units=d_out, return_brick=True)
        G.node[id]['brick_type'] = brick_type
    G.node[id]['brick'] = brick
    G.node[id]["out"] = out
    return out


def build(G, agent):
    """Build the keras model described by a networkx graph."""
    outs = [get_out(G, agent, id) for id in list(G.predecessors("sink"))]
    inputs = [G.node[id]['brick'] for id in list(G.successors('source'))]
    return K.Model(inputs, outs)


def use_graph_model(agent, id, input, return_brick):
    id = make_uuid([id, "graph_model"])
    G = Graph(agent, id)
    model = build(G, agent)
    parts = dict(graph=G, model=model)
    call = model.call
    return agent.pull_brick(parts)
