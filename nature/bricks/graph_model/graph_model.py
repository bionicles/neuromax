# graph_model.py - bion
# why?: learn to map N inputs to M outputs with graph GP
import tensorflow as tf

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
        parts = dict(brick_type="input", id=id, inputs=None)
        brick, out = agent.pull_brick(parts, result="both")
    else:
        parent_ids = list(G.predecessors(id))
        inputs = [get_out(parent_id) for parent_id in parent_ids]
        inputs = L.Concatenate(-1)(inputs) if len(inputs) > 1 else inputs[0]
        G.node[id]['inputs'] = inputs
        parts = dict(brick_type="preact_norm", id=id, inputs=inputs)
        out = agent.pull_brick(parts, result="out")
        d_out = inputs.shape[-1]
        brick_type = agent.pull_choices(f"{id}_brick_type", BRICKS)
        parts = dict(brick_type=brick_type, id=id, inputs=inputs, d_out=d_out)
        out, brick = agent.pull_brick(parts, result="brick")
        G.node[id]['brick_type'] = brick_type
    G.node[id]['brick'] = brick
    G.node[id]["out"] = out
    return out


def build(G, agent):
    """Build the keras model described by a networkx graph."""
    outs = [get_out(G, agent, id) for id in list(G.predecessors("sink"))]
    inputs = [G.node[id]['brick'] for id in list(G.successors('source'))]
    return K.Model(inputs, outs)


def use_graph_model(agent, parts):
    """Get parts for a graph_model brick"""
    parts["G"] = Graph(agent, parts)
    parts["model"] = build(parts["G"], agent)
    parts["call"] = parts["model"].call
    return parts
