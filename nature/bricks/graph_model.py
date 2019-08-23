# graph_model.py - bion
import tensorflow as tf
from tools import get_size
import nature

K = tf.keras
L = K.layers


def get_out(G, agent, id, task_model=False):
    """
    Get the output of a node in a computation graph.
    Pull inputs from predecessors.
    """
    node = G.node[id]
    if node["shape"] is "cylinder":
        return
    node_type = node["node_type"]
    if node["out"]:
        return node["out"]
    if node_type is "input":
        node['input'] = input = out = nature.use_input(node["shape"])
        if task_model:
            coder = nature.use_coder(agent, node['spec'])
            normie, out, reconstruction = coder(input)
            node["reconstruction"] = reconstruction
            node["normie"] = normie
    if node["shape"] is "square" and node["color"] is "black":
        inputs = [get_out(parent_id) for parent_id in list(G.predecessors(id))]
        inputs = L.Concatenate(-1)(inputs) if len(inputs) > 1 else inputs[0]
        if task_model:
            node["code"] = inputs
            brick = use_graph_model(agent)
            node["prediction"] = out = brick(inputs)
        else:
            node['brick'] = brick = nature.use_swag(
                units=get_size(inputs), reshape=inputs.shape)
            out = brick(inputs)
        out = L.Concatenate(-1)([inputs, out])
    elif node_type is "out":
        node['brick'] = brick = nature.use_actuator(agent, node['spec'])
        out = brick(inputs)
    node["out"] = out
    return out


def build_task_model(G, agent):
    """Build the keras model described by a networkx graph."""
    outs = [get_out(G, agent, id, task_model=True)
            for id in list(G.predecessors("sink"))]
    in_nodes = [G.node[id] for id in list(G.successors('source'))]
    code = G.node[0]['code']
    reconstructions = [n['reconstruction'] for n in in_nodes]
    normies = [n['normie'] for n in in_nodes]
    prediction = G.node[0]['prediction']
    outputs = normies + code + reconstructions + prediction + outs
    inputs = [n['input'] for n in in_nodes]
    n = ['normie'] * len(normies)
    c = ['code']
    r = ['reconstruction'] * len(reconstructions)
    p = ["prediction"]
    o = ['out'] * len(outs)
    roles = n + c + r + p + o
    return K.Model(inputs, outputs), roles


def build(G, agent):
    outs = [get_out(G, agent, id) for id in list(G.predecessors("sink"))]
    inputs = [G.node[id]['input'] for id in list(G.successors('source'))]
    return K.Model(inputs, outs)


def use_graph_model(agent, in_specs=None, out_specs=None):
    G = nature.Graph(agent, in_specs=in_specs, out_specs=out_specs)
    if in_specs and out_specs:
        return build_task_model(G, agent)
    return build(G, agent)
