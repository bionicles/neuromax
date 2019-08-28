# graph_model.py - bion
import tensorflow as tf

from tools import log
import nature

K = tf.keras
L = K.layers


def get_out(G, agent, id, task_model=False):
    """
    Get the output of a node in a computation graph.
    Pull inputs from predecessors.
    """
    node = G.node[id]
    log(node, color="green", debug=True)
    if node["shape"] is "cylinder":
        return
    node_type = node["node_type"]
    if node["out"] is not None:
        return node["out"]
    if node_type is "input":
        drop_batch_dim = not task_model
        node['input'] = input = out = nature.use_input(
            node["spec"]["shape"],
            batch_size=agent.batch_size, drop_batch_dim=drop_batch_dim)
        if task_model:
            coder = nature.use_coder(agent, node['spec'])
            normie, out, reconstruction = coder(input)
            node["reconstruction"] = reconstruction
            node["normie"] = normie
    else:
        inputs = [get_out(G, agent, parent_id, task_model=task_model)
                  for parent_id in list(G.predecessors(id))]
        inputs = L.Concatenate(-1)(inputs) if len(inputs) > 1 else inputs[0]
        brick = None
        if node_type is 'brick':
            log('inputs', inputs, color="red")
            if task_model:
                node["code"] = inputs
                brick = use_graph_model(agent, inputs=inputs)
                node["prediction"] = out = brick(inputs)
            else:
                d_out = inputs.shape[-1]
                brick = nature.use_residual_block(
                    units_or_filters=d_out, layer_fn=nature.use_linear)
                out = brick(inputs)
            log('out', out, color="red")
            out = L.Concatenate(-1)([inputs, out])
        if node_type is "out":
            if task_model:
                brick = nature.use_actuator(agent, node['spec'])
                out = brick(inputs)
            else:
                out = inputs
        node["brick"] = brick
    node["out"] = out
    return out


def build_task_model(G, agent):
    """Build the keras model described by a networkx graph."""
    log("build task model", color="red")
    outs = [get_out(G, agent, id, task_model=True)
            for id in list(G.predecessors("sink"))]
    in_nodes = [G.node[id] for id in list(G.successors('source'))]
    code = [G.node[0]['code']]
    reconstructions = [n['reconstruction'] for n in in_nodes]
    normies = [n['normie'] for n in in_nodes]
    prediction = [G.node[0]['prediction']]
    outputs = normies + code + reconstructions + prediction + outs
    inputs = [n['input'] for n in in_nodes]
    n = ['normie'] * len(normies)
    c = ['code']
    r = ['reconstruction'] * len(reconstructions)
    p = ["prediction"]
    o = ['action'] * len(outs)
    roles = n + c + r + p + o
    return K.Model(inputs, outputs), roles


def build(G, agent):
    log("build graph_model brick", color="red")
    outs = [get_out(G, agent, id) for id in list(G.predecessors("sink"))]
    inputs = [G.node[id]['input'] for id in list(G.successors('source'))]
    return K.Model(inputs, outs)


def use_graph_model(agent, in_specs=None, out_specs=None, inputs=None):
    graph = nature.Graph(
        agent, in_specs=in_specs, out_specs=out_specs, inputs=inputs)
    G = graph.G
    if in_specs and out_specs:
        return build_task_model(G, agent)
    return build(G, agent)
