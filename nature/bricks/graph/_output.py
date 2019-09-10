# graph_model.py - bion
import tensorflow as tf
import random

from nature import Input, Predictor, Fn, Brick, Actuator
from tools import log, pipe, stack

K = tf.keras
L = K.layers


def get_output(G, agent, id):
    node = G.node[id]
    log('get_output', id, node, color="red")
    if node["shape"] is "cylinder":
        return
    if node["output"] is not None:
        return node["output"]
    node_type = node["node_type"]
    if node_type is "input":
        node['input'] = inputs = Input(
            node["spec"]["shape"], batch_size=agent.batch)
        brick = Predictor(out_shape=agent.code_spec.shape)
    else:
        inputs = [get_output(G, agent, p) for p in list(G.predecessors(id))]
        inputs = L.Concatenate(-1)(inputs) if len(inputs) > 1 else inputs[0]
        if node_type is 'brick':
            brick = Fn(key=None) if id is 0 else Brick(agent.code_spec.shape)
        if node_type is "output":
            brick = Actuator(agent, node['spec'])
    output = brick(inputs)
    node["output"] = output
    node["brick"] = brick
    return output
