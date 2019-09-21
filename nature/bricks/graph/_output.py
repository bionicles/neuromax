# graph_model.py - bion
import tensorflow as tf
import random

from tools import log
import nature

L = tf.keras.layers

MERGE_SENSORS_WITH = tf.identity

bricks = [
    lambda units: nature.Fn(key=None),
    nature.Recirculator,
    nature.Slim,
    nature.SWAG,
    nature.Transformer,
    nature.OP_1D,
    nature.WideDeep,
    nature.MLP,
]

def pick_brick():
    return random.choice([nature.Chain, nature.Stack])


def get_output(G, AI, id):
    node = G.node[id]
    log('get_output', id, node, color="red")
    if node["shape"] is "cylinder":
        return
    if node["output"] is not None:
        return node["output"]
    node_type = node["node_type"]
    brick = MERGE_SENSORS_WITH
    if node_type is "input":
        spec = node['spec']
        shape = spec["shape"]
        node['input'] = inputs = nature.Input(shape, batch_size=AI.batch)
        brick = nature.Sensor(AI, spec)
    else:
        inputs = [get_output(G, AI, p) for p in list(G.predecessors(id))]
        if len(inputs) > 1:
            axis = 1
            inputs = L.Concatenate(axis)(inputs)
        else:
            inputs = inputs[0]
        if node_type is 'brick':
            brick_class = bricks[id] if id <= len(bricks) else pick_brick()
            brick = brick_class(units=AI.code_spec.shape[-1])
        if node_type is "output":
            brick = nature.Actuator(AI, node['spec'])
    output = brick(inputs)
    node["output"] = output
    node["brick"] = brick
    return output
