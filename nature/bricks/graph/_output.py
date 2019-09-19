# graph_model.py - bion
import tensorflow as tf
import random

from tools import log
import nature

L = tf.keras.layers

MERGE_SENSORS_WITH = tf.identity


def pick_brick():
    return random.choice([nature.Chain, nature.Stack])


def get_output(G, agent, id):
    node = G.node[id]
    log('get_output', id, node, color="red")
    if node["shape"] is "cylinder":
        return
    if node["output"] is not None:
        return node["output"]
    node_type = node["node_type"]
    brick = MERGE_SENSORS_WITH
    if node_type is "input":
        shape = node["spec"]["shape"]
        node['input'] = inputs = nature.Input(shape, batch_size=agent.batch)
        brick = nature.Sensor(agent.code_spec.shape[-1])
    else:
        inputs = [get_output(G, agent, p) for p in list(G.predecessors(id))]
        if len(inputs) > 1:
            axis = 1
            inputs = L.Concatenate(axis)(inputs)
        else:
            inputs = inputs[0]
        if node_type is 'brick':
            if id is 0:
                brick = nature.Fn(key=None)
            else:
                BRICK = pick_brick()
                if id is 1:
                    BRICK = nature.Slim
                if id is 2:
                    BRICK = nature.Transformer
                if id is 3:
                    BRICK = nature.Recirculator
                if id is 4:
                    BRICK = nature.SWAG
                brick = BRICK(units=agent.code_spec.shape[-1])
        if node_type is "output":
            brick = nature.Actuator(agent, node['spec'])
    output = brick(inputs)
    node["output"] = output
    node["brick"] = brick
    return output
