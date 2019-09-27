import tensorflow as tf
from tools import log, pipe
import nature

L = tf.keras.layers


def get_output(G, AI, id):
    node = G.node[id]
    log('get output for', node, color="red")
    if node["shape"] is "cylinder":
        return
    if node["output"] is not None:
        return node["output"]
    node_type = node["node_type"]
    brick = tf.identity
    if node_type is "input":
        spec = node['spec']
        node['input'] = inputs = nature.Input(spec, batch_size=AI.batch)
        inputs = L.BatchNormalization(name=f"n_{id[6:]}")(inputs)
        brick = nature.Sensor(AI, spec)
    else:
        inputs = [get_output(G, AI, p) for p in list(G.predecessors(id))]
        if len(inputs) > 1:
            if node_type is 'add':
                inputs = nature.Add(units=AI.code_spec.shape[-1])(inputs)
            else:
                inputs = nature.Merge(AI.code_spec.shape)(inputs)
        else:
            inputs = inputs[0]
        if node_type in ["merge", 'add']:
            brick = pipe(nature.Regularizer(), L.BatchNormalization())
        if node_type is 'brick':
            brick_class = nature.Brick(id)
            brick = brick_class(units=AI.code_spec.shape[-1])
        if node_type in ["output", 'critic']:
            brick = nature.Actuator(AI, node['spec'])
    output = brick(inputs)
    node["output"] = output
    node["brick"] = brick
    return output
