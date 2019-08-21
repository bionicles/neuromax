import tensorflow as tf

from nature import use_dense
from tools import make_uuid

K = tf.keras
L = K.layers

MIN_POWER, MAX_POWER = 2, 3
UNITS = 4


def use_swag(agent, id, out, units=UNITS, power=None, return_brick=False):
    id = make_uuid([id, "swag"])

    if power is None:
        power = agent.pull_numbers(f"{id}_power", MIN_POWER, MAX_POWER)
    if "swag" not in id:
        id = f"{id}_{power}"

    layers = []
    for x in range(power):
        _, dense = use_dense(agent, id, out, units=units, return_brick=True)
        layers.append(dense)
    multiply = L.Multiply()
    flatten = L.Flatten()
    concat = L.Concatenate(-1)
    _, output_layer = use_dense(agent, id, out, units=units, return_brick=True)
    parts = dict(
        layers=layers, multiply=multiply, flatten=flatten,
        concat=concat, output_layer=output_layer)

    def call(out):
        outs = []
        for i in range(power):
            out = layers[i](agent, id, out, units=units)
            if len(outs) > 0:
                out = multiply([outs[-1], out])
            outs.append(out)
        outs = [flatten(o) for o in outs]
        out = concat(outs)
        out = output_layer(out)
        return out
    return agent.pull_brick(parts)
