import tensorflow as tf

from .layers.dense import get_dense_out
from tools.get_brick import get_brick

K = tf.keras
L = K.layers

MIN_POWER, MAX_POWER = 2, 3
UNITS = 4


def get_swag_out(agent, id, out, units=UNITS, power=None, return_brick=False):
    if power is None:
        power = agent.pull_numbers(f"{id}_swag_power", MIN_POWER, MAX_POWER)
    if "swag" not in id:
        id = f"{id}_swag_{power}"

    layers = []
    for x in range(power):
        _, dense = get_dense_out(
            agent, id, out, units=units, return_brick=True)
        layers.append(dense)
    multiply = L.Multiply()
    flatten = L.Flatten()
    concat = L.Concatenate(-1)
    _, output_layer = get_dense_out(agent, id, out,
                                    units=units, return_brick=True)

    def swag(out):
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
    return get_brick(swag, out, return_brick)
