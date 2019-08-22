import tensorflow as tf

from nature import use_dense

K = tf.keras
L = K.layers

POWER = 3
UNITS = 4


def use_swag(agent, parts):
    keys = parts.keys()
    if "power" not in keys:
        parts.power = POWER
    if "units" not in keys:
        parts.units = UNITS

    parts.layers = layers = [
        use_dense(parts.units) for _ in range(parts.power)]
    parts.multiply = multiply = L.Multiply()
    parts.flatten = flatten = L.Flatten()
    parts.concat = concat = L.Concatenate(-1)
    parts.output_layer = output_layer = use_dense(parts.units)

    def call(out):
        outs = []
        for layer in layers:
            out = layer(out)
            if len(outs) > 0:
                out = multiply([outs[-1], out])
            outs.append(out)
        outs = [flatten(o) for o in outs]
        out = concat(outs)
        out = output_layer(out)
        return out
    parts.call = call
    return parts
