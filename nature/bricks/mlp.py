import tensorflow as tf

from nature import use_dense, use_input, swish
K = tf.keras

UNITS, FN, LAYERS = 256, swish, 2

BRICK_TYPE = "mlp"


def use_mlp(agent, parts):
    # we design layers if we didn't get them
    if parts.layer_list is None:
        layer_list = [(UNITS, FN) for _ in range(LAYERS)]
        if parts.last_layer is not None:
            layer_list[-1] = parts.last_layer
    # we make parts
    parts.layers = [use_input(parts.inputs)]
    for units, fn in layer_list:
        dense = use_dense(units)
        parts.layers.append(dense)
    parts.model = K.Sequential(parts.layers)
    parts.call = parts.model.call
    return parts
