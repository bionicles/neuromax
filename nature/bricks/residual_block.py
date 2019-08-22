import tensorflow as tf

from nature import use_norm, use_fn, use_conv_2D

K = tf.keras
L = K.layers


LAYER_FN = use_conv_2D
N_LAYERS = 1


def use_residual_block(agent, parts):
    if "layer_fn" not in parts.keys():
        parts.layer_fn = LAYER_FN
    layers = []
    for i in range(N_LAYERS):
        layers.append(use_norm())
        layers.append(use_fn())
        layer = parts.layer_fn(parts.units)
        layers.append(layer)
    parts.adder = adder = L.Add()

    def call(x):
        y = x
        for layer in layers:
            y = layer(y)
        return adder([x, y])
    parts.call = call
    return parts
