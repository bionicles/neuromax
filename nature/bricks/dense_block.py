import tensorflow as tf

from nature import use_norm, use_layer, use_fn, use_conv_2D

K = tf.keras
L = K.layers

DEFAULT_LAYER = use_conv_2D
N_LAYERS = 2


def use_dense_block(parts):
    layers = []
    for i in range(N_LAYERS):
        layer = use_layer(parts.layer_fn, parts.units)
        layers.append((layer, use_fn(), use_norm()))

    def call(x):
        for layer, fn, norm in layers:
            y = layer(fn(norm(x)))
            x = tf.concat([x, y], axis=-1)
        return x
    parts.call = call
    return parts
