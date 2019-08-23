import tensorflow as tf

from nature import use_norm, use_layer, use_fn, use_conv_2D

K = tf.keras
L = K.layers

DEFAULT_LAYER = use_conv_2D
N_LAYERS = 2
UNITS = 2
FN = "lrelu"


def use_dense_block(
        layer_fn=DEFAULT_LAYER, n_layers=N_LAYERS, units_or_filters=UNITS,
        fn=FN):
    """
    Get a residual block
    kwargs:
        n_layers: number of times to apply this layer
        layer_fn: the layer you'd like to use
        units_or_filters: dimensionality of the layer
        fn: activation to use
    """
    layers = []
    for i in range(N_LAYERS):
        layer = use_layer(layer_fn, units_or_filters)
        layers.append((layer, use_fn(fn), use_norm()))

    def call(x):
        for layer, fn, norm in layers:
            y = norm(x)
            y = fn(x)
            y = layer(x)
            x = tf.concat([x, y], axis=-1)
        return x
    return call
