import tensorflow as tf

from nature import use_norm, use_fn, use_layer, use_conv_2D

K = tf.keras
L = K.layers

LAYER_FN = use_conv_2D
UNITS = 32
N_LAYERS = 1
FN = "lrelu"


def use_residual_block(
        units_or_filters=UNITS, n_layers=N_LAYERS, layer_fn=LAYER_FN, fn=FN):
    """
    Get a residual block
    kwargs:
        units_or_filters: dimensionality of the layer
        layer_fn: the layer you'd like to use
        n_layers: number of times to apply this layer
        fn: string name of the activation
    """
    layers = []
    for i in range(n_layers):
        layer = use_layer(layer_fn, units_or_filters)
        layers.append(use_norm(), use_fn(fn), layer)
    adder = L.Add()

    def call(x):
        y = tf.identity(x)
        for norm, fn, layer in layers:
            y = layer(y)
        return adder([x, y])
    return call
