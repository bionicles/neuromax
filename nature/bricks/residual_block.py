import tensorflow as tf

from nature import Norm, Fn, Layer, Conv2D, Brick

K = tf.keras
L = K.layers

LAYER_FN = Conv2D
UNITS = 32
N_LAYERS = 1
FN = "tanh"


def ResidualBlock(
        agent,
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
        layer = Layer(layer_fn, units_or_filters)
        layers.append((Norm(), Fn(fn), layer))
    adder = L.Add()

    def call(self, x):
        y = tf.identity(x)
        for norm, fn, layer in layers:
            y = layer(y)
        return adder([x, y])
    return Brick(layers, adder, call, agent)
