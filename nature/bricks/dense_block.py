import tensorflow as tf

from nature import Norm, Layer, Fn, Conv2D

K = tf.keras
L = K.layers

DEFAULT_LAYER = Conv2D
N_LAYERS = 2
UNITS = 2
FN = "mish"


class DenseBlock(L.Layer):
    def __init__(
            self, layer_class=DEFAULT_LAYER,
            n_layers=N_LAYERS, units_or_filters=UNITS,
            fn=FN):
        """
        Get a residual block
        kwargs:
            n_layers: number of times to apply this layer
            layer_fn: the layer you'd like to use
            units_or_filters: dimensionality of the layer
            fn: activation to use
        """
        super(DenseBlock, self).__init__()
        self.layers = []
        for i in range(N_LAYERS):
            layer = Layer(layer_class, units_or_filters)
            self.layers.append((layer, Fn(fn), Norm()))
        self.built = True

    def call(self, x):
        for layer, fn, norm in self.layers:
            y = norm(x)
            y = fn(x)
            y = layer(x)
            x = tf.concat([x, y], axis=-1)
        return x
