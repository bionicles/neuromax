import tensorflow as tf

from nature import Norm, Fn, Layer, Conv2D

K = tf.keras
L = K.layers

LAYER_FN = Conv2D
UNITS = 32
N_LAYERS = 1
FN = "tanh"


class ResBlock(L.Layer):
    def __init__(
            self,
            units_or_filters=UNITS, n_layers=N_LAYERS,
            layer_fn=LAYER_FN, fn=FN):
        """
        Get a res block
        kwargs:
            units_or_filters: dimensionality of the layer
            layer_fn: the layer you'd like to use
            n_layers: number of times to apply this layer
            fn: string name of the activation
        """
        super(ResBlock, self).__init__()
        self.layers = []
        for i in range(n_layers):
            layer = Layer(layer_fn, units_or_filters)
            self.layers.append((Norm(), Fn(fn), layer))
        self.adder = L.Add()

    def call(self, x):
        y = tf.identity(x)
        for norm, fn, layer in self.layers:
            y = layer(y)
        return self.adder([x, y])
