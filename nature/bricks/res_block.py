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
        self.norm = Norm()
        self.fn = Fn(fn)
        self.layer = Layer(layer_fn, units_or_filters)
        self.adder = L.Add()

    def call(self, x):
        x = self.norm(x)
        y = self.fn(x)
        y = self.layer(y)
        y = self.adder([x, y])
        return y
