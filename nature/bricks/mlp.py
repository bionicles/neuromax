import tensorflow as tf

from nature import Linear, Fn

K = tf.keras
L = K.layers

LAYERS, UNITS, FN = 2, 256, None
UNITS_LIST = [UNITS for _ in range(LAYERS)]
FNS_LIST = [FN for _ in range(LAYERS)]


class MLP(L.Layer):
    """ make a sequential MLP... default [256, 256] with no activation
    Args:
        tensor_or_shape: tensor, or tuple
    kwargs:
        units_list: list of int for units
        fns_list: list of strings for activations (default None)
    """

    def __init__(self, units_list=UNITS_LIST, fns_list=FNS_LIST):
        super(MLP, self).__init__()
        self.layers = []
        for i, units in enumerate(units_list):
            dense = Linear(units=units)
            self.layers.append(dense)
            if fns_list[i]:
                self.layers.append(Fn(fns_list[i]))
        self.built = True

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
