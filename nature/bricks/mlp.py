import tensorflow as tf

from nature import Layer, Fn

K = tf.keras
L = K.layers

LAYERS, UNITS, FN = 2, 8, 'mish'
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

    def __init__(self, units=None, units_list=UNITS_LIST, fns_list=FNS_LIST):
        super(MLP, self).__init__()
        if units:
            units_list[-1] = units
        self.layers = []
        for i, units in enumerate(units_list):
            fc = Layer(units=units)
            super(MLP, self).__setattr__(f'fc_{i}', fc)
            self.layers.append(fc)
            if fns_list[i]:
                fn = Fn(key=fns_list[i])
                super(MLP, self).__setattr__(f'fn_{i}', fn)
                self.layers.append(fn)
        self.built = True

    @tf.function
    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
