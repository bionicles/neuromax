import tensorflow as tf
import nature

L = tf.keras.layers
LAYER = nature.FC

LAYERS, UNITS, FN = 2, 256, "mish"
UNITS_LIST = [UNITS for _ in range(LAYERS)]
FNS_LIST = [FN for _ in range(LAYERS)]


class MLP(L.Layer):
    """
    units_list: list of int for units
    fns_list: list of strings for activations (default None)
    """

    def __init__(self, units=None, units_list=UNITS_LIST, fns_list=FNS_LIST):
        super().__init__()
        if units:
            units_list[-1] = units
        self.layers = []
        for i, units in enumerate(units_list):
            fc = LAYER(units=units)
            super().__setattr__(f'fc_{i}', fc)
            self.layers.append(fc)
            if fns_list[i]:
                fn = nature.Fn(key=fns_list[i])
                super().__setattr__(f'fn_{i}', fn)
                self.layers.append(fn)
        self.built = True

    @tf.function
    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
