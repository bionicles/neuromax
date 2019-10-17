import tensorflow as tf
import nature


LAYER = nature.Layer
MIN_LAYERS, MAX_LAYERS = 2, 4
UNITS_OPTIONS = [64, 128, 256, 512]


class MLP(tf.keras.layers.Layer):
    """
    units_list: list of int for units
    fns_list: list of strings for activations (default None)
    """

    def __init__(self, AI, units=None, layer_list=None, layer_fn=LAYER):
        if not layer_list:
            LAYERS = AI.pull('mlp_layers', MIN_LAYERS, MAX_LAYERS)
            UNITS = AI.pull('mlp_units', UNITS_OPTIONS)
            layer_list = [(UNITS, None) for _ in range(LAYERS)]
        super().__init__(f"{LAYERS}_layer_mlp")
        if units:
            layer_list[-1] = (units, layer_list[-1][1])
        self.layers = []
        for i, (units, fn) in enumerate(layer_list):
            fc = nature.Layer(AI, units=units, layer_fn=layer_fn)
            super().__setattr__(f'fc_{i}', fc)
            self.layers.append(fc)
            if fn:
                fn = nature.Fn(AI, key=fn)
                super().__setattr__(f'fn_{i}', fn)
                self.layers.append(fn)
        self.built = True

    @tf.function
    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
