import tensorflow as tf

import nature

L = tf.keras.layers

N_COMPONENTS = 2
LAYER_FN = nature.Quadratic
UNITS = 32

class Mixture(L.Layer):

    def __init__(self, n_components=N_COMPONENTS, layer_fn=LAYER_FN, units=UNITS):
        super(Mixture, self).__init__()
        self.n_components = n_components
        self.layer_fn = layer_fn
        self.units = units

    def build(self, shape):
        self.components = []
        for n in range(self.n_components):
            component = nature.Sandwich(layer_fn=self.layer_fn, units=self.units)
            setattr(self, f"component_{n}", component)
            self.components.append(component)
        self.mix = L.Add()
        self.built = True

    @tf.function
    def call(self, x):
        components = []
        for component in self.components:
            components.append(component(x))
        y = self.mix(components)
        return y
