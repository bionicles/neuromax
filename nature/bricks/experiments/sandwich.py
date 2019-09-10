import tensorflow as tf
import nature

LAYER = nature.Multiply
UNITS = 32

L = tf.keras.layers

class Sandwich(L.Layer):

    def __init__(self, units=UNITS, layer_fn=LAYER):
        super(Sandwich, self).__init__()
        self.np1 = nature.NormPreact()
        self.layer = nature.Layer(units, layer_fn=layer_fn)
        self.np2 = nature.NormPreact()

    @tf.function
    def call(self, x):
        x = self.np1(x)
        x = self.layer(x)
        x = self.np2(x)
        return x
