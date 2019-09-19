import tensorflow as tf

from tools import get_size
import nature

LAYER = nature.NoiseDrop
L = tf.keras.layers


class Resizer(L.Layer):
    def __init__(self, out_shape, fn=None, layer=LAYER):
        super(Resizer, self).__init__()
        self.out_shape = out_shape
        self.flatten = L.Flatten()
        self.resize = layer(units=get_size(out_shape))
        self.reshape = L.Reshape(out_shape)
        self.built = True

    @tf.function
    def call(self, x):
        x = self.flatten(x)
        x = self.resize(x)
        x = self.reshape(x)
        return x

    def compute_output_shape(self, shape):
        return self.out_shape
