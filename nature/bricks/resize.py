import tensorflow as tf
from tools import get_size
import nature

LAYER = nature.Layer
L = tf.keras.layers


class Resizer(L.Layer):

    def __init__(self, out_shape, fn=None, layer=LAYER):
        super(Resizer, self).__init__()
        self.resize = layer(units=get_size(out_shape))
        self.reshape = L.Reshape(out_shape)
        self.out_shape = out_shape
        self.flatten = L.Flatten()
        self.fn = nature.Fn(key=fn)
        self.built = True

    @tf.function
    def call(self, x):
        x = self.flatten(x)
        x = self.resize(x)
        x = self.reshape(x)
        return x

    def compute_output_shape(self, shape):
        return self.out_shape
