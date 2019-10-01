import tensorflow as tf
from tools import get_size
import nature

LAYER = nature.Layer
L = tf.keras.layers


class Resizer(L.Layer):

    def __init__(self, AI, out_shape, key=None, layer=None, hyper=None):
        super(Resizer, self).__init__()
        size = get_size(out_shape)
        self.resize = nature.Layer(AI, units=size, layer_fn=layer, hyper=hyper)
        self.reshape = L.Reshape(out_shape)
        self.fn = nature.Fn(AI, key=key)
        self.out_shape = out_shape
        self.flatten = L.Flatten()
        self.built = True

    @tf.function
    def call(self, x):
        x = self.flatten(x)
        x = self.resize(x)
        x = self.reshape(x)
        x = self.fn(x)
        return x

    def compute_output_shape(self, shape):
        return self.out_shape
