import tensorflow as tf
from tools import get_size
import nature

L = tf.keras.layers
KEY = "identity"


class Resizer(L.Layer):

    def __init__(self, AI, out_shape, key=KEY):
        super(Resizer, self).__init__()
        size = get_size(out_shape)
        self.resize = nature.FC(units=size)
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
        return self.fn(x)

    def compute_output_shape(self, shape):
        return self.out_shape
