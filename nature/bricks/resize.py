import tensorflow as tf

from nature import Quadratic, Norm
from tools import get_size

K = tf.keras
L = K.layers


class Resizer(L.Layer):
    def __init__(self, out_shape, fn=None):
        super(Resizer, self).__init__()
        self.out_shape = out_shape
        self.flatten = L.Flatten()
        self.resize = Quadratic(get_size(out_shape))
        self.reshape = L.Reshape(out_shape)
        self.norm = Norm()
        self.built = True

    @tf.function
    def call(self, x):
        x = self.flatten(x)
        x = self.resize(x)
        x = self.reshape(x)
        x = self.norm(x)
        return x

    def compute_output_shape(self, shape):
        return self.out_shape
