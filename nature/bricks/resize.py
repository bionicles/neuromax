import tensorflow as tf

from nature import Linear
from tools import get_size

K = tf.keras
L = K.layers


class Resizer(L.Layer):
    def __init__(self, out_shape):
        super(Resizer, self).__init__()
        self.flatten = L.Flatten()
        self.resize = Linear(get_size(out_shape))
        self.reshape = L.Reshape(out_shape)
        self.built = True

    def call(self, x):
        x = self.flatten(x)
        x = self.resize(x)
        return self.reshape(x)
