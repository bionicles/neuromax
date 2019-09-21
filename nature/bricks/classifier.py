import tensorflow as tf

from tools import get_size
import nature

L = tf.keras.layers

NORM = L.BatchNormalization
LAYER = L.Dense
FN = 'softmax'
# L1 = 0.1


class Classifier(L.Layer):

    def __init__(self, shape):
        super(Classifier, self).__init__()
        self.out_shape = shape

    def build(self, shape):
        self.norm = NORM()
        self.resize = nature.Resizer(self.out_shape)
        self.out = LAYER(units=get_size(self.out_shape))
        self.fn = nature.Fn(key=FN)
        self.built = True

    # @tf.function
    def call(self, x):
        x = self.norm(x)
        x = self.resize(x)
        x = self.out(x)
        x = x - tf.math.reduce_max(x, axis=1, keepdims=True)
        x = x / tf.math.reduce_sum(x, axis=1, keepdims=True)
        x = self.fn(x)
        return x
