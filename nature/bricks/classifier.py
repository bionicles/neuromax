import tensorflow as tf

from tools import get_size
import nature

L = tf.keras.layers


class Classifier(L.Layer):

    def __init__(self, shape):
        super(Classifier, self).__init__()
        self.out = L.Dense(units=get_size(shape))
        self.fn = nature.Fn(key='softmax')
        self.resize = nature.Resizer(shape)
        self.norm_1 = nature.Norm()
        self.built = True

    @tf.function
    def call(self, x):
        x = self.norm_1(x)
        x = self.resize(x)
        x = x - tf.math.reduce_max(x, axis=1, keepdims=True)
        x = x / tf.math.reduce_sum(x, axis=1, keepdims=True)
        x = self.out(x)
        x = self.fn(x)
        return x
