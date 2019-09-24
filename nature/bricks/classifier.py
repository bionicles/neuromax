import tensorflow as tf
from tools import get_size
import nature

L = tf.keras.layers
LAYER = L.Dense
FN = 'softmax'


class Classifier(L.Layer):
    def __init__(self, shape):
        super(Classifier, self).__init__()
        self.out_shape = shape

    def build(self, shape):
        self.out = LAYER(units=get_size(self.out_shape), activation=FN)
        self.resize = nature.Resizer(self.out_shape)
        # self.norm = L.BatchNormalization()
        self.built = True

    @tf.function
    def call(self, x):
        # x = self.norm(x)
        x = self.resize(x)
        x = x - tf.math.reduce_max(x, axis=1, keepdims=True)
        x = x / tf.math.reduce_sum(x, axis=1, keepdims=True)
        x = self.out(x)
        return x
