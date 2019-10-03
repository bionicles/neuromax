import tensorflow as tf
import nature as N
L = tf.keras.layers


class Filter(L.Layer):

    def __init__(self, ai):
        super().__init__()
        self.ai = ai

    def build(self, shape):
        self.filter = N.OP(self.ai, units=1)
        super().built(shape)

    def call(self, x):
        d_out = tf.shape(x)[-1]
        y = self.filter(x)
        y = tf.tile(y, [1, 1, d_out])
        return self.multiply([x, y])
