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

    @tf.function
    def call(self, x):
        return self.multiply([
            x, tf.tile(self.filter(x), [1, 1, tf.shape(x)[-1]])])
