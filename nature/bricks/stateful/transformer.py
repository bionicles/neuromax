import tensorflow as tf
import nature


class Transformer(tf.keras.layers.Layer):

    def __init__(self, AI, units=None):
        super(Transformer, self).__init__()
        self.units = units
        self.ai = AI

    def build(self, shape):
        self.attention = nature.Attention(self.ai, units=shape[-1])
        self.add_norm = nature.AddNorm()
        # self.layer = LAYER(units=units)
        # self.add_norm_2 = nature.AddNorm()
        super().build(shape)

    @tf.function
    def call(self, x):
        y = self.attention(x)
        y = self.add_norm([x, y])
        # z = self.layer(x)
        # y = self.add_norm_2([y, z])
        return y
