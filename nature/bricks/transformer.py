import tensorflow as tf
import nature

L = tf.keras.layers
LAYER = nature.Layer


class Transformer(L.Layer):

    def __init__(self, units=None):
        super(Transformer, self).__init__()
        self.units = units

    def build(self, shape):
        units = shape[-1] if not self.units else self.units
        self.attention = nature.Attention(units=units)
        self.add_norm_1 = nature.AddNorm()
        self.layer = LAYER(units=units)
        self.add_norm_2 = nature.AddNorm()
        self.built = True

    @tf.function
    def call(self, x):
        y = self.attention(x)
        y = self.add_norm_1([x, y])
        z = self.layer(x)
        y = self.add_norm_2([x, y, z])
        return y
