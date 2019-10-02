# argument for units > size ** 2
# https://arxiv.org/pdf/1902.04674.pdf

# argument for units >= size + 4
# https://arxiv.org/abs/1709.02540

import tensorflow as tf
import nature

L = tf.keras.layers
UNITS = 4


def get_units(shape):
    # units = tf.math.reduce_prod(shape) / 5 + 4
    return 64


class OP_FC(L.Layer):

    def __init__(self, units=UNITS):
        super(OP_FC, self).__init__()
        self.units = units

    def build(self, shape):
        self.one = nature.FC(units=get_units(shape))
        self.two = nature.FC(units=self.units)
        super().build(shape)

    @tf.function
    def call(self, x):
        return self.two(self.one(x))
