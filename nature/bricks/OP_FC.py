# argument for units > size ** 2
# https://arxiv.org/pdf/1902.04674.pdf

# argument for units >= size + 4
# https://arxiv.org/abs/1709.02540

import tensorflow as tf
import nature

L = tf.keras.layers
LAYER = nature.Layer
NORM = nature.NoOp
UNITS = 4


def get_units(shape):
    units = tf.math.reduce_prod(shape) / 5 + 4
    return units


class OP_FC(L.Layer):

    def __init__(self, AI, units=UNITS, layer_fn=LAYER):
        super(OP_FC, self).__init__()
        self.layer_fn = layer_fn
        self.units = units
        self.ai = AI

    def build(self, shape):
        self.layer_1 = self.layer_fn(self.ai, units=get_units(shape))
        self.layer_2 = self.layer_fn(self.ai, units=self.units)
        self.norm_1 = NORM(self.ai)
        self.norm_2 = NORM(self.ai)
        self.built = True

    @tf.function
    def call(self, x):
        x = self.layer_1(x)
        x = self.norm_1(x)
        x = self.layer_2(x)
        x = self.norm_2(x)
        return x
