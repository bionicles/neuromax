# argument for units > size ** 2
# https://arxiv.org/pdf/1902.04674.pdf

# argument for units >= size + 4
# http://bjlkeng.github.io/posts/universal-resnet-the-one-neuron-approximator/
# https://arxiv.org/abs/1709.02540

import tensorflow as tf
import nature

L = tf.keras.layers
LAYER = nature.NoiseDrop
NORM = nature.NoOp
UNITS = 4


def get_units(shape):
    size = tf.math.reduce_prod(shape)
    return size + 4


class OP_FC(L.Layer):

    def __init__(self, units=UNITS, layer_fn=LAYER):
        super(OP_FC, self).__init__()
        self.layer_fn = layer_fn
        self.units = units

    def build(self, shape):
        self.layer_1 = self.layer_fn(units=get_units(shape))
        self.layer_2 = self.layer_fn(units=self.units)
        self.norm_1 = NORM()
        self.norm_2 = NORM()
        self.built = True

    @tf.function
    def call(self, x):
        x = self.layer_1(x)
        x = self.norm_1(x)
        x = self.layer_2(x)
        x = self.norm_2(x)
        return x
