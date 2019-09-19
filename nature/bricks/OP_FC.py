# argument for units > size ** 2
# https://arxiv.org/pdf/1902.04674.pdf

# argument for units >= size + 4
# http://bjlkeng.github.io/posts/universal-resnet-the-one-neuron-approximator/
# https://arxiv.org/abs/1709.02540

import tensorflow as tf

from tools import next_power_of_two_past
from nature import NoiseDrop, Norm

L = tf.keras.layers

LAYER = NoiseDrop
UNITS = 4

def get_units(size):
    return size + 4
    # return size * size + 2

class OP_FC(L.Layer):

    def __init__(self, units=UNITS, layer_fn=LAYER):
        super(OP_FC, self).__init__()
        self.layer_fn = layer_fn
        self.units = units

    def build(self, shape):
        size = tf.math.reduce_prod(shape)
        units = get_units(size)
        self.layer_1 = self.layer_fn(units=units)
        self.norm_1 = Norm()
        self.layer_2 = self.layer_fn(units=self.units)
        self.norm_2 = Norm()
        self.built = True

    @tf.function
    def call(self, x):
        x = self.layer_1(x)
        x = self.norm_1(x)
        x = self.layer_2(x)
        x = self.norm_2(x)
        return x
