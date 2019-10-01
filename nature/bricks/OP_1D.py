import tensorflow as tf

from tools import next_power_of_two_past
import nature

L = tf.keras.layers

UNITS = 4


def get_units(shape):
    d_in = shape[-1]
    return next_power_of_two_past(d_in * d_in + 4)


class OP_1D(L.Layer):

    def __init__(self, AI, units=UNITS):
        super(OP_1D, self).__init__()
        self.d_out = units
        self.ai = AI

    def build(self, shape):
        self.layer_1 = nature.Conv1D(units=get_units(shape))
        self.layer_2 = nature.Conv1D(units=self.d_out)
        super().build(shape)

    @tf.function
    def call(self, x):
        return self.layer_2(self.layer_1(x))
