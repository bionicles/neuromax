import tensorflow as tf
from tools import get_size
import nature

L = tf.keras.layers
UNITS = 4


def get_units(shape):
    d_in = get_size(shape)
    return d_in * d_in + 1


class OP_1D(L.Layer):

    def __init__(self, units=UNITS):
        super(OP_1D, self).__init__()
        self.units = units

    def build(self, shape):
        self.one = nature.Conv1D(units=get_units(shape))
        self.two = nature.Conv1D(units=self.units)
        super().build(shape)

    @tf.function
    def call(self, x):
        return self.two(self.one(x))
