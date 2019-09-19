import tensorflow as tf

from tools import next_power_of_two_past
from nature import Conv1D, Norm

L = tf.keras.layers

UNITS = 4

class OP_1D(L.Layer):

    def __init__(self, units=UNITS):
        super(OP_1D, self).__init__()
        self.d_out = units

    def build(self, shape):
        d_in = shape[-1]
        units = d_in * d_in + 8
        units = next_power_of_two_past(units)
        self.layer_1 = Conv1D(units=units)
        self.norm_1 = Norm()
        self.layer_2 = Conv1D(units=self.d_out)
        self.norm_2 = Norm()
        self.built = True

    @tf.function
    def call(self, x):
        x = self.layer_1(x)
        x = self.norm_1(x)
        x = self.layer_2(x)
        x = self.norm_2(x)
        return x
