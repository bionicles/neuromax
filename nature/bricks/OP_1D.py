import tensorflow as tf

from tools import next_power_of_two_past
from nature import Conv1D, Norm

L = tf.keras.layers

NORM = Norm
UNITS = 4

def GET_UNITS(shape):
    d_in = shape[-1]
    return next_power_of_two_past(d_in * d_in + 4)

class OP_1D(L.Layer):

    def __init__(self, units=UNITS):
        super(OP_1D, self).__init__()
        self.d_out = units

    def build(self, shape):
        self.layer_1 = Conv1D(units=GET_UNITS(shape))
        self.layer_2 = Conv1D(units=self.d_out)
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
