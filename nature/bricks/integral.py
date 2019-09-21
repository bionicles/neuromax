import tensorflow as tf
from tools import tile_for_batch

L = tf.keras.layers


def Integral(units=UNITS):
    return L.RNN(IntegralCell)


class IntegralCell(L.Layer):

    def __init__(self):
        super(IntegralCell, self).__init__()

    def build(self, shape):
        self.batch_size = shape[0]
        self.output_size = shape
        self.state_size = shape
        self.built = True

    @tf.function
    def call(self, x, states):
        sum = tf.math.reduce_sum(x, axis=0, keepdims=True)
        sum = tile_for_batch(self.batch_size, sum)
        sum = states[0] + sum
        return sum, [sum]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.zeros_like(inputs)
