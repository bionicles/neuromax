import tensorflow as tf
from tools import tile_for_batch
L = tf.keras.layers


class Integral(L.Layer):

    def __init__(self, *args, **kwargs):
        super(Integral, self).__init__()

    def build(self, shape):
        self.state = tf.zeros(shape)
        self.batch = shape[0]
        self.built = True

    @tf.function
    def call(self, x):
        mean = tf.math.reduce_mean(x, axis=0)
        mean = tile_for_batch(self.batch, mean)
        self.state = self.state + mean
        return self.state

#
# class IntegralCell(L.Layer):
#
#     def __init__(self, units):
#         super(IntegralCell, self).__init__()
#         self.state_size = units
#
#     def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
#         return tf.zeros_like(inputs)
#
#     def build(self, shape):
#         self.batch_size = shape[0]
#         self.built = True
#
#     @tf.function
#     def call(self, x, states):
#         tf.print("x shape", tf.shape(x), "state shape", tf.shape(states[0]))

#         sum = states[0] + sum
#         return sum, [sum]
