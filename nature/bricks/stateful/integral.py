import tensorflow as tf
L = tf.keras.layers


class Integral(L.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def build(self, shape):
        self.state = self.add_weight("state", shape, trainable=False)
        super().build(shape)

    @tf.function
    def call(self, x):
        self.state.assign_add(x)
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
