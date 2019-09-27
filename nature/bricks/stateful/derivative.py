import tensorflow as tf

L = tf.keras.layers


class Derivative(L.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def build(self, shape):
        self.state = self.add_weight("state", shape, trainable=False)
        super().build(shape)

    @tf.function
    def call(self, x):
        derivative = self.state - x
        self.state.assign(x)
        return derivative
