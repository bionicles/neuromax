import tensorflow as tf

L = tf.keras.layers


class Derivative(L.Layer):

    def __init__(self, *args, **kwargs):
        super(Derivative, self).__init__()

    def build(self, shape):
        self.state = tf.zeros(shape)
        self.built = True

    @tf.function
    def call(self, x):
        derivative = self.state - x
        self.state = tf.identity(x)
        return derivative
