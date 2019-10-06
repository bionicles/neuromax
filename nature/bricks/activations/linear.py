import tensorflow as tf

from nature import L1L2

L = tf.keras.layers


class Linear(L.Layer):
    """ y = mx + b
    broadcast scalar weight and bias to all inputs (trainable)
    """

    def __init__(self):
        super(Linear, self).__init__()
        self.m = self.add_weight(
            initializer="ones", regularizer=L1L2(), trainable=True)
        self.b = self.add_weight(
            initializer="glorot_normal", regularizer=L1L2(), trainable=True)

    @tf.function
    def call(self, x):
        return self.m * x + self.b
