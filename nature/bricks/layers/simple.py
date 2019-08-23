import tensorflow as tf

L = tf.keras.layers

INITIAL_M = 0.96
INITIAL_B = 0.04


def use_simple(initial_m=INITIAL_M, initial_b=INITIAL_B):
    """ y = mx + b
    broadcast one weight and one bias to all inputs (trainable)

    kwargs:
        initial_m: float initial m (default 0.96)
        initial_b: float initial b (default 0.04)
    """
    return Simple(initial_m=INITIAL_M, initial_b=INITIAL_B)


class Simple(L.Layer):
    """ y = mx + b
    broadcast one weight and one bias to all inputs (trainable)

    kwargs:
        initial_m: float initial m (default 0.96)
        initial_b: float initial b (default 0.04)
    """

    def __init__(self, initial_m=INITIAL_M, initial_b=INITIAL_B):
        super(Simple, self).__init__()
        self.m = tf.Variable(initial_value=INITIAL_M, trainable=True)
        self.b = tf.Variable(initial_value=INITIAL_B, trainable=True)

    def call(self, x):
        return self.m * x + self.b
