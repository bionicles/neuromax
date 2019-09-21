import tensorflow as tf

from nature import L1L2

init = tf.keras.initializers.TruncatedNormal


class Polynomial(tf.keras.layers.Layer):

    def __init__(self, power=4):
        super(Polynomial, self).__init__()
        self.powers = []
        for p in list(range(power)):
            coefficient = self.add_weight(
                initializer=init(), trainable=True, regularizer=L1L2())
            super(Polynomial, self).__setattr__(f"{p}", coefficient)
            self.powers.append((coefficient, p))
        self.built = True

    @tf.function
    def call(self, x):
        y = 0.
        for coefficient, power in self.powers:
            y = y + coefficient * tf.math.pow(x, power)
        return y
