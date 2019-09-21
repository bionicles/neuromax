import tensorflow as tf

from nature import Polynomial, Logistic, Linear

L = tf.keras.layers


class PSwish(L.Layer):
    def __init__(self, layer_fn=Linear):
        super(PSwish, self).__init__()
        self.multiply = L.Multiply()
        self.logistic = Logistic()
        self.linear_or_polynomial = layer_fn()
        self.built = True

    @tf.function
    def call(self, x):
        one = self.linear_or_polynomial(x)
        two = self.logistic(x)
        return self.multiply([one, two])


def PolySwish():
    return PSwish(layer_fn=Polynomial)
