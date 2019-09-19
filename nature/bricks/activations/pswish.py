import tensorflow as tf

from nature import Polynomial, Logistic, Linear

L = tf.keras.layers


class PSwish(L.Layer):
    def __init__(self, channelwise=False):
        super(PSwish, self).__init__()
        self.logistic = Logistic()
        self.linear = Linear()
        self.multiply = L.Multiply()
        self.built = True

    @tf.function
    def call(self, x):
        one = self.linear(x)
        two = self.logistic(one)
        y = self.multiply([one, two])
        return y


class PolySwish(L.Layer):
    def __init__(self, channelwise=False):
        super(PolySwish, self).__init__()
        self.logistic = Logistic()
        self.poly = Polynomial()
        self.multiply = L.Multiply()
        self.built = True

    @tf.function
    def call(self, x):
        y = self.logistic(self.poly(x))
        y = self.multiply([x, y])
        return y
