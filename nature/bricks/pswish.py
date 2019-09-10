import tensorflow as tf

from nature import Channelwise, Logistic, Linear

L = tf.keras.layers

class PSwish(L.Layer):
    def __init__(self, channelwise=False):
        super(PSwish, self).__init__()
        if channelwise:
            self.logistic = Channelwise(key="logistic")
            self.linear = Channelwise(key="linear")
        else:
            self.logistic = Logistic()
            self.linear = Linear()
        self.multiply = L.Multiply()
        self.built = True

    @tf.function
    def call(self, x):
        one = self.logistic(x)
        two = self.linear(x)
        y = self.multiply([one, two])
        return y
