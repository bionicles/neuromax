import tensorflow as tf

from nature import FC, Norm
from tools import pipe

L = tf.keras.layers

UNITS = 128


class Multiply(L.Layer):

    def __init__(self, units=UNITS):
        super(Multiply, self).__init__()
        self.fc_1 = pipe(FC(units=units), Norm())
        self.fc_2 = pipe(FC(units=units), Norm())
        self.multiply = L.Multiply()
        self.built = True

    @tf.function
    def call(self, x):
        o1 = self.fc_1(x)
        o2 = self.fc_2(x)
        y = self.multiply([o1, o2])
        return y
