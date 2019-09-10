import tensorflow as tf

from nature import FC, Norm
from tools import pipe

L = tf.keras.layers

UNITS = 128


class Add(L.Layer):

    def __init__(self):
        super(Add, self).__init__()

    def build(self, shape):
        d_in = shape[-1]
        self.fc = pipe(FC(units=d_in), Norm())
        self.add = L.Add()
        self.built = True

    @tf.function
    def call(self, x):
        y = self.fc(x)
        y = self.add([x, y])
        return y
