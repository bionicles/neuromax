import tensorflow as tf

from nature import FC, Norm
from tools import pipe

K = tf.keras
L, BE = K.layers, K.backend

LAYER = FC
UNITS = 8


class Quadratic(L.Layer):

    def __init__(self, units=UNITS):
        super(Quadratic, self).__init__()
        self.R = pipe(LAYER(units=units), Norm())
        self.G = pipe(LAYER(units=units), Norm())
        self.B = pipe(LAYER(units=units), Norm())
        self.square = L.Lambda(BE.square)
        self.multiply = L.Multiply()
        self.built = True

    @tf.function
    def call(self, x):
        r = self.R(x)
        g = self.G(x)
        b = self.B(self.square(x))
        return self.multiply([r, g]) + b
