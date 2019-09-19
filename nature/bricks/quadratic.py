import tensorflow as tf

from nature import NoiseDrop, Norm
from tools import pipe

K = tf.keras
L, BE = K.layers, K.backend

LAYER = NoiseDrop
UNITS = 16


class Quadratic(L.Layer):

    def __init__(self, units=UNITS, layer=LAYER):
        super(Quadratic, self).__init__()
        self.layer = layer
        self.units = units

    def build(self, shape):
        units = self.units if self.units else shape[-1]
        self.R = self.layer(units=units)
        self.G = self.layer(units=units)
        self.B = self.layer(units=units)
        self.square = L.Lambda(BE.square)
        self.multiply = L.Multiply()
        self.norm = Norm()
        self.built = True

    @tf.function
    def call(self, x):
        r = self.R(x)
        g = self.G(x)
        b = self.B(self.square(x))
        y = self.multiply([r, g]) + b
        y = self.norm(y)
        return y
