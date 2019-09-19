import tensorflow as tf

from tools import make_id
import nature

L = tf.keras.layers

LAYER = nature.Layer
UNITS = 3
N = 8


class Slim(L.Layer):

    def __init__(self, units=UNITS):
        name = make_id(f"slim_{N}")
        super(Slim, self).__init__(name=name)
        self.units = units

    def build(self, shape):
        self.blocks = []
        for n in range(N):
            np = nature.NormPreact()
            layer = LAYER(units=1)
            super(Slim, self).__setattr__(f"np_{n}", np)
            super(Slim, self).__setattr__(f"l_{n}", layer)
            self.blocks.append((np, layer))
        self.norm = nature.Norm()
        self.add = L.Add()
        self.built = True

    @tf.function
    def call(self, x):
        for np, layer in self.blocks:
            y = layer(np(x))
            x = self.add([x, y])
        y = self.norm(x)
        return y
