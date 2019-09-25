import tensorflow as tf

from tools import make_id
import nature

L = tf.keras.layers
LAYER = nature.Layer
NORM = nature.NoOp
N = 16


class Slim(L.Layer):

    def __init__(self, units=None):
        super(Slim, self).__init__(name=make_id(f"slim_{N}"))

    def build(self, shape):
        self.blocks = []
        for n in range(N):
            np = nature.NormPreact()
            units = shape[-1] if (n + 1) is N else 1
            layer = LAYER(units=units)
            super(Slim, self).__setattr__(f"np_{n}", np)
            super(Slim, self).__setattr__(f"l_{n}", layer)
            self.blocks.append((np, layer))
        self.norm = NORM()
        self.add = L.Add()
        self.built = True

    @tf.function
    def call(self, x):
        x = self.norm(x)
        for np, layer in self.blocks:
            y = layer(np(x))
            x = self.add([x, y])
        y = self.norm(y)
        return y
