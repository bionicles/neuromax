import tensorflow as tf

from tools import make_id
import nature

L = tf.keras.layers

LAYER = nature.Layer
N = 16


class SlimDense(L.Layer):

    def __init__(self, units=None):
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
        self.concatenate = L.Concatenate(-1)
        self.np_out = nature.NormPreact()
        self.out = nature.OP_1D(units=self.units)
        self.norm = nature.Norm()
        self.built = True

    @tf.function
    def call(self, x):
        for np, layer in self.blocks:
            y = layer(np(x))
            x = self.concatenate([x, y])
        y = self.np_out(x)
        y = self.out(y)
        y = self.norm(y)
        return y
