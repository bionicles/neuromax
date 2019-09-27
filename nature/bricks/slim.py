import tensorflow as tf

from tools import make_id
import nature

L = tf.keras.layers
LAYER = nature.Layer
N = 8


class Slim(L.Layer):

    def __init__(self, units=None):
        super().__init__(name=make_id(f"slim_{N}"))

    def build(self, shape):
        self.blocks = []
        for n in range(N):
            np = nature.NormPreact()
            units = shape[-1] if (n + 1) is N else 1
            layer = LAYER(units=units)
            super().__setattr__(f"np_{n}", np)
            super().__setattr__(f"l_{n}", layer)
            self.blocks.append((np, layer))
        self.add = L.Add()
        super().build(shape)

    @tf.function
    def call(self, x):
        for np, layer in self.blocks:
            y = layer(np(x))
            x = self.add([x, y])
        return y
