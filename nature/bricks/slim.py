# http://bjlkeng.github.io/posts/universal-resnet-the-one-neuron-approximator/

import tensorflow as tf
from tools import make_id
import nature

L = tf.keras.layers
LAYER = nature.Layer
N_OPTIONS = [16, 32, 64]


class Slim(L.Layer):

    def __init__(self, AI, units=None):
        self.N = AI.pull("slim_n", N_OPTIONS, id=False)
        super().__init__(name=make_id(f"slim_{self.N}"))
        self.ai = AI

    def build(self, shape):
        self.zero = LAYER(self.ai, units=1)
        self.f0 = nature.Fn(self.ai)
        self.blocks = []
        for n in range(self.N):
            units = shape[-1] if (n + 1) is self.N else 1
            np = nature.NormPreact(self.ai)
            layer = LAYER(self.ai, units=units)
            super().__setattr__(f"np_{n}", np)
            super().__setattr__(f"l_{n}", layer)
            self.blocks.append((np, layer))
        self.add = L.Add()
        super().build(shape)

    @tf.function
    def call(self, x):
        x = self.zero(self.f0(x))
        for np, layer in self.blocks:
            y = layer(np(x))
            x = self.add([x, y])
        return y
