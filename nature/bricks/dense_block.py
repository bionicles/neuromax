import tensorflow as tf
import random

from tools import pipe
import nature

K = tf.keras
L = K.layers

LAYER_FN = [nature.Attention, nature.SWAG, nature.NoiseDrop, nature.Conv1D]
N_LAYERS = 2
UNITS = 4


class DenseBlock(L.Layer):

    def __init__(self, layer_fn=LAYER_FN, units=UNITS):
        super(DenseBlock, self).__init__()
        self.concat = L.Concatenate(-1)
        self.units = units
        self.layers = []
        for n in range(N_LAYERS):
            if isinstance(layer_fn, list):
                layer_fn = random.choice(LAYER_FN)
            np = nature.NormPreact()
            super(DenseBlock, self).__setattr__(f"np_{n}", np)
            layer = nature.Layer(units, layer_fn=layer_fn)
            super(DenseBlock, self).__setattr__(f"layer_{n}", layer)
            self.layers.append(pipe(np, layer))
        self.built = True

    @tf.function
    def call(self, x):
        y = x
        for layer in self.layers:
            y = self.concat([y, layer(y)])
        return y
