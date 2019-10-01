import tensorflow as tf
import random
from tools import pipe
import nature

L = tf.keras.layers
LAYER_FN = [nature.Attention, nature.SWAG, nature.NoiseDrop, nature.Conv1D]
LAYERS = [2, 4]
UNITS = [1, 2, 4, 8]


class DenseBlock(L.Layer):

    def __init__(self, AI, layer_fn=LAYER_FN):
        super().__init__()
        n_layers = AI.pull("dense_n_layers", LAYERS, id=False)
        units = AI.pull("dense_units", UNITS, id=False)
        self.d_increase = units * n_layers
        self.concat = L.Concatenate(-1)
        self.layers = []
        for n in range(n_layers):
            if isinstance(layer_fn, list):
                layer_fn = random.choice(layer_fn)
            np = nature.NormPreact(AI)
            super().__setattr__(f"np_{n}", np)
            layer = nature.Layer(AI, units=units, layer_fn=layer_fn)
            super().__setattr__(f"layer_{n}", layer)
            self.layers.append(pipe(np, layer))
        self.built = True

    @tf.function
    def call(self, x):
        for layer in self.layers:
            x = self.concat([x, layer(x)])
        return x
