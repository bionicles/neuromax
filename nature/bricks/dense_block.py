import tensorflow as tf
import random

from tools import pipe
import nature

K = tf.keras
L = K.layers

LAYER_FN = [nature.Attention, nature.SWAG, nature.NoiseDrop, nature.Conv1D]
N_LAYERS = random.randint(1, 2)
UNITS = 8


class DenseBlock(L.Layer):
    def __init__(self, layer_fn=LAYER_FN, units=UNITS):
        """
        layer_fn: callable returning layer to use
        units: int growth rate of the channels
        """
        super(DenseBlock, self).__init__()
        self.layers = []
        for n in range(N_LAYERS):
            if isinstance(layer_fn, list):
                layer_fn = random.choice(LAYER_FN)
            else:
                layer_fn = layer_fn
            norm_preact = nature.NormPreact()
            setattr(self, f"norm_preact_{n}", norm_preact)
            layer = nature.Layer(
                units, layer_fn=layer_fn, keepdim=False)
            setattr(self, f"layer_{n}", layer)
            self.layers.append(pipe(norm_preact, layer))
        self.built = True

    @tf.function
    def call(self, x):
        y = tf.identity(x)
        for layer in self.layers:
            y = layer(y)
            y = tf.concat([x, y], axis=-1)
        return y
