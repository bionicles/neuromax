import tensorflow as tf
import random

from tools import pipe
import nature

K = tf.keras
L = K.layers

LAYER_FN = [nature.AllAttention, nature.Quadratic, nature.FC]
N_LAYERS = random.randint(1, 2)

class ResBlock(L.Layer):

    def __init__(self, layer_fn=LAYER_FN):
        """
        units_or_filters: dimensionality of the layer
        layer_fn: callable returning the layer to use
        fn: string name of the activation
        """
        super(ResBlock, self).__init__()
        self.layer_fn = layer_fn

    def build(self, shape):
        self.layers = []
        for n in range(N_LAYERS):
            norm_preact = nature.NormPreact()
            setattr(self, f"norm_preact_{n}", norm_preact)
            if isinstance(self.layer_fn, list):
                layer_fn = random.choice(self.layer_fn)
            else:
                layer_fn = self.layer_fn
            layer = nature.Layer(
                shape[-1], layer_fn=layer_fn, keepdim=True)
            setattr(self, f"layer_{n}", layer)
            self.layers.append(pipe(norm_preact, layer))
        self.adder = L.Add()
        self.built = True

    @tf.function
    def call(self, x):
        y = tf.identity(x)
        for layer in self.layers:
            y = layer(y)
        y = tf.reshape(y, tf.shape(x))
        y = self.adder([x, y])
        return y
