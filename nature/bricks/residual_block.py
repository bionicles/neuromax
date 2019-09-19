import tensorflow as tf
import random

from tools import make_id
import nature

L = tf.keras.layers

OPTIONS = [nature.Recirculator]
N = 1

class ResBlock(L.Layer):

    def __init__(self, layer_fn=OPTIONS, n=N):
        """
        layer_fn: callable returning the layer to use
        """
        if isinstance(layer_fn, list):
            layer_fn = random.choice(layer_fn)
        key = make_id(f"{layer_fn.__name__}_block")
        super(ResBlock, self).__init__(name=key)
        self.layer_fn = layer_fn
        self.n = N

    def build(self, shape):
        self.layers = []
        for n in range(self.n):
            norm_preact = nature.NormPreact()
            layer = self.layer_fn(units=shape[-1])
            super(ResBlock, self).__setattr__(f"np_{n}", norm_preact)
            super(ResBlock, self).__setattr__(f"l_{n}", layer)
            self.layers.append((norm_preact, layer))
        # self.norm_preact_2 = nature.NormPreact()
        # self.layer_2 = self.layer_fn(units=shape[-1])
        self.add_norm = nature.AddNorm()
        self.built = True

    @tf.function
    def call(self, x):
        y = tf.identity(x)
        for np, l in self.layers:
            y = l(np(y))
        # y = self.norm_preact_1(y)
        # y = self.layer_1(y)
        # y = self.norm_preact_2(y)
        # y = self.layer_2(y)
        y = tf.reshape(y, tf.shape(x))
        y = self.add_norm([x, y])
        return y
