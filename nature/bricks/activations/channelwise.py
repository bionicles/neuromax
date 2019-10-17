import tensorflow as tf
from nature import Fn

L = tf.keras.layers

KEY = "logistic"


class Channelwise(L.Layer):

    def __init__(self, AI, key=KEY):
        super().__init__()
        self.keys = key
        self.ai = AI

    def build(self, shape):
        d_in = shape[-1]
        if isinstance(self.keys, str):
            self.keys = [self.keys] * d_in
        else:
            assert len(self.keys) is d_in
        self.split = L.Lambda(lambda x: tf.split(x, d_in, -1))
        self.fns = []
        for k in range(d_in):
            fn = Fn(self.ai, key=self.keys[k])
            super().__setattr__(self, f"f_{k}", fn)
            self.fns.append((k, fn))
        self.concat = L.Concatenate(-1)
        self.built = True

    @tf.function
    def call(self, x):
        channels = self.split(x)
        for k, fn in self.fns:
            channels[k] = fn(channels[k])
        y = self.concat(channels)
        return y