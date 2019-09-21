import tensorflow as tf
import nature

L = tf.keras.layers
NORM = nature.Norm
CHANNELWISE = False
DEFAULT = "tanh"

class NormPreact(L.Layer):
    def __init__(self, key=DEFAULT, channelwise=CHANNELWISE):
        super(NormPreact, self).__init__()
        layer_fn = nature.Channelwise if channelwise else nature.Fn
        self.fn = layer_fn(key=key)
        self.norm = NORM()
        self.built = True

    @tf.function
    def call(self, x):
        return self.fn(self.norm(x))
