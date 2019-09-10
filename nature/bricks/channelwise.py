import tensorflow as tf

from nature import Fn, Norm

K = tf.keras
L = K.layers

KEY = "logistic"

class Channelwise(L.Layer):

    def __init__(self, key=KEY):
        super(Channelwise, self).__init__()
        self.key = key

    def build(self, shape):
        d_in = shape[-1]
        self.split = L.Lambda(lambda x: tf.split(x, d_in, -1))
        self.fns = []
        for k in range(d_in):
            fn = Fn(key=self.key)
            setattr(self, f"fn_{k}", fn)
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
