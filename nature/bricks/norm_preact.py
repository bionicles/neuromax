import tensorflow as tf

from nature import Norm, Fn


class NormPreact(tf.keras.layers.Layer):
    def __init__(self, key=None):
        super(NormPreact, self).__init__()
        self.norm = Norm()
        self.fn = Fn(key=key) if key else Fn()

    def call(self, x):
        return self.fn(self.norm(x))
