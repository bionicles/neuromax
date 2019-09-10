import tensorflow as tf

from nature import Norm, Fn, PSwish

DEFAULT = "tanh"

class NormPreact(tf.keras.layers.Layer):
    def __init__(self, key=DEFAULT):
        super(NormPreact, self).__init__()
        self.norm = Norm()
        if key is "pswish":
            self.norm2 = Norm()
            self.pswish = PSwish()
            self.fn = lambda x: self.norm2(self.pswish(x))
        else:
            self.fn = Fn(key=key)

    def call(self, x):
        return self.fn(self.norm(x))
