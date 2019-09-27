import tensorflow as tf
import nature

L = tf.keras.layers
NORM = nature.Norm
UNITS = 8


class WideDeep(L.Layer):

    def __init__(self, units=UNITS):
        super(WideDeep, self).__init__()
        self.d_out = units

    def build(self, shape):
        self.np1 = nature.NormPreact()
        self.fc1 = nature.NoiseDrop(units=self.d_out)
        self.np2 = nature.NormPreact()
        self.concat = L.Concatenate(1)
        self.out = nature.NoiseDrop(units=self.d_out)
        self.norm = NORM()
        self.built = True

    @tf.function
    def call(self, x):
        deep = self.np1(x)
        deep = self.fc1(deep)
        wide_deep = self.concat([x, deep])
        wide_deep = self.np2(wide_deep)
        wide_deep = self.out(wide_deep)
        return self.norm(wide_deep)
