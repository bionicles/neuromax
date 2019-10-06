import tensorflow as tf
import nature

L = tf.keras.layers

UNITS = 128


class Add(L.Layer):

    def __init__(self, units=UNITS, axis=None):
        super().__init__()
        self.units = units

    def build(self, shapes):
        self.remotes = []
        for k, shape in enumerate(shapes):
            if shape[-1] is not self.units:
                remote = nature.OP_1D(units=self.units)
                super().__setattr__(f"r_{k}", remote)
            else:
                remote = tf.identity
            self.remotes.append(remote)
        self.add = L.Add()
        super().build(shapes)

    @tf.function
    def call(self, inputs):
        Y = []
        for k, x in enumerate(inputs):
            Y.append(self.remotes[k](x))
        return self.add(Y)
