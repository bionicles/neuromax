import tensorflow as tf
from tools import get_size
import nature

L, B = tf.keras.layers, tf.keras.backend
INIT = nature.Init()


class Projector(L.Layer):

    def __init__(self, units, init=INIT, trainable=False):
        assert units
        super().__init__(name=f"{units}D_projector")
        self.trainable = trainable
        self.units = units
        self.init = init

    def build(self, shape):
        dense = L.Dense(self.units)
        dense.build([shape[0], get_size(shape[1:])])
        self.kernel = self.add_weight(
            "kernel", dense.kernel.shape, initializer=self.init,
            trainable=self.trainable)
        self.flatten = L.Flatten()
        super().build(shape)

    @tf.function
    def call(self, x):
        return B.dot(self.flatten(x), self.kernel)
