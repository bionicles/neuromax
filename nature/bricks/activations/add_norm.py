import tensorflow as tf
import nature

L = tf.keras.layers
NORM = nature.Norm


class AddNorm(L.Layer):

    def __init__(self):
        super(AddNorm, self).__init__()
        self.add = L.Add()
        self.norm = NORM()
        self.built = True

    @tf.function
    def call(self, x):
        return self.norm(self.add(x))
