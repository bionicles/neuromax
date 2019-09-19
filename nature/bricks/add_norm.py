import tensorflow as tf
import nature

L = tf.keras.layers

class AddNorm(L.Layer):

    def __init__(self):
        super(AddNorm, self).__init__()
        self.add = L.Add()
        self.norm = nature.Norm()
        self.built = True

    def call(self, x):
        x = self.add(x)
        x = self.norm(x)
        return x
