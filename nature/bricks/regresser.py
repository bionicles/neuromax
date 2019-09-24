import tensorflow as tf
import nature

L = tf.keras.layers
ENSEMBLE_SIZE = 5


class Regresser(L.Layer):

    def __init__(self, out_shape):
        super(Regresser, self).__init__()
        self.out_shape = out_shape

    def build(self, shape):
        ensemble_shape = list(self.out_shape)
        ensemble_shape.append(ENSEMBLE_SIZE)
        self.resize = nature.Resizer(self.ensemble_shape)
        self.built = True

    @tf.function
    def call(self, x):
        x = self.resize(x)
        x = self.reduce_mean(x, axis=-1)
        return x
