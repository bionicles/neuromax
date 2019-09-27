import tensorflow as tf
import nature

L = tf.keras.layers
# NORM = L.BatchNormalization
ENSEMBLE_SIZE = 5
FN = None


class Regresser(L.Layer):

    def __init__(self, out_shape):
        super(Regresser, self).__init__()
        self.out_shape = out_shape

    def build(self, shape):
        self.one = nature.MLP()
        # self.norm = NORM()
        ensemble_shape = list(self.out_shape)
        ensemble_shape.append(ENSEMBLE_SIZE)
        self.resize = nature.Resizer(
            ensemble_shape, layer=nature.FC, key=FN)
        super().build(shape)

    @tf.function
    def call(self, x):
        x = self.one(x)
        # x = self.norm(x)
        x = self.resize(x)
        x = tf.reduce_mean(x, axis=-1)
        return x
