import tensorflow as tf
import nature as N
from tools import get_size

L, B = tf.keras.layers, tf.keras.backend


class HyperFC(L.Layer):

    def __init__(self, ai, units, key="identity"):
        super().__init__()
        self.fn = N.Fn(ai, key)
        self.ai = ai

    def build(self, shape):
        dense = L.Dense(units=self.units)
        dense.build(shape)
        k_shape = dense.kernel.shape
        b_shape = dense.bias.shape
        k_size = get_size(k_shape)
        b_size = get_size(b_shape)
        self.hyper_layer = N.Layer(self.ai, units=k_size + b_size)
        self.reshape_kernel = L.Reshape(k_shape)
        self.reshape_bias = L.Reshape(b_shape)
        self.split = L.Lambda(lambda x: tf.split(x, [k_size, b_size]))
        super().build(shape)

    @tf.function
    def call(self, x):
        k, b = self.split(self.hyper_layer(x))
        return tf.nn.bias_add(
                B.batch_dot(x, self.reshape_kernel(k)),
                self.reshape_bias(b)
            )
