import tensorflow as tf
import nature as N
L = tf.keras.layers


class Template(L.Layer):

    def __init__(self, ai):
        super().__init__()
        self.ai = ai

    def build(self, shape):
        super().build(shape)

    @tf.function
    def call(self, x):
        return x
