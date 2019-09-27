import tensorflow as tf


class ConcatNumber(tf.keras.layers.Layer):

    def __init__(self, number):
        super().__init__()
        self.number = number

    def build(self, shape):
        shape = list(shape)
        shape[-1] = 1
        ones = tf.ones(shape, tf.float32)
        self.number = self.number * ones
        super().build(shape)

    @tf.function
    def call(self, x):
        return tf.concat([x, self.number], -1)
