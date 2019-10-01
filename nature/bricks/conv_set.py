import tensorflow as tf
from tools import make_id
import nature
L = tf.keras.layers

N = [2, 3]


class ConvSet(L.Layer):

    def __init__(self, AI, units=None):
        n = AI.pull(f"conv_set_n", N, id=False)
        super().__init__(name=make_id(f"conv_set_{n}"))
        self.call = self.call_for_two if n is 2 else self.call_for_three
        self.ai = AI

    def build(self, shape):
        self.kernel = nature.FC(units=shape[-1])
        self.concat = L.Concatenate(-1)
        self.sub = L.Subtract()
        super().build(shape)

    @tf.function
    def call_for_two(self, x, training=None):
        return tf.map_fn(
            lambda uno: tf.reduce_sum(
                tf.map_fn(
                    lambda dos: self.kernel(
                            self.concat([uno, dos])
                        ), x), 0), x)

    @tf.function
    def call_for_three(self, x, training=None):
        return tf.map_fn(
            lambda uno: tf.reduce_sum(
                tf.map_fn(
                    lambda dos: tf.reduce_sum(
                        tf.map_fn(
                            lambda tres: self.kernel(
                                self.concat([uno, dos, tres])
                                ), x), 0), x), 0), x)

    def compute_output_shape(self, shape):
        return shape
