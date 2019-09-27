from tools import map_sets_tf
import tensorflow as tf
import nature
L = tf.keras.layers
CONCAT_AXIS = 1
POOL_AXIS = 1
N = 3


class ConvSet(L.Layer):
    def __init__(self, n=N):
        super(ConvSet, self).__init__()
        self.kernel = Kernel()
        self.built = True
        self.n = n

    @tf.function
    def op(self, set, i):
        inputs = tf.concat([*set, i], CONCAT_AXIS)
        return self.kernel(inputs)

    @tf.function
    def call(self, x):
        return tf.map_fn(
            lambda i: map_sets_tf(
                x, self.n, lambda set: self.op(set, i),
                lambda y: tf.reduce_sum(y, POOL_AXIS)), x)


class Kernel(L.Layer):
    def __init__(self):
        super(Kernel, self).__init__()

    def build(self, shape):
        self.layer = nature.Resize(shape)
        self.built = True

    @tf.function
    def call(self, x):
        return self.layer(x)
