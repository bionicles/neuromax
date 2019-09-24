import tensorflow as tf
from nature import Resizer
from tools import log, tile_for_code
L = tf.keras.layers


class Merge(L.Layer):

    def __init__(self, code_shape):
        super(Merge, self).__init__()
        self.code_shape = code_shape
        self.d_code = code_shape[-1]

    def build(self, shapes):
        self.merge = L.Concatenate(1)
        self.tiler = L.Lambda(
            lambda x: tf.tile(tf.expand_dims(x, -1), [1, 1, self.d_code]))
        self.built = True

    @tf.function
    def call(self, x):
        items = []
        for item in x:
            if len(tf.shape(item)) is 2:
                item = self.tiler(item)
            items.append(item)
        y = self.merge(items)
        return y
