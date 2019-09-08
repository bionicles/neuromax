# https://github.com/titu1994/keras-coordconv/blob/master/coord.py
import tensorflow as tf

from tools import normalize, tile_for_batch

K = tf.keras
L, B = K.layers, K.backend


class ConcatCoords2D(L.Layer):
    def __init__(self):
        super(ConcatCoords2D, self).__init__()

    def build(self, shape):
        h = tf.range(shape[1], dtype=tf.float32)
        w = tf.range(shape[2], dtype=tf.float32)
        h, w = normalize(h), normalize(w)
        self.hw = tf.stack(tf.meshgrid(h, w, indexing='ij'), axis=-1)
        self.built = True

    @tf.function
    def call(self, x):
        coords = tile_for_batch(tf.shape(x)[0], self.hw)
        return tf.concat([x, coords], -1)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = output_shape[-1] + 2
        return tuple(output_shape)
