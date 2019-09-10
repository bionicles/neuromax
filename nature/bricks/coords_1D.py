import tensorflow as tf

from tools import tile_for_batch, normalize

L = tf.keras.layers


class ConcatCoords1D(L.Layer):
    def __init__(self):
        super(ConcatCoords1D, self).__init__()
        self.built = True

    @tf.function
    def call(self, x):
        shape = tf.shape(x)
        coords = tf.range(shape[1])
        coords = tf.expand_dims(coords, 0)
        coords = tf.expand_dims(coords, -1)
        coords = tf.tile(coords, [shape[0], 1, 1])
        coords = tf.cast(coords, tf.float32)
        coords = normalize(coords)
        return tf.concat([x, coords], -1)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = output_shape[-1] + 1
        return tuple(output_shape)
