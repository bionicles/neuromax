# https://github.com/titu1994/keras-coordconv/blob/master/coord.py
import tensorflow as tf

from tools import normalize, log

K, L, B = tf.keras, tf.keras.layers, tf.keras.backend


def Coordinator(shape):
    log("build coordinator with shape:", shape)
    if len(shape) is 2:
        return ConcatCoords2D()
    elif len(shape) is 3:
        return ConcatCoords3D()
    elif len(shape) is 4:
        return ConcatCoords4D()
    else:
        raise Exception(f"{shape} not supported by coordinator")


class ConcatCoords2D(L.Layer):
    def __init__(self):
        super(ConcatCoords2D, self).__init__()
        self.handler = ConcatCoords3D()
        self.built = True

    @tf.function
    def call(self, x):
        x = tf.expand_dims(x, -1)
        return self.handler(x)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape.append(1)
        output_shape[-1] = output_shape[-1] + 1
        return tuple(output_shape)


class ConcatCoords3D(L.Layer):
    def __init__(self):
        super(ConcatCoords3D, self).__init__()
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


class ConcatCoords4D(L.Layer):
    def __init__(self):
        super(ConcatCoords4D, self).__init__()

    def build(self, shape):
        h = tf.range(shape[1], dtype=tf.float32)
        w = tf.range(shape[2], dtype=tf.float32)
        h, w = normalize(h), normalize(w)
        hw = tf.stack(tf.meshgrid(h, w, indexing='ij'), axis=-1)
        hw = tf.expand_dims(hw, 0)
        self.hw = tf.tile(hw, [shape[0], 1, 1, 1])
        super().build(shape)

    @tf.function
    def call(self, x):
        return tf.concat([x, self.hw], -1)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = output_shape[-1] + 2
        return tuple(output_shape)
