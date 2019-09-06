import tensorflow as tf
import numpy as np

from tools import tile_to_batch_size

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.indices.html
# https://github.com/tchaton/CoordConv-keras/blob/master/coord.py#L60
# https://github.com/titu1994/keras-coordconv/blob/master/coord.py
# https://github.com/tensorflow/tensorflow/issues/32221

L = tf.keras.layers


class Coordinator(L.Layer):
    def __init__(self):
        super(Coordinator, self).__init__()
        self.save_coords = True
        self.coords = None

    def build(self, input_shape):
        if None in input_shape[1:-1]:
            self.save_coords = False
        self.get_coords(input_shape)
        self.built = True

    def get_coords(self, input_shape):
        if self.save_coords:
            if self.coords is not None:
                return self.coords
        shape_for_coords = input_shape[1:-1]
        coords = np.indices(shape_for_coords)
        coords = tf.convert_to_tensor(coords, dtype=tf.float32)
        coords = tf.transpose(coords)
        coords = tf.expand_dims(coords, 0)
        if input_shape[0] is not coords.shape[0]:
            coords = tile_to_batch_size(input_shape[0], coords)
        if self.save_coords:
            self.coords = coords
        return coords

    def call(self, x):
        coords = self.get_coords(x.shape)
        return tf.concat([x, coords], -1)
